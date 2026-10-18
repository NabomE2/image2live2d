"""Durable, alpha-only editor for locally generated rigs. No inference dependencies.

Published revisions are immutable. All writes build in private directories; the atomic
project manifest update is the commit point. POSIX flock serializes writers across workers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from ..backends.live2d.bundle import bundle_zip, validate_bundle
from ..backends.live2d.moc3_binary import write_moc3
from ..backends.live2d.moc3_emit import build_atlas, rig_to_moc3


class Conflict(ValueError):
    """The approved draft no longer matches the current revision."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_tree(root):
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        sync_dir(path)
    sync_dir(root)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid editor identifier")
    return value


def relative(root, name):
    path = (root / name).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError("Invalid asset reference")
    return path


@contextmanager
def locked(project):
    # App runs in Linux Docker/WSL or POSIX. Do not silently omit process locking.
    try:
        import fcntl
    except ImportError as exc:
        raise ValueError("Editor locking requires Docker/WSL, Linux or macOS") from exc

    with (project / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def brush(image, original, stroke):
    """One bounded dab, in source pixels; RGB and pixels outside its radius are untouched."""
    if set(stroke) != {"x", "y", "radius", "hardness", "mode"}:
        raise ValueError("Invalid brush fields")
    x, y, radius, hardness = (stroke[k] for k in ("x", "y", "radius", "hardness"))
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in (x, y, radius, hardness)):
        raise ValueError("Brush values must be finite numbers")
    if not (0 <= x <= image.width and 0 <= y <= image.height
            and 0.5 <= radius <= 256 and 0 <= hardness <= 1):
        raise ValueError("Brush outside allowed range")
    if stroke["mode"] not in {"erase", "restore"}:
        raise ValueError("Only erase and original restore are supported")
    a = np.array(image)
    base = np.asarray(original)
    x0, y0 = max(0, math.floor(x-radius)), max(0, math.floor(y-radius))
    x1, y1 = min(image.width, math.ceil(x+radius)), min(image.height, math.ceil(y+radius))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.sqrt((xx+.5-x)**2 + (yy+.5-y)**2) / radius
    weight = ((distance <= 1).astype(float) if hardness == 1 else
              np.clip((1-distance) / max(1-hardness, 1e-9), 0, 1))
    alpha = a[y0:y1, x0:x1, 3].astype(float)
    ceiling = base[y0:y1, x0:x1, 3]
    target = 0 if stroke["mode"] == "erase" else ceiling
    a[y0:y1, x0:x1, 3] = np.minimum(ceiling, np.rint(alpha+(target-alpha)*weight)).astype("uint8")
    return Image.fromarray(a)


class EditorStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def project(self, pid):
        p = self.root / identifier(pid)
        if not (p / "project.json").is_file():
            raise FileNotFoundError("Editor project not found")
        return p

    def list(self):
        return [read(p) for p in sorted(self.root.glob("*/project.json"))
                if re.fullmatch(r"[a-f0-9]{32}", p.parent.name)]

    def info(self, pid):
        p = self.project(pid)
        doc = read(p / "project.json")
        applied = {r.get("draft") for r in doc["revisions"]}
        doc["drafts"] = [read(d) for d in sorted((p / "drafts").glob("*/draft.json"))
                         if d.parent.name not in applied]
        return doc

    def create(self, rig, assets, bundle, source, filename, normalized):
        """Snapshot only after byte-checking reconstructed moc3 and atlas against the bundle."""
        assets, bundle, normalized = Path(assets), Path(bundle), Path(normalized)
        if not normalized.is_dir():
            raise ValueError("Normalized snapshot missing; generate a new job with this editor version")
        rig = rig.model_copy(update={"meta": rig.meta.model_copy(update={"name": "model"})})
        atlas, remap = build_atlas(rig, assets)
        manifest = bundle / "model.model3.json"
        files = validate_bundle(manifest)
        refs = read(manifest)["FileReferences"]
        if refs["Textures"] != ["textures/atlas.png"]:
            raise ValueError("Only the generated single-atlas layout is editable")
        with Image.open(bundle / refs["Textures"][0]) as actual:
            if not np.array_equal(np.asarray(actual.convert("RGBA")), np.asarray(atlas)):
                raise ValueError("Atlas cannot be reproduced safely; original export remains unchanged")
        if write_moc3(rig_to_moc3(rig, atlas_uv=remap)) != relative(bundle, refs["Moc"]).read_bytes():
            raise ValueError("Rig/UV/keyforms do not match exported moc3; editing refused")
        pid, rid = uuid.uuid4().hex, uuid.uuid4().hex
        temp = self.root / (".building-" + pid)
        temp.mkdir()
        try:
            original = temp / "original"
            (original / "parts").mkdir(parents=True)
            shutil.copytree(normalized, original / "normalized")
            (original / "source").write_bytes(source)
            atomic_json(original / "source.json", {"filename": filename})
            atomic_json(original / "rig.json", rig.model_dump(mode="json"))
            (original / "bundle").mkdir()
            for rel in files:
                dest = original / "bundle" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(relative(bundle, rel), dest)
            tex = {t.id: t for t in rig.textures}
            parts, layout = [], {}
            used = []
            for i, part in enumerate(rig.parts_in_draw_order()):
                texture = tex[part.texture_id]
                path = relative(assets, texture.path)
                asset = f"{i:03d}.png"
                shutil.copyfile(path, original / "parts" / asset)
                with Image.open(path) as im:
                    width, height = im.size
                editable = part.semantic_role.value == "mouth" and part.id in remap
                parts.append({"id": part.id, "role": part.semantic_role.value, "asset": asset,
                              "width": width, "height": height, "editable": editable,
                              "texture_id": texture.id, "original_texture_path": texture.path})
                if part.id not in remap:
                    continue
                c = remap[part.id]
                src = [round(v*n) for v, n in zip(c[:4], (width, height, width, height))]
                dst = [round(v*n) for v, n in zip(c[4:], (*atlas.size, *atlas.size))]
                if not (0 <= dst[0] < dst[2] <= atlas.width and
                        0 <= dst[1] < dst[3] <= atlas.height):
                    raise ValueError("Invalid atlas tile")
                for b in used:
                    if min(b[2], dst[2]) > max(b[0], dst[0]) and min(b[3], dst[3]) > max(b[1], dst[1]):
                        raise ValueError("Overlapping atlas tiles cannot be edited safely")
                used.append(dst)
                layout[part.id] = {"source": src, "tile": dst, "uv_remap": c}
            atomic_json(original / "layout.json", layout)
            hashes = {str(p.relative_to(original)): digest(p)
                      for p in original.rglob("*") if p.is_file()}
            atomic_json(temp / "original_hashes.json", hashes)
            rev = temp / "revisions" / rid
            shutil.copytree(original / "parts", rev / "parts")
            shutil.copytree(original / "bundle", rev / "bundle")
            atomic_json(rev / "revision.json", {"id": rid, "parent": None, "kind": "original"})
            (temp / "drafts").mkdir()
            doc = {"schema": 1, "id": pid, "name": Path(filename).stem, "current": rid,
                   "parts": parts, "revisions": [read(rev / "revision.json")],
                   "moc_reuse": "byte-verified", "runtime_quality": "unverified"}
            atomic_json(temp / "project.json", doc)
            sync_tree(temp)
            os.replace(temp, self.root / pid)
            sync_dir(self.root)
            return doc
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def _original_valid(self, p):
        for name, expected in read(p / "original_hashes.json").items():
            if digest(relative(p / "original", name)) != expected:
                raise ValueError("Original snapshot integrity check failed")

    def _revision(self, p, rid):
        if rid not in {r["id"] for r in read(p / "project.json")["revisions"]}:
            raise ValueError("Unknown revision")
        return p / "revisions" / identifier(rid)

    def _part(self, p, part_id):
        return next((x for x in read(p / "project.json")["parts"] if x["id"] == part_id), None)

    def draft(self, pid, base, part_id):
        p = self.project(pid)
        with locked(p):
            if read(p / "project.json")["current"] != base:
                raise Conflict("Current revision changed; reload before editing")
            part = self._part(p, part_id)
            if not part or not part["editable"]:
                raise ValueError("Only mouth alpha is editable")
            rev = self._revision(p, base)
            did = uuid.uuid4().hex
            d = p / "drafts" / did
            d.mkdir()
            shutil.copyfile(rev / "parts" / part["asset"], d / "0.png")
            doc = {"id": did, "base": base, "part_id": part_id, "sequence": 0}
            atomic_json(d / "draft.json", doc)
            return doc

    def strokes(self, pid, did, sequence, strokes):
        if not isinstance(strokes, list) or not 1 <= len(strokes) <= 256:
            raise ValueError("Expected 1 to 256 brush dabs")
        p = self.project(pid)
        with locked(p):
            d = p / "drafts" / identifier(did)
            doc = read(d / "draft.json")
            applied = any(r.get("draft") == did for r in read(p / "project.json")["revisions"])
            if doc["sequence"] != sequence or "revision" in doc or applied:
                raise Conflict("Draft changed or already applied")
            part = self._part(p, doc["part_id"])
            with Image.open(d / f"{sequence}.png") as src:
                image = src.convert("RGBA")
            with Image.open(p / "original" / "parts" / part["asset"]) as src:
                original = src.convert("RGBA")
            for stroke in strokes:
                image = brush(image, original, stroke)
            doc["sequence"] += 1
            image.save(d / f"{doc['sequence']}.png")
            with (d / f"{doc['sequence']}.png").open("rb") as stream:
                os.fsync(stream.fileno())
            atomic_json(d / "draft.json", doc)
            return doc

    def asset(self, pid, part_id, *, revision=None, draft=None):
        p = self.project(pid)
        part = self._part(p, part_id)
        if not part:
            raise ValueError("Unknown part")
        if draft:
            d = p / "drafts" / identifier(draft)
            doc = read(d / "draft.json")
            if doc["part_id"] != part_id:
                raise ValueError("Draft belongs to another part")
            return d / f"{doc['sequence']}.png"
        root = p / "original" if revision == "original" else self._revision(p, revision)
        return root / "parts" / part["asset"]

    def _publish(self, p, doc, temp, rid, entry):
        validate_bundle(temp / "bundle/model.model3.json")
        atomic_json(temp / "revision.json", entry)
        sync_tree(temp)
        os.replace(temp, p / "revisions" / rid)
        sync_dir(p / "revisions")
        # Only revisions named in project.json are visible, including after an interrupted commit.
        doc["revisions"].append(entry)
        doc["current"] = rid
        atomic_json(p / "project.json", doc)
        return doc

    def apply(self, pid, did, sequence, approved):
        if approved is not True:
            raise ValueError("Explicit approval is required")
        p = self.project(pid)
        with locked(p):
            self._original_valid(p)
            d = p / "drafts" / identifier(did)
            draft = read(d / "draft.json")
            doc = read(p / "project.json")
            # Transaction identity survives a crash between publication and draft receipt.
            prior = next((r for r in doc["revisions"] if r.get("draft") == did), None)
            if prior:
                if prior["sequence"] != sequence:
                    raise Conflict("Approval targets a different draft sequence")
                return doc
            if draft["base"] != doc["current"] or draft["sequence"] != sequence:
                raise Conflict("Revision or draft changed; compare and approve again")
            base = self._revision(p, draft["base"])
            part = self._part(p, draft["part_id"])
            with Image.open(d / f"{sequence}.png") as src:
                edited = src.convert("RGBA")
            with Image.open(p / "original/parts" / part["asset"]) as src:
                original = src.convert("RGBA")
            a, b = np.asarray(edited), np.asarray(original)
            if a.shape != b.shape or not np.array_equal(a[..., :3], b[..., :3]) or np.any(a[..., 3] > b[..., 3]):
                raise ValueError("Only alpha reductions within the original image are supported")
            rid = uuid.uuid4().hex
            temp = p / (".building-" + rid)
            try:
                shutil.copytree(base, temp)
                edited.save(temp / "parts" / part["asset"])
                layout = read(p / "original/layout.json")[part["id"]]
                atlas_path = temp / "bundle/textures/atlas.png"
                with Image.open(atlas_path) as src:
                    atlas = np.array(src.convert("RGBA"))
                with Image.open(p / "original/bundle/textures/atlas.png") as src:
                    original_atlas = np.asarray(src.convert("RGBA"))
                x0, y0, x1, y1 = layout["tile"]
                alpha = edited.getchannel("A").crop(layout["source"])
                alpha = alpha.resize((x1-x0, y1-y0), Image.Resampling.LANCZOS)
                atlas[y0:y1, x0:x1, 3] = np.minimum(np.asarray(alpha), original_atlas[y0:y1, x0:x1, 3])
                Image.fromarray(atlas).save(atlas_path)
                # All non-atlas files must still be byte-identical to the original bundle.
                for name, expected in read(p / "original_hashes.json").items():
                    if (name.startswith("bundle/") and name != "bundle/textures/atlas.png"
                            and digest(temp / name) != expected):
                        raise ValueError("Geometry or bundle changed; moc3 reuse refused")
                entry = {"id": rid, "parent": draft["base"], "kind": "mouth_alpha",
                         "draft": did, "sequence": sequence, "part_id": part["id"]}
                result = self._publish(p, doc, temp, rid, entry)
                draft["revision"] = rid
                atomic_json(d / "draft.json", draft)
                return result
            finally:
                if temp.exists():
                    shutil.rmtree(temp)

    def restore(self, pid, target, base, approved):
        if approved is not True:
            raise ValueError("Explicit approval is required")
        p = self.project(pid)
        with locked(p):
            self._original_valid(p)
            doc = read(p / "project.json")
            if doc["current"] != base:
                raise Conflict("Current revision changed")
            source = self._revision(p, target)
            rid = uuid.uuid4().hex
            temp = p / (".building-" + rid)
            try:
                shutil.copytree(source, temp)
                return self._publish(p, doc, temp, rid,
                                     {"id": rid, "parent": base, "kind": "restore", "target": target})
            finally:
                if temp.exists():
                    shutil.rmtree(temp)

    def download(self, pid, revision):
        p = self.project(pid)
        self._original_valid(p)
        return bundle_zip(self._revision(p, revision) / "bundle/model.model3.json")
