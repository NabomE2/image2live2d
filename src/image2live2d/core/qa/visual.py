"""Opt-in, CPU-only stage diagnostics; never edits source art or inference settings.

Run ``python -m image2live2d.core.qa.visual --help``. Pixel differences are observations,
not proof of decomposition defects. The diagnostic renderer is not Cubism Core.
"""
from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from ...irr.schema import Rig


def triangle_samples(points, width, height):
    """Pixel-centre barycentric samples, with no edge dilation (unlike preview rendering)."""
    p = np.asarray(points, dtype=float)
    lo = np.maximum(0, np.floor(p.min(axis=0)).astype(int))
    hi = np.minimum([width, height], np.ceil(p.max(axis=0)).astype(int))
    if np.any(hi <= lo):
        return None
    x, y = np.meshgrid(np.arange(lo[0], hi[0]) + .5, np.arange(lo[1], hi[1]) + .5)
    a, b, c = p
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-12:
        return None
    wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
    wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
    weights = np.stack([wa, wb, 1 - wa - wb], axis=-1)
    inside = (weights >= -1e-9).all(axis=-1)
    return (slice(lo[1], hi[1]), slice(lo[0], hi[0])), weights, inside


def mesh_coverage(image, mesh, threshold=8):
    """Return counts and an overlay: red = supported alpha outside all UV triangles."""
    rgba = np.asarray(image.convert("RGBA"))
    h, w = rgba.shape[:2]
    covered = np.zeros((h, w), dtype=bool)
    uv = np.asarray(mesh.uvs) * [w, h]
    for tri in mesh.triangles:
        sample = triangle_samples(uv[list(tri)], w, h)
        if sample is not None:
            region, _, inside = sample
            covered[region] |= inside
    support = rgba[..., 3] >= threshold
    missing = support & ~covered
    count = int(support.sum())
    overlay = rgba.copy()
    overlay[missing] = [255, 0, 0, 255]
    return {"alpha_pixels": count, "missing_pixels": int(missing.sum()),
            "coverage": float((support & covered).sum() / count) if count else None,
            "triangles": len(mesh.triangles)}, Image.fromarray(overlay)


def save_views(image, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGBA")
    image.save(directory / "rgba.png")
    image.convert("RGB").save(directory / "rgb.png")  # includes hidden RGB intentionally
    image.getchannel("A").save(directory / "alpha.png")
    for name, color in (("white", "white"), ("black", "black")):
        Image.alpha_composite(Image.new("RGBA", image.size, color), image).save(
            directory / f"on_{name}.png")


def difference(before, after, directory):
    """No implicit registration: mismatched sizes must be investigated, not stretched away."""
    if before.size != after.size:
        return {"status": "unaligned", "before_size": before.size, "after_size": after.size}
    a, b = (np.asarray(im.convert("RGBA"), dtype=float) for im in (before, after))
    # Premultiply so hidden RGB cannot masquerade as a visible defect.
    pa = a[..., :3] * a[..., 3:4] / 255
    pb = b[..., :3] * b[..., 3:4] / 255
    delta = np.maximum(np.max(np.abs(pa - pb), axis=-1), np.abs(a[..., 3] - b[..., 3]))
    Path(directory).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(delta)).save(Path(directory) / "difference.png")
    return {"status": "compared", "changed_pixels": int((delta > 1).sum()),
            "mean_visible_difference": float(delta.mean())}


def _keyforms(param, value):
    forms = sorted(param.keyforms, key=lambda k: k.value)
    if not forms:
        return []
    if value <= forms[0].value:
        return [(forms[0], 1.)]
    if value >= forms[-1].value:
        return [(forms[-1], 1.)]
    for a, b in pairwise(forms):
        if a.value <= value <= b.value:
            t = (value - a.value) / (b.value - a.value)
            return [(a, 1-t), (b, t)]
    return []


def render_reference(rig, images, size, settings=None):
    """Neutral/blink reference: per-vertex offsets + opacity, nearest texels, once per part.

    No physics, group rotation or Cubism warp evaluation. Only neutral and eye-open
    sweeps are supported by the report. A triangle shared edge never double-composites.
    """
    settings = settings or {}
    canvas = Image.new("RGBA", size)
    w, h = size
    for part in rig.parts_in_draw_order():
        mesh = rig.mesh_for(part.id)
        if mesh is None:
            continue
        verts = np.asarray(mesh.vertices).copy()
        opacity = part.opacity
        for param in rig.parameters:
            forms = _keyforms(param, settings.get(param.id, param.default))
            for form, weight in forms:
                if part.id in form.mesh_offsets:
                    verts += np.asarray(form.mesh_offsets[part.id]) * weight
            if any(part.id in form.opacity_overrides for form, _ in forms):
                opacity *= sum(form.opacity_overrides.get(part.id, 1.) * weight
                               for form, weight in forms)
        dest = verts * [w, -h] + [0, h]
        tex = np.asarray(images[part.id].convert("RGBA"))
        th, tw = tex.shape[:2]
        uv = np.asarray(mesh.uvs) * [tw, th]
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        for tri in mesh.triangles:
            sample = triangle_samples(dest[list(tri)], w, h)
            if sample is None:
                continue
            region, weights, inside = sample
            coords = weights @ uv[list(tri)]
            xx = np.clip(np.floor(coords[..., 0]).astype(int), 0, tw-1)
            yy = np.clip(np.floor(coords[..., 1]).astype(int), 0, th-1)
            layer[region][inside] = tex[yy[inside], xx[inside]]
        layer[..., 3] = np.uint8(layer[..., 3] * np.clip(opacity, 0, 1))
        canvas = Image.alpha_composite(canvas, Image.fromarray(layer))
    return canvas


def duplicate_candidates(face, feature):
    """Conservative report-only dark RGB agreement; cannot prove semantic duplication."""
    if face.size != feature.size:
        return None
    a, b = (np.asarray(im.convert("RGBA"), dtype=np.int16) for im in (face, feature))
    mask = ((a[..., 3] >= 64) & (b[..., 3] >= 64)
            & (b[..., :3].max(axis=-1) < 120)
            & (np.abs(a[..., :3] - b[..., :3]).max(axis=-1) <= 24))
    overlay = np.asarray(feature.convert("RGBA")).copy()
    overlay[mask] = [255, 0, 255, 255]
    return int(mask.sum()), Image.fromarray(overlay)


def generate_report(rig, asset_root, output, *, original=None, layerdiff=None, psd=None,
                    atlas=None, uv_remap=None):
    """Collect actual input stages and evaluate existing meshes. Missing stages stay unknown.

    If no atlas is supplied, rebuild one with the production packer and label it rebuilt.
    Source inputs are read-only. Output must be a new directory to prevent stale reports.
    """
    from ...backends.live2d.moc3_emit import build_atlas
    from ..decompose.sources import _raw_layers_from_psd, raws_to_stack

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(asset_root)
    textures = {t.id: t for t in rig.textures}
    images = {}
    report = {"schema": 1, "limits": [
        "CPU reference, not Cubism Core/VTube Studio; GPU unverified.",
        "Differences/duplicate candidates are not proof of a defect; no automatic art removal.",
        "PSD leaf extraction follows the importer; group effects are not independently rendered.",
        "LayerDiff PNGs are unordered snapshots, not a depth-ordered neutral pose."],
        "mesh": {}, "comparisons": {}, "duplicate_candidates": [], "stages": {}}
    for index, part in enumerate(rig.parts_in_draw_order()):
        with Image.open(root / textures[part.texture_id].path) as src:
            im = src.convert("RGBA")
        images[part.id] = im
        folder = output / "parts" / f"{index:03d}"
        save_views(im, folder)
        mesh = rig.mesh_for(part.id)
        if mesh:
            stats, overlay = mesh_coverage(im, mesh)
            stats["directory"] = str(folder.relative_to(output))
            report["mesh"][part.id] = stats
            overlay.save(folder / "missing_alpha.png")
    (output / "rig.json").write_text(rig.model_dump_json(indent=2), encoding="utf-8")
    report["stages"]["parts"] = "provided rig textures and meshes"
    size = next(iter(images.values())).size if images else (512, 512)
    neutral = render_reference(rig, images, size)
    save_views(neutral, output / "neutral")
    if original:
        with Image.open(original) as src:
            src = src.convert("RGBA")
        save_views(src, output / "original")
        report["comparisons"]["original_to_neutral"] = difference(
            src, neutral, output / "original_to_neutral")
    else:
        report["stages"]["original"] = "missing"
    for value in (0., .25, .5, .75, 1.):
        settings = {p.id: value for p in rig.parameters if p.id.startswith("ParamEye")
                    and p.id.endswith("Open")}
        if settings:
            save_views(render_reference(rig, images, size, settings),
                       output / "blink" / str(value))
    faces = [p for p in rig.parts if p.semantic_role.value == "face_base"]
    features = [p for p in rig.parts if p.semantic_role.value.startswith(("eye", "mouth"))]
    for face in faces:
        for feature in features:
            result = duplicate_candidates(images[face.id], images[feature.id])
            if result is not None and result[0]:
                index = len(report["duplicate_candidates"])
                path = output / f"duplicate_{index:03d}.png"
                result[1].save(path)
                report["duplicate_candidates"].append({"face": face.id, "feature": feature.id,
                    "matching_dark_pixels": result[0], "overlay": path.name,
                    "status": "candidate_only"})
    if layerdiff:
        files = sorted(Path(layerdiff).glob("*.png"))
        report["stages"]["layerdiff"] = {"files": [p.name for p in files]}
        for i, file in enumerate(files):
            with Image.open(file) as im:
                save_views(im, output / "layerdiff" / f"{i:03d}")
    else:
        report["stages"]["layerdiff"] = "missing"
    if psd:
        raws, canvas_size = _raw_layers_from_psd(psd)
        composite = Image.new("RGBA", canvas_size)
        names = []
        ld_composite = Image.new("RGBA", canvas_size)
        matched = []
        for i, raw in enumerate(raws):
            im = Image.new("RGBA", canvas_size)
            im.paste(raw.image.convert("RGBA"), raw.offset)
            save_views(im, output / "psd" / f"{i:03d}")
            composite = Image.alpha_composite(composite, im)
            names.append(raw.name)
            # Match exact upstream tag names only; do not guess L/R or composite mappings.
            candidate = Path(layerdiff) / f"{raw.name}.png" if layerdiff else None
            if candidate and candidate.is_file() and candidate.parent == Path(layerdiff):
                with Image.open(candidate) as src:
                    ld = src.convert("RGBA")
                report["comparisons"][f"layerdiff_to_psd:{i}:{raw.name}"] = difference(
                    ld, im, output / "layerdiff_to_psd" / f"{i:03d}")
                if ld.size == canvas_size:
                    ld_composite = Image.alpha_composite(ld_composite, ld)
                    matched.append(raw.name)
        save_views(composite, output / "psd" / "composite")
        report["stages"]["psd"] = {"layers": names}
        if matched:
            save_views(ld_composite, output / "layerdiff" / "psd_order_composite")
            report["stages"]["layerdiff_composite"] = {
                "matched_names": matched, "complete": len(matched) == len(raws),
                "note": "PSD order reused; partial if tags/size differ, not a LayerDiff render"}
        if original:
            with Image.open(original) as src:
                report["comparisons"]["original_to_psd"] = difference(
                    src, composite, output / "original_to_psd")
        normalized = raws_to_stack(raws, canvas_size, output / "normalized_assets")
        norm_composite = Image.new("RGBA", canvas_size)
        for i, layer in enumerate(sorted(normalized.layers, key=lambda item: item.draw_order)):
            with Image.open(layer.texture_path) as image:
                save_views(image, output / "normalized" / f"{i:03d}")
                norm_composite = Image.alpha_composite(norm_composite, image.convert("RGBA"))
        save_views(norm_composite, output / "normalized" / "composite")
        report["stages"]["normalized"] = "reconstructed with current PSD importer (before synthesis)"
        report["comparisons"]["psd_to_normalized"] = difference(
            composite, norm_composite, output / "psd_to_normalized")
        report["comparisons"]["normalized_to_neutral"] = difference(
            norm_composite, neutral, output / "normalized_to_neutral")
        report["comparisons"]["psd_to_neutral"] = difference(
            composite, neutral, output / "psd_to_neutral")
    else:
        report["stages"]["psd"] = "missing"
    if atlas is None:
        atlas, uv_remap = build_atlas(rig, root)
        report["stages"]["atlas"] = "rebuilt using production packer"
    else:
        if uv_remap is None:
            raise ValueError("An existing atlas requires its matching UV remap")
        report["stages"]["atlas"] = "provided atlas with UV remap"
    save_views(atlas, output / "atlas")
    (output / "atlas" / "uv_remap.json").write_text(json.dumps(uv_remap, indent=2))
    restored = dict(images)
    for i, (pid, remap) in enumerate(uv_remap.items()):
        im = images[pid]
        w, h = im.size
        cx0, cy0, cx1, cy1, ax0, ay0, ax1, ay1 = remap
        src_box = tuple(round(v * n) for v, n in zip((cx0, cy0, cx1, cy1), (w, h, w, h)))
        dst_box = tuple(round(v * n) for v, n in zip(
            (ax0, ay0, ax1, ay1), (*atlas.size, *atlas.size)))
        crop = atlas.crop(dst_box)
        expected = im.crop(src_box)
        recovered = crop.resize(expected.size, Image.Resampling.NEAREST)
        info = difference(expected, recovered, output / "atlas_parts" / f"{i:03d}")
        info.update(source_size=expected.size, packed_size=crop.size)
        report["comparisons"][f"part_to_atlas:{pid}"] = info
        full = Image.new("RGBA", im.size)
        full.paste(recovered, src_box[:2])
        restored[pid] = full
        save_views(full, output / "atlas_parts" / f"{i:03d}")
    atlas_neutral = render_reference(rig, restored, size)
    save_views(atlas_neutral, output / "atlas_neutral")
    report["comparisons"]["neutral_to_atlas"] = difference(
        neutral, atlas_neutral, output / "neutral_to_atlas")
    report["confirmed_mesh_loss"] = [pid for pid, s in report["mesh"].items()
                                     if s["missing_pixels"]]
    ordered = [k for k in report["comparisons"] if k.startswith("layerdiff_to_psd:")]
    ordered += ["psd_to_normalized", "normalized_to_neutral", "neutral_to_atlas"]
    report["first_observed_change"] = next(
        (k for k in ordered if report["comparisons"].get(k, {}).get("changed_pixels", 0)), None)
    report["earliest_defect"] = (
        "Mesh UV coverage loss confirmed; earlier stages require aligned source review."
        if report["confirmed_mesh_loss"] else
        "Undetermined: no mesh coverage loss; inspect stage images and candidate overlays.")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "report.md").write_text(
        "# Visual quality diagnostics\n\n" + report["earliest_defect"] + "\n\n"
        + "\n".join(f"- {item}" for item in report["limits"])
        + "\n\nFirst observed stage change (not necessarily a defect): "
        + str(report["first_observed_change"])
        + "\n\n| Part | Alpha pixels | Missing pixels | Triangles |\n"
        + "|---|---:|---:|---:|\n"
        + "".join(f"| {pid} | {stats['alpha_pixels']} | {stats['missing_pixels']} | "
                  f"{stats['triangles']} |\n" for pid, stats in report["mesh"].items())
        + "\n\nSee report.json for counts, stage availability, names and comparisons. "
        "Numbered stage folders contain RGBA, RGB, alpha and white/black composites.\n",
        encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True, type=Path, help="IRR Rig JSON, not browser /rig JSON")
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New diagnostic directory")
    parser.add_argument("--original", type=Path, help="Image aligned to the model canvas")
    parser.add_argument("--layerdiff", type=Path, help="Directory of LayerDiff PNGs")
    parser.add_argument("--psd", type=Path)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--uv-remap", type=Path)
    args = parser.parse_args()
    if bool(args.atlas) != bool(args.uv_remap):
        parser.error("--atlas and --uv-remap must be supplied together")
    rig = Rig.model_validate_json(args.rig.read_text())
    atlas = None
    if args.atlas:
        with Image.open(args.atlas) as im:
            atlas = im.convert("RGBA")
    generate_report(rig, args.assets, args.output, original=args.original,
                    layerdiff=args.layerdiff, psd=args.psd, atlas=atlas,
                    uv_remap=json.loads(args.uv_remap.read_text()) if args.uv_remap else None)


if __name__ == "__main__":
    main()
