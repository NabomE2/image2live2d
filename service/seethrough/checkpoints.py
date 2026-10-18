"""Durable stage snapshots. A manifest is published only after successful validation.

Every stage writes into its own copy, never into an earlier committed snapshot.
One service process owns a job; the application's GPU slot serializes writers.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import uuid
from pathlib import Path

STAGES = ("layerdiff", "depth", "psd")
PARTS = ("hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth",
         "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear",
         "tail", "wings", "objects")
BODY = ("front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
        "bottomwear", "legwear", "footwear", "tail", "wings", "objects")
HEAD = ("headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear", "ears",
        "earwear", "nose", "mouth")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def files(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint output must not contain symlinks")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def validate(root, stage):
    """Validate the pinned CLI's stage contract, including every serialized PNG."""
    from PIL import Image

    root = Path(root)
    pages = list(root.glob("*/src_img.png"))
    if len(pages) != 1:
        raise ValueError("Expected one LayerDiff src_img.png")
    page = pages[0].parent
    pngs = list(root.rglob("*.png"))
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
    expected = BODY if (page / "front hair.png").exists() else PARTS
    if any(not (page / f"{tag}.png").is_file() for tag in expected):
        raise ValueError("LayerDiff part PNGs are incomplete")
    if (page / "src_head.png").exists() and any(not (page / f"{tag}.png").is_file() for tag in HEAD):
        raise ValueError("LayerDiff head PNGs are incomplete")
    if stage in {"depth", "psd"}:
        info = json.loads((page / "info.json").read_text())
        if not isinstance(info.get("parts"), dict) or not info["parts"]:
            raise ValueError("Depth parts metadata is missing")
        if not (page / "reconstruction.png").is_file():
            raise ValueError("Depth reconstruction is missing")
        expected_depth = set(PARTS)
        for composite, tags in {"hair": ("back hair", "front hair"),
                                "eyes": ("eyewhite", "irides", "eyelash", "eyebrow")}.items():
            present = {tag for tag in tags if (page / f"{tag}.png").exists()}
            if present:
                expected_depth.discard(composite)
                expected_depth.update(present)
        if not expected_depth.issubset(info["parts"]):
            raise ValueError("Depth parts metadata is incomplete")
        for tag in info["parts"]:
            if Path(tag).name != tag or not (page / f"{tag}_depth.png").is_file():
                raise ValueError("Depth output is incomplete")
    if stage == "psd":
        from psd_tools import PSDImage
        outputs = list(root.rglob("*.psd"))
        layered = [p for p in outputs if not p.stem.endswith("_depth")]
        if len(layered) != 1 or not layered[0].with_name(layered[0].stem + "_depth.psd").is_file():
            raise ValueError("Expected layered and depth PSDs")
        for path in outputs:
            PSDImage.open(path)
        json.loads(Path(str(layered[0]) + ".json").read_text())
        return layered[0]
    return None


def fingerprint(source, command, upstream, model_paths):
    """Content hashes, not repository names/mtimes. Unknown remote models fail closed.

    Model hashing is intentionally streaming; no weights are loaded into CPU/GPU RAM.
    Timeout/retention settings are excluded because they do not change output.
    """
    code = {}
    for root in (Path(__file__).parent, Path(upstream) / "common", Path(upstream) / "inference/scripts"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".txt"}:
                code[str(path)] = digest(path)
    model_keys = {}
    for name in ("layerdiff", "depth"):
        root = Path(model_paths.get(name, ""))
        if name not in model_paths or not root.is_dir():
            # A mutable remote repo name is not evidence of model identity.
            model_keys[name] = "unresolved:" + uuid.uuid4().hex
            continue
        model_keys[name] = key({str(p.relative_to(root)): digest(p)
                               for p in sorted(root.rglob("*"))
                               if p.is_file() and ".cache" not in p.relative_to(root).parts})
    versions = {}
    for package in ("torch", "diffusers", "transformers", "numpy", "Pillow", "psd-tools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    script = Path(command[0])
    if not script.is_absolute():
        script = Path(upstream) / script
    if script.is_file():
        code["entry_script"] = digest(script)
    common = {"input": digest(source), "command": command, "code": code,
              "versions": versions, "schema": 1}
    layer = key([common, model_keys["layerdiff"]])
    depth = key([layer, model_keys["depth"]])
    return {"layerdiff": layer, "depth": depth, "psd": key([depth, "psd"]) }


class Checkpoints:
    def __init__(self, work):
        self.root = Path(work) / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.json"
        try:
            self.state = json.loads(self.path.read_text())
            if not isinstance(self.state, dict) or any(not isinstance(v, dict) for v in self.state.values()):
                raise ValueError("Invalid checkpoint manifest")
        except (OSError, ValueError):
            self.state = {}

    def save(self):
        atomic_json(self.path, self.state)

    def valid(self, stage, identity):
        entry = self.state.get(stage, {})
        if entry.get("status") != "done" or entry.get("identity") != identity:
            return False
        try:
            directory = entry["directory"]
            if Path(directory).name != directory:
                return False
            root = self.root / directory
            return root.is_dir() and bool(entry["files"]) and files(root) == entry["files"]
        except (OSError, ValueError, KeyError):
            return False

    def invalidate(self, stage):
        for name in STAGES[STAGES.index(stage):]:
            self.state[name] = {"status": "pending"}
        self.save()

    def execute(self, identities, timeouts, run, on_state, force=None, stop_after=None):
        if force:
            # A requested Depth-only retry must never silently run LayerDiff.
            index = STAGES.index(force)
            for prior in STAGES[:index]:
                if not self.valid(prior, identities[prior]):
                    raise ValueError(f"Cannot restart {force}: {prior} checkpoint is invalid; resume normally")
            self.invalidate(force)
        for stage in STAGES:
            if self.valid(stage, identities[stage]):
                self.state[stage]["reused"] = True
                on_state(stage, self.state[stage])
            else:
                self.invalidate(stage)
                attempt = self.root / ("attempt-" + uuid.uuid4().hex)
                previous = STAGES.index(stage) - 1
                if previous >= 0:
                    shutil.copytree(self.root / self.state[STAGES[previous]]["directory"], attempt)
                else:
                    attempt.mkdir()
                started = time.time()
                self.state[stage] = {"status": "running", "started": started,
                                     "timeout_seconds": timeouts[stage]}
                self.save()
                on_state(stage, self.state[stage])
                try:
                    run(stage, attempt, time.monotonic() + timeouts[stage])
                    validate(attempt, stage)
                    inventory = files(attempt)
                    # Sync payloads before committing the manifest that references them.
                    for path in attempt.rglob("*"):
                        if path.is_file():
                            with path.open("rb") as stream:
                                os.fsync(stream.fileno())
                    for directory in [*sorted((p for p in attempt.rglob("*") if p.is_dir()), reverse=True), attempt]:
                        fd = os.open(directory, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    committed = self.root / (stage + "-" + uuid.uuid4().hex)
                    os.replace(attempt, committed)
                    self.state[stage].update(status="done", identity=identities[stage],
                                             directory=committed.name, files=inventory,
                                             finished=time.time(), reused=False)
                    self.save()
                    on_state(stage, self.state[stage])
                except BaseException as exc:
                    self.state[stage].update(status="error", error=str(exc),
                                             error_code=getattr(exc, "code", "inference_failed"),
                                             finished=time.time())
                    self.save()
                    on_state(stage, self.state[stage])
                    raise
                finally:
                    shutil.rmtree(attempt, ignore_errors=True)
                    self.prune()
            if stage == stop_after:
                return None
        return validate(self.root / self.state["psd"]["directory"], "psd")

    def prune(self):
        referenced = {e.get("directory") for e in self.state.values()}
        for path in self.root.iterdir():
            if path.is_dir() and path.name not in referenced:
                shutil.rmtree(path)
