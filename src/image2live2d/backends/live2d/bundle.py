"""Validate referenced artifacts before offering a portable model ZIP.

Checks use our codec, not the proprietary Cubism Core. No SDK, weights, or
reference character assets are included in generated downloads.
"""
from __future__ import annotations

import io
import json
import re
import struct
import zipfile
from pathlib import Path, PurePosixPath


def safe_model_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:80] or "model"
    if name.upper() in {"CON", "PRN", "AUX", "NUL", *(
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    )}:
        name = "model_" + name
    return name


def validate_bundle(model_path: str | Path) -> list[str]:
    """Return the complete referenced file set, or raise ValueError.

    Relative paths must be portable and remain inside the bundle. Check actual
    MOC3 structure, texture indices, PNGs, and parameter/part references as well
    as existence. This does not establish runtime rendering compatibility.
    """
    from PIL import Image

    from .moc3_binary import read_moc3

    model_path = Path(model_path)
    root = model_path.parent.resolve()
    files = {model_path.name}

    def artifact(rel):
        if (not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel
                or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts):
            raise ValueError(f"Unsafe Live2D reference: {rel!r}")
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"Missing or unsafe Live2D artifact: {rel}")
        files.add(rel)
        return target

    def doc(path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"Invalid Live2D JSON object: {path.name}")
        return value

    try:
        manifest = doc(model_path)
        if manifest["Version"] != 3:
            raise ValueError("Expected model3 Version 3")
        refs = manifest["FileReferences"]
        moc = read_moc3(artifact(refs["Moc"]).read_bytes())
        if not moc.counts["artMeshes"] or not moc.counts["parameters"]:
            raise ValueError("Live2D model has no meshes or parameters")
        textures = refs["Textures"]
        if not isinstance(textures, list) or not textures:
            raise ValueError("Live2D model has no textures")
        for rel in textures:
            with Image.open(artifact(rel)) as image:
                if image.format != "PNG":
                    raise ValueError("Live2D textures must be PNG")
                image.verify()
        if any(i >= len(textures) for i in moc.get("artMeshes", "textureNos")):
            raise ValueError("MOC3 references an absent texture index")
        params = set(moc.get("parameters", "ids"))
        parts = set(moc.get("parts", "ids"))
        drawables = set(moc.get("artMeshes", "ids"))

        def require(ids, known):
            missing = set(ids) - known
            if missing:
                raise ValueError(f"Unknown Live2D IDs: {sorted(missing)}")

        for group in manifest.get("Groups", []):
            require(group["Ids"], params if group["Target"] == "Parameter" else parts)
        require([area["Id"] for area in manifest.get("HitAreas", [])], drawables)
        if "Physics" in refs:
            physics = doc(artifact(refs["Physics"]))
            for setting in physics["PhysicsSettings"]:
                require([x["Source"]["Id"] for x in setting["Input"]], params)
                require([x["Destination"]["Id"] for x in setting["Output"]], params)
                for output in setting["Output"]:
                    if not 1 <= output["VertexIndex"] < len(setting["Vertices"]):
                        raise ValueError("Physics output references an absent vertex")
        display = doc(artifact(refs["DisplayInfo"]))
        require([x["Id"] for x in display["Parameters"]], params)
        require([x["Id"] for x in display["Parts"]], parts)
        for group in refs.get("Motions", {}).values():
            for entry in group:
                motion = doc(artifact(entry["File"]))
                for curve in motion["Curves"]:
                    if curve["Target"] == "Parameter":
                        require([curve["Id"]], params)
                    elif curve["Target"] == "PartOpacity":
                        require([curve["Id"]], parts)
        for entry in refs.get("Expressions", []):
            expression = doc(artifact(entry["File"]))
            require([x["Id"] for x in expression["Parameters"]], params)
        if "Pose" in refs:
            pose = doc(artifact(refs["Pose"]))
            for group in pose["Groups"]:
                for part in group:
                    require([part["Id"], *part.get("Link", [])], parts)
        if "UserData" in refs:
            doc(artifact(refs["UserData"]))
    except (OSError, KeyError, TypeError, IndexError, AssertionError, struct.error,
            NotImplementedError) as exc:
        raise ValueError(f"Invalid Live2D bundle: {exc}") from exc
    return sorted(files)


def bundle_zip(model_path: str | Path) -> bytes:
    """Revalidate on every download; include only the referenced artifacts."""
    model_path = Path(model_path)
    files = validate_bundle(model_path)
    folder = safe_model_name(model_path.name.removesuffix(".model3.json"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(model_path.parent / rel, f"{folder}/{rel}")
        archive.writestr(f"{folder}/IMPORT_README.txt", (
            "Extract this entire folder before importing into VTube Studio.\n"
            "Select the .model3.json and keep its companion files together.\n"
            "Structural checks passed using image2live2d's own codec.\n"
            "Official Cubism Core and VTube Studio compatibility: UNVERIFIED.\n"
            "Tracking mappings and physics may require manual tuning.\n"
            "No Cubism SDK/Core or third-party model weights are included.\n"
            "Use only character artwork you have permission to use.\n"
        ))
    return buf.getvalue()


def write_bundle_zip(model_path: str | Path) -> Path:
    """Write a validated ZIP next to its bundle directory for CLI/API callers."""
    model_path = Path(model_path)
    directory = model_path.parent.resolve()
    target = directory.with_name(directory.name + ".zip")
    data = bundle_zip(model_path)
    target.write_bytes(data)
    return target
