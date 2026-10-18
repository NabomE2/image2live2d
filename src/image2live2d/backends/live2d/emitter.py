"""Shared complete Live2D export, used by the API, CLI, web app and tools.

Default generation and atlas packing reuse tools/emit_cubism_bundle.py's pipeline.
Structural validation is not official Cubism Core or VTube Studio certification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...irr.schema import Rig
from ..base import Emitter
from . import cdi3 as _cdi3
from . import model3 as _model3
from . import motion3 as _motion3
from . import physics3 as _physics3
from .bundle import safe_model_name, validate_bundle
from .moc3 import MocWriter, write_moc3_from_template


@dataclass
class Live2DBundle:
    model3_path: Path
    files: list[str]
    moc_written: bool


class Live2DEmitter(Emitter):
    name = "live2d"
    extension = ".model3.json"

    def __init__(self, *, asset_root: str | Path | None = None,
                 moc_writer: MocWriter | None = None, moc_template: str | Path | None = None):
        self.asset_root = Path(asset_root) if asset_root is not None else None
        self.moc_writer = moc_writer
        self.moc_template = moc_template

    def emit(self, rig: Rig, out_dir: Path) -> Path:
        return self.build(rig, out_dir).model3_path

    def build(self, rig: Rig, out_dir: str | Path, *,
              diagnostics_dir: str | Path | None = None) -> Live2DBundle:
        from .moc3_binary import write_moc3
        from .moc3_emit import build_atlas, rig_to_moc3

        if self.asset_root is None:
            raise ValueError("Live2D export requires asset_root with real PNG textures")
        # Check before creating a manifest: never substitute placeholder art.
        for tex in rig.textures:
            if not (self.asset_root / tex.path).is_file():
                raise ValueError(f"Missing Live2D texture: {tex.path}")
        out = Path(out_dir)
        (out / "textures").mkdir(parents=True, exist_ok=True)
        name = safe_model_name(rig.meta.name)

        def write_json(rel: str, doc: dict) -> None:
            (out / rel).write_text(json.dumps(doc, indent=2), encoding="utf-8")

        if self.moc_writer is None:
            atlas, uv_remap = build_atlas(rig, self.asset_root)
            atlas.save(out / "textures/atlas.png")
            textures = ["textures/atlas.png"]
            data = write_moc3(rig_to_moc3(rig, atlas_uv=uv_remap))
            if diagnostics_dir is not None:
                from ...core.qa.visual import generate_report

                generate_report(rig, self.asset_root, diagnostics_dir,
                                atlas=atlas, uv_remap=uv_remap)
        else:
            if diagnostics_dir is not None:
                raise ValueError("Atlas diagnostics require the default Live2D writer")
            # Preserve the explicit writer extension point and its per-texture numbering.
            textures = []
            for i, tex in enumerate(rig.textures):
                rel = f"textures/{i:03d}.png"
                (out / rel).write_bytes((self.asset_root / tex.path).read_bytes())
                textures.append(rel)
            data = write_moc3_from_template(rig, self.moc_template, writer=self.moc_writer)
        moc_file = f"{name}.moc3"
        (out / moc_file).write_bytes(data)
        physics_file = None
        if rig.physics:
            physics_file = f"{name}.physics3.json"
            write_json(physics_file, _physics3.physics3(rig))
        cdi_file = f"{name}.cdi3.json"
        write_json(cdi_file, _cdi3.cdi3(rig))
        motions: dict[str, list[str]] = {}
        for anim in rig.animations:
            rel = f"{name}.{safe_model_name(anim.name)}.motion3.json"
            write_json(rel, _motion3.motion3(anim))
            group = motions.setdefault(anim.name.capitalize(), [])
            if rel not in group:
                group.append(rel)
        model_path = out / f"{name}.model3.json"
        write_json(model_path.name, _model3.model3(
            rig, moc=moc_file, textures=textures, physics=physics_file,
            display_info=cdi_file, motions=motions or None))
        files = validate_bundle(model_path)
        return Live2DBundle(model_path, files, True)
