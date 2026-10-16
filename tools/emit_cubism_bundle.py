"""Emit a complete, Cubism-Viewer-loadable Live2D bundle from a layer directory.

Uses the same complete exporter as the CLI, API and web app. Official Cubism Core
and VTube Studio behavior remains unverified until tested in those applications.

Usage::

    python tools/emit_cubism_bundle.py out/char_layers_fixed out/eyeball/char_fixed_v10_cubism

Produces ``model.moc3``, ``model.model3.json``, ``model.physics3.json``, ``model.cdi3.json``,
``model.<anim>.motion3.json`` (idle + expressions) and ``textures/atlas.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image2live2d.backends.live2d import Live2DEmitter
from image2live2d.core import decompose
from image2live2d.pipeline import rig_from_stack


def emit_bundle(layer_dir: Path, out: Path) -> Path:
    stack = decompose.from_layer_dir(layer_dir)
    rig = rig_from_stack(stack, name="model", source=str(layer_dir))
    bundle = Live2DEmitter(asset_root=layer_dir).build(rig, out)
    from image2live2d.backends.live2d.bundle import write_bundle_zip
    write_bundle_zip(bundle.model3_path)
    print(f"wrote {bundle.model3_path} (structurally validated; runtime unverified)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer_dir", type=Path, help="directory of {order}_{role}.png layers")
    ap.add_argument("out", type=Path, help="output bundle directory")
    args = ap.parse_args(argv)
    if not args.layer_dir.is_dir():
        print(f"error: {args.layer_dir} is not a directory", file=sys.stderr)
        return 2
    emit_bundle(args.layer_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
