#!/usr/bin/env python3
"""Preserve LayerDiff mouth RGBA in pinned See-through PSD postprocessing.

The original fullpage RGB replacement imports skin/background into every mouth texel,
including wide soft-alpha margins. Do not classify/remove skin colors: lips, teeth and
mouth interiors can legitimately share those colors. Nose handling is unchanged.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "image2live2d: preserve generated mouth RGB"
ANCHOR = """    if 'mouth' in tag2pinfo:
        xyxy = tag2pinfo['mouth']['xyxy']
        tag2pinfo['mouth']['img'][..., :3] = fullpage[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2], :3]
"""


def patch_text(text: str) -> tuple[str, str]:
    if MARKER in text and ANCHOR not in text:
        return text, "already-patched"
    if text.count(ANCHOR) != 1:
        return text, "anchor-not-found"
    result = text.replace(ANCHOR, f"    # {MARKER}; retain alpha and all generated colors.\n")
    ast.parse(result)
    return result, "patched"


def main(path: str) -> int:
    target = Path(path)
    result, status = patch_text(target.read_text(encoding="utf-8"))
    if status == "anchor-not-found":
        print(f"preserve_mouth_rgb: {status}: {path}", file=sys.stderr)
        return 3
    if status == "patched":
        target.write_text(result, encoding="utf-8")
    print(f"preserve_mouth_rgb: {status}: {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: preserve_mouth_rgb.py <inference_utils.py>")
    raise SystemExit(main(sys.argv[1]))
