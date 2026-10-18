"""Small real serialized outputs for staged checkpoint contract tests (no ML)."""
import json

import pytest


@pytest.fixture
def stage_output():
    parts = ("hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear", "nose", "mouth",
             "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear",
             "tail", "wings", "objects")
    def write(root, stage):
        from PIL import Image
        from psd_tools import PSDImage

        page = root / "input"
        page.mkdir(parents=True, exist_ok=True)
        if stage == "layerdiff":
            for name in ("src_img", *parts):
                Image.new("RGBA", (8, 8), (100, 50, 30, 255)).save(page / f"{name}.png")
        elif stage == "depth":
            for name in parts:
                Image.new("L", (8, 8), 100).save(page / f"{name}_depth.png")
            Image.new("RGBA", (8, 8)).save(page / "reconstruction.png")
            (page / "info.json").write_text(json.dumps({"parts": {name: {} for name in parts}}))
        else:
            PSDImage.new("RGBA", (8, 8)).save(root / "input.psd")
            PSDImage.new("L", (8, 8)).save(root / "input_depth.psd")
            (root / "input.psd.json").write_text('{}')
    return write
