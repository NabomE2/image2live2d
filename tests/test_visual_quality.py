"""Pixel-level regressions: no GPU, generated art or proprietary runtime required."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from image2live2d.core.mesh.build import alpha_bbox, build_mesh, grid_mesh
from image2live2d.core.qa.visual import (
    difference,
    duplicate_candidates,
    generate_report,
    mesh_coverage,
    render_reference,
)
from image2live2d.core.types import Layer
from image2live2d.irr.schema import (
    Keyform,
    Meta,
    Parameter,
    Part,
    Rig,
    SemanticRole,
    Texture,
)


def fixture_art(kind):
    im = Image.new("RGBA", (128, 128))
    draw = ImageDraw.Draw(im)
    draw.rectangle((14, 50, 113, 99), fill=(120, 60, 80, 255))
    if kind == "hair_tip":
        draw.line((64, 10, 64, 50), fill=(20, 10, 30, 255))
    elif kind == "eyelash":
        draw.line((30, 50, 4, 20), fill=(20, 10, 30, 255))
    elif kind == "ornament":
        draw.rectangle((117, 8, 118, 9), fill=(255, 230, 10, 255))
    elif kind == "thin_line":
        draw.line((120, 10, 120, 110), fill=(20, 10, 30, 255))
    elif kind == "soft_edge":
        draw.line((64, 10, 64, 50), fill=(20, 10, 30, 8))
    return im


def make_layer(tmp_path, im):
    path = tmp_path / "part.png"
    im.save(path)
    return Layer("part", SemanticRole.face_base, path, 0, *im.size)


@pytest.mark.parametrize("kind", ["hair_tip", "eyelash", "ornament", "thin_line", "soft_edge"])
@pytest.mark.parametrize("tighten", [True, False])
def test_every_retained_pixel_covered_without_more_grid_resolution(tmp_path, kind, tighten):
    im = fixture_art(kind)
    layer = make_layer(tmp_path, im)
    mesh = build_mesh(layer, tighten=tighten)
    stats, _ = mesh_coverage(im, mesh)
    assert stats["missing_pixels"] == 0
    assert stats["coverage"] == 1
    assert len(mesh.triangles) <= 2 * 14 * 14
    assert len(mesh.triangles) < 2 * 14 * 14  # still prunes empty cells


def legacy_mesh(im):
    """Frozen pre-fix bbox + 3x3 probes, also used for before/after artifact comparisons."""
    w, h = im.size
    px = im.getchannel("A").load()
    box = alpha_bbox(lambda x, y: px[x, y], w, h, 8)
    x0, y0, x1, y1 = box
    uv = (x0/w, y0/h, (x1+1)/w, (y1+1)/h)
    return grid_mesh("part", (uv[0], 1-uv[3], uv[2], 1-uv[1]),
                     lambda u, v: px[min(w-1, int(u*w)), min(h-1, int(v*h))], uv_rect=uv)


def test_reproduce_old_tip_loss(tmp_path):
    im = fixture_art("hair_tip")
    old, _ = mesh_coverage(im, legacy_mesh(im))
    new, _ = mesh_coverage(im, build_mesh(make_layer(tmp_path, im)))
    assert old["missing_pixels"] == 40
    assert new["missing_pixels"] == 0


def tiny_rig(tmp_path):
    im = Image.new("RGBA", (16, 16), (10, 30, 40, 128))
    layer = make_layer(tmp_path, im)
    rig = Rig(meta=Meta(name="diagnostic"),
              textures=[Texture(id="t", path="part.png", width=16, height=16)],
              parts=[Part(id="part", texture_id="t", semantic_role=SemanticRole.face_base,
                          draw_order=0)], meshes=[build_mesh(layer, grid=2)])
    return rig, im


def test_renderer_shared_edges_do_not_double_alpha_and_respects_opacity(tmp_path):
    rig, im = tiny_rig(tmp_path)
    rig.parameters = [Parameter(id="ParamEyeLOpen", min=0, max=1, default=1,
                                keyforms=[Keyform(value=0, opacity_overrides={"part": 0}),
                                          Keyform(value=1, opacity_overrides={"part": 1})])]
    assert np.array_equal(render_reference(rig, {"part": im}, im.size), im)
    half = render_reference(rig, {"part": im}, im.size, {"ParamEyeLOpen": .5})
    assert np.all(np.asarray(half)[..., 3] == 64)
    shut = render_reference(rig, {"part": im}, im.size, {"ParamEyeLOpen": 0})
    assert shut.getbbox() is None


def test_difference_ignores_invisible_rgb_and_refuses_stretching(tmp_path):
    a = Image.new("RGBA", (3, 3), (255, 0, 0, 0))
    b = Image.new("RGBA", (3, 3), (0, 255, 0, 0))
    assert difference(a, b, tmp_path)["changed_pixels"] == 0
    assert difference(a, b.resize((2, 2)), tmp_path)["status"] == "unaligned"


def test_duplicate_detector_is_report_only():
    face = Image.new("RGBA", (4, 4), (30, 30, 30, 255))
    feature = face.copy()
    result = duplicate_candidates(face, feature)
    assert result[0] == 16
    assert np.array_equal(face, feature)  # no automatic deletion
    assert duplicate_candidates(face, Image.new("RGBA", (4, 4)))[0] == 0


def test_report_real_atlas_and_missing_stage_status(tmp_path):
    from image2live2d.backends.live2d.moc3_emit import build_atlas

    rig, _im = tiny_rig(tmp_path)
    atlas, remap = build_atlas(rig, tmp_path, atlas_size=64)
    out = tmp_path / "diagnostic"
    report = generate_report(rig, tmp_path, out, original=tmp_path / "part.png",
                             atlas=atlas, uv_remap=remap)
    assert report["stages"]["layerdiff"] == "missing"
    assert report["stages"]["psd"] == "missing"
    assert report["mesh"]["part"]["coverage"] == 1
    assert report["comparisons"]["neutral_to_atlas"]["changed_pixels"] == 0
    assert (out / "atlas" / "alpha.png").exists()
    assert (out / "parts" / "000" / "missing_alpha.png").exists()
    assert json.loads((out / "report.json").read_text())["schema"] == 1
    with pytest.raises(FileExistsError):
        generate_report(rig, tmp_path, out)


def load_patch():
    path = Path(__file__).resolve().parents[1] / "service/seethrough/patches/preserve_mouth_rgb.py"
    spec = importlib.util.spec_from_file_location("mouth_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mouth_patch_preserves_lips_teeth_cavity_and_alpha():
    patch = load_patch()
    source = "def post(tag2pinfo, fullpage):\n" + patch.ANCHOR + "    return tag2pinfo\n"
    revised, status = patch.patch_text(source)
    assert status == "patched"
    assert patch.patch_text(revised) == (revised, "already-patched")
    colors = np.array([[[180, 60, 100, 180], [250, 245, 225, 255], [30, 10, 20, 255],
                        [80, 20, 40, 32]]], dtype=np.uint8)
    skin = np.full((1, 4, 4), [240, 200, 170, 255], dtype=np.uint8)
    old_ns, new_ns = {}, {}
    exec(source, old_ns)  # noqa: S102 -- fixed in-test source, never user input
    exec(revised, new_ns)  # noqa: S102 -- fixed in-test patched source
    def run(ns):
        info = {"mouth": {"xyxy": [0, 0, 4, 1], "img": colors.copy()}}
        return ns["post"](info, skin)["mouth"]["img"]
    assert np.array_equal(run(old_ns)[..., :3], skin[..., :3])
    assert np.array_equal(run(new_ns), colors)
    assert patch.patch_text("changed upstream")[1] == "anchor-not-found"


def test_mouth_patch_applies_only_to_pinned_postprocessing():
    upstream = os.environ.get("SEE_THROUGH_TEST_SOURCE")
    if not upstream:
        pytest.skip("pinned See-through checkout not configured")
    patch = load_patch()
    source = (Path(upstream) / "common/utils/inference_utils.py").read_text()
    revised, status = patch.patch_text(source)
    assert status == "patched"
    assert revised == source.replace(patch.ANCHOR,
        f"    # {patch.MARKER}; retain alpha and all generated colors.\n")
    assert "tag2pinfo['nose']['img'][..., :3] = fullpage" in revised


def test_psd_stage_comparisons_and_cli(tmp_path, monkeypatch):
    import sys

    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    from image2live2d.core.qa.visual import main

    rig, im = tiny_rig(tmp_path)
    psd = PSDImage.new("RGBA", im.size)
    PixelLayer.frompil(im, psd, name="face")
    path = tmp_path / "sample.psd"
    psd.save(path)
    ld = tmp_path / "layerdiff"
    ld.mkdir()
    im.save(ld / "face.png")
    rig_path = tmp_path / "rig.json"
    rig_path.write_text(rig.model_dump_json())
    out = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["visual", "--rig", str(rig_path), "--assets", str(tmp_path),
                                    "--output", str(out), "--psd", str(path), "--layerdiff", str(ld)])
    main()
    report = json.loads((out / "report.json").read_text())
    assert report["comparisons"]["layerdiff_to_psd:0:face"]["changed_pixels"] == 0
    assert report["comparisons"]["psd_to_normalized"]["changed_pixels"] == 0
    assert report["stages"]["layerdiff_composite"]["complete"]
    assert (out / "normalized" / "composite" / "alpha.png").exists()


def test_emitter_diagnostics_use_actual_atlas(tmp_path, monkeypatch):
    from image2live2d.backends.live2d import moc3_emit
    from image2live2d.backends.live2d.emitter import Live2DEmitter

    rig, _im = tiny_rig(tmp_path)
    rig.parameters = [Parameter(id="ParamTest", min=0, max=1, default=0,
                                keyforms=[Keyform(value=0), Keyform(value=1)])]
    packer = moc3_emit.build_atlas
    monkeypatch.setattr(moc3_emit, "build_atlas",
                        lambda rig, root: packer(rig, root, atlas_size=64))
    out = tmp_path / "bundle"
    debug = tmp_path / "debug"
    Live2DEmitter(asset_root=tmp_path).build(rig, out, diagnostics_dir=debug)
    assert (out / "textures/atlas.png").read_bytes() == (debug / "atlas/rgba.png").read_bytes()
    assert (out / "diagnostic.model3.json").exists()
    assert json.loads((debug / "report.json").read_text())["stages"]["atlas"].startswith("provided")
