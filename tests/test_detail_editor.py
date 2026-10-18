"""Editor invariants, transactions and HTTP workflows using real CPU-produced bundles."""
from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from image2live2d.app.editor import Conflict, EditorStore, brush, digest, read
from image2live2d.backends.live2d.emitter import Live2DEmitter
from image2live2d.core.mesh.build import build_mesh
from image2live2d.core.types import Layer
from image2live2d.irr.schema import Keyform, Meta, Parameter, Part, Rig, SemanticRole, Texture


@pytest.fixture
def generated(tmp_path, monkeypatch):
    from image2live2d.app import editor
    from image2live2d.backends.live2d import moc3_emit

    pack = moc3_emit.build_atlas
    # Small real atlas avoids 4096px PNG overhead in transaction fault-injection tests.
    monkeypatch.setattr(moc3_emit, "build_atlas", lambda rig, root: pack(rig, root, atlas_size=128))
    monkeypatch.setattr(editor, "build_atlas", lambda rig, root: pack(rig, root, atlas_size=128))
    assets = tmp_path / "layers"
    assets.mkdir()
    parts, meshes, textures = [], [], []
    for i, role in enumerate((SemanticRole.face_base, SemanticRole.mouth)):
        pid = role.value
        im = Image.new("RGBA", (32, 32), (130+i*40, 90, 80, 180))
        im.save(assets / f"{pid}.png")
        textures.append(Texture(id=pid, path=f"{pid}.png", width=32, height=32))
        parts.append(Part(id=pid, texture_id=pid, semantic_role=role, draw_order=i))
        meshes.append(build_mesh(Layer(pid, role, assets / f"{pid}.png", i, 32, 32), grid=2))
    rig = Rig(meta=Meta(name="model"), parts=parts, meshes=meshes, textures=textures,
              parameters=[Parameter(id="ParamMouthOpenY", min=0, max=1, default=0,
                                    keyforms=[Keyform(value=0), Keyform(value=1)])])
    bundle = tmp_path / "live2d"
    Live2DEmitter(asset_root=assets).build(rig, bundle)
    store = EditorStore(tmp_path / "projects")
    return SimpleNamespace(store=store, rig=rig, assets=assets, bundle=bundle, root=tmp_path, pack=pack)


def create(g):
    return g.store.create(g.rig, g.assets, g.bundle, b"original source bytes", "hero.png", g.assets)


def stroke(mode="erase"):
    return {"x": 16, "y": 16, "radius": 4, "hardness": .6, "mode": mode}


def changed(g, project):
    draft = g.store.draft(project["id"], project["current"], "mouth")
    return g.store.strokes(project["id"], draft["id"], 0, [stroke()])


def test_brush_preserves_rgb_outside_pixels_and_original_ceiling():
    im = Image.new("RGBA", (32, 32), (80, 90, 100, 180))
    erased = brush(im, im, stroke())
    a, b = np.asarray(im), np.asarray(erased)
    assert np.array_equal(a[..., :3], b[..., :3])
    yy, xx = np.mgrid[:32, :32]
    outside = (xx+.5-16)**2+(yy+.5-16)**2 > 16
    assert np.array_equal(a[outside], b[outside])
    assert b[..., 3].min() == 0
    restored = erased
    for _ in range(8):
        restored = brush(restored, im, {**stroke("restore"), "hardness": 1})
    assert np.array_equal(restored, im)


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"radius": 0}, {"hardness": 2},
                                 {"mode": "execute"}, {"x": -1}])
def test_invalid_strokes(bad):
    im = Image.new("RGBA", (32, 32))
    with pytest.raises(ValueError):
        brush(im, im, {**stroke(), **bad})


def test_snapshot_restart_apply_restore_and_zip(generated):
    g = generated
    initial = create(g)
    pid, original_rid = initial["id"], initial["current"]
    p = g.store.project(pid)
    hashes = read(p / "original_hashes.json")
    draft = changed(g, initial)
    # A new store instance has no in-memory job/rig state.
    g.store = EditorStore(g.store.root)
    assert g.store.info(pid)["drafts"][0]["sequence"] == 1
    result = g.store.apply(pid, draft["id"], 1, True)
    rid = result["current"]
    before = p / "revisions" / original_rid
    after = p / "revisions" / rid
    assert (before / "bundle/model.moc3").read_bytes() == (after / "bundle/model.moc3").read_bytes()
    assert (before / "parts/000.png").read_bytes() == (after / "parts/000.png").read_bytes()
    a = np.array(Image.open(before / "bundle/textures/atlas.png"))
    b = np.array(Image.open(after / "bundle/textures/atlas.png"))
    assert np.array_equal(a[..., :3], b[..., :3])
    box = read(p / "original/layout.json")["mouth"]["tile"]
    x0, y0, x1, y1 = box
    outside = np.ones(a.shape[:2], dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(a[outside], b[outside])
    assert np.any(a != b)
    assert g.store.apply(pid, draft["id"], 1, True)["current"] == rid  # idempotent retry
    archive = zipfile.ZipFile(io.BytesIO(g.store.download(pid, rid)))
    assert archive.read("model/model.moc3") == (before / "bundle/model.moc3").read_bytes()
    restored = g.store.restore(pid, original_rid, rid, True)
    assert restored["current"] not in (rid, original_rid)
    assert (p / "revisions" / restored["current"] / "bundle/textures/atlas.png").read_bytes() == (
        before / "bundle/textures/atlas.png").read_bytes()
    assert all(digest(p / "original" / name) == value for name, value in hashes.items())


def test_conflict_and_explicit_approval(generated):
    g = generated
    project = create(g)
    a, b = changed(g, project), changed(g, project)
    with pytest.raises(ValueError, match="approval"):
        g.store.apply(project["id"], a["id"], 1, False)
    g.store.apply(project["id"], a["id"], 1, True)
    with pytest.raises(Conflict):
        g.store.apply(project["id"], b["id"], 1, True)
    with pytest.raises(Conflict):
        g.store.strokes(project["id"], a["id"], 0, [stroke()])


def test_concurrent_writers_only_publish_one_revision(generated):
    g = generated
    project = create(g)
    drafts = [changed(g, project), changed(g, project)]
    def apply(d):
        try:
            return EditorStore(g.store.root).apply(project["id"], d["id"], 1, True)
        except Conflict:
            return None
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(apply, drafts))
    assert sum(x is not None for x in results) == 1
    assert len(g.store.info(project["id"])["revisions"]) == 2


def test_failed_export_never_publishes(generated, monkeypatch):
    from image2live2d.app import editor

    g = generated
    project = create(g)
    draft = changed(g, project)
    def fail(_):
        raise ValueError("injected validation failure")
    monkeypatch.setattr(editor, "validate_bundle", fail)
    with pytest.raises(ValueError, match="injected"):
        g.store.apply(project["id"], draft["id"], 1, True)
    assert g.store.info(project["id"])["current"] == project["current"]
    assert len(g.store.info(project["id"])["revisions"]) == 1


def test_commit_point_failure_leaves_old_version_visible(generated, monkeypatch):
    from image2live2d.app import editor

    g = generated
    project = create(g)
    draft = changed(g, project)
    write = editor.atomic_json
    def fail(path, value):
        if Path(path).name == "project.json":
            raise OSError("simulated interrupted commit")
        return write(path, value)
    monkeypatch.setattr(editor, "atomic_json", fail)
    with pytest.raises(OSError):
        g.store.apply(project["id"], draft["id"], 1, True)
    reopened = EditorStore(g.store.root).info(project["id"])
    assert reopened["current"] == project["current"]
    assert len(reopened["revisions"]) == 1


def test_refuse_mismatched_moc_and_non_mouth(generated):
    g = generated
    project = create(g)
    with pytest.raises(ValueError, match="Only mouth"):
        g.store.draft(project["id"], project["current"], "face_base")
    # Valid geometry, but different from the actual shipped binary.
    rig = g.rig.model_copy(deep=True)
    rig.meshes[0].vertices[0] = (.1, .1)
    with pytest.raises(ValueError, match="moc3"):
        g.store.create(rig, g.assets, g.bundle, b"source", "hero.png", g.assets)
    with pytest.raises(ValueError):
        g.store.info("../../etc")


def test_integrity_rejects_original_tampering(generated):
    g = generated
    project = create(g)
    draft = changed(g, project)
    (g.store.project(project["id"]) / "original/rig.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        g.store.apply(project["id"], draft["id"], 1, True)


def test_project_survives_job_cleanup(generated, monkeypatch):
    import os

    from image2live2d.app import server

    g = generated
    monkeypatch.setenv("IMAGE2LIVE2D_JOB_DIR", str(g.root / "jobs"))
    monkeypatch.setenv("IMAGE2LIVE2D_RETENTION_SECONDS", "1")
    g.store = EditorStore(g.root / "jobs/editor-projects")
    project = create(g)
    old = g.root / "jobs/old"
    old.mkdir()
    (old / "job.json").write_text("{}")
    os.utime(old / "job.json", (1, 1))
    server._cleanup_web_jobs()
    assert not old.exists()
    assert EditorStore(g.store.root).info(project["id"])["current"] == project["current"]


def test_http_workflow_and_restart(generated, monkeypatch):
    from image2live2d.app import server

    g = generated
    monkeypatch.setenv("IMAGE2LIVE2D_EDITOR_DIR", str(g.store.root))
    project = create(g)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server._make_handler())
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{http.server_port}"
    def request(path, data=None, good_origin=True):
        headers = {"Origin": origin if good_origin else "http://untrusted.example"}
        req = urllib.request.Request(origin+path, data=json.dumps(data).encode() if data is not None else None,
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    try:
        assert b"editor.js" in request("/editor")
        assert json.loads(request("/api/editor/projects"))[0]["id"] == project["id"]
        endpoint = "/api/editor/projects/"+project["id"]
        draft = json.loads(request(endpoint+"/drafts", {"base": project["current"], "part_id": "mouth"}))
        draft = json.loads(request(endpoint+"/strokes", {"draft": draft["id"], "sequence": 0, "strokes": [stroke()]}))
        assert request(endpoint+"/asset?part=mouth&draft="+draft["id"]).startswith(b"\x89PNG")
        with pytest.raises(urllib.error.HTTPError) as exc:
            request(endpoint+"/apply", {"draft": draft["id"], "sequence": 1, "approved": True}, False)
        assert exc.value.code == 403
        result = json.loads(request(endpoint+"/apply", {"draft": draft["id"], "sequence": 1, "approved": True}))
        assert request(endpoint+"/download?revision="+result["current"]).startswith(b"PK")
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_browser_brush_approval_reload_restore(generated, monkeypatch):
    import shutil

    from image2live2d.app import server

    playwright = pytest.importorskip("playwright.sync_api")
    chromium = shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium unavailable")
    g = generated
    monkeypatch.setenv("IMAGE2LIVE2D_EDITOR_DIR", str(g.store.root))
    # Exercise conversion of a real, completed app job through the HTTP adapter.
    shutil.copytree(g.assets, g.root / "normalized")
    jid = "a" * 32
    job = server._Job(id=jid, name="model", status="done", work=g.root,
                      rig=g.rig, meshes=g.rig.meshes, params=g.rig.parameters,
                      stack=SimpleNamespace(layers=[SimpleNamespace(texture_path=g.assets / "mouth.png")]),
                      source=(g.assets / "face_base.png").read_bytes(), filename="hero.png")
    monkeypatch.setitem(server._JOBS, jid, job)
    http = ThreadingHTTPServer(("127.0.0.1", 0), server._make_handler())
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{http.server_port}"
    try:
        with playwright.sync_playwright() as engine:
            browser = engine.chromium.launch(executable_path=chromium, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("dialog", lambda dialog: dialog.accept())
            page.goto(origin+"/editor")
            created = page.evaluate("""async jid => {
                const r=await fetch('/api/jobs/'+jid+'/editor-project', {method:'POST',
                  headers:{'Content-Type':'application/json'},body:'{}'});
                return {status:r.status, data:await r.json()};
            }""", jid)
            assert created["status"] == 201, created
            project = created["data"]
            page.goto(origin+"/editor?project="+project["id"])
            page.wait_for_function("document.getElementById('status').textContent.includes('입 alpha')")
            box = page.locator("#edited").bounding_box()
            page.mouse.click(box["x"]+box["width"]/2, box["y"]+box["height"]/2)
            page.wait_for_function("document.getElementById('status').textContent.includes('초안 저장 완료')")
            # Reload before approval; the draft is persisted, not just an in-browser mask.
            page.reload()
            page.wait_for_function("document.querySelectorAll('#drafts option').length === 2")
            did = g.store.info(project["id"])["drafts"][0]["id"]
            page.select_option("#drafts", did)
            page.click("#resume-draft")
            page.wait_for_function("document.getElementById('status').textContent.includes('저장된 초안')")
            page.click("#apply")
            page.wait_for_function("document.getElementById('status').textContent.includes('새 버전 저장 완료')")
            saved = g.store.info(project["id"])
            assert saved["current"] != project["current"]
            import os

            screenshot = os.environ.get("EDITOR_TEST_SCREENSHOT")
            if screenshot:
                page.screenshot(path=screenshot, full_page=True)
            page.check("#alpha")
            page.select_option("#background", "black")
            page.locator("#edited").hover()
            page.mouse.wheel(0, -150)
            with page.expect_download() as download:
                page.click("#download")
            assert zipfile.is_zipfile(download.value.path())
            page.select_option("#revisions", project["current"])
            page.wait_for_function("!document.getElementById('restore').disabled")
            page.click("#restore")
            page.wait_for_function("document.getElementById('status').textContent.includes('복원 버전')")
            assert len(g.store.info(project["id"])["revisions"]) == 3
            assert not errors
            browser.close()
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_new_process_reopens_without_generation(generated):
    import subprocess
    import sys

    g = generated
    project = create(g)
    draft = changed(g, project)
    saved = g.store.apply(project["id"], draft["id"], 1, True)
    code = """
import sys
from image2live2d.app import server
from image2live2d.app.editor import EditorStore
assert not server._JOBS
store = EditorStore(sys.argv[1])
p = store.info(sys.argv[2])
assert p['current'] == sys.argv[3]
assert store.download(p['id'], p['current']).startswith(b'PK')
"""
    subprocess.run([sys.executable, "-c", code, str(g.store.root), project["id"], saved["current"]],
                   check=True, capture_output=True, timeout=20)


def test_geometry_guard_rejects_changed_base_bundle(generated):
    g = generated
    project = create(g)
    draft = changed(g, project)
    p = g.store.project(project["id"])
    (p / "revisions" / project["current"] / "bundle/model.moc3").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Geometry"):
        g.store.apply(project["id"], draft["id"], 1, True)
    assert g.store.info(project["id"])["current"] == project["current"]


def test_draft_generation_failures_do_not_change_visible_sequence(generated, monkeypatch):
    from image2live2d.app import editor

    g = generated
    project = create(g)
    draft = g.store.draft(project["id"], project["current"], "mouth")
    def fail(*_):
        raise OSError("disk full")
    monkeypatch.setattr(editor, "atomic_json", fail)
    with pytest.raises(OSError):
        g.store.strokes(project["id"], draft["id"], 0, [stroke()])
    visible = g.store.asset(project["id"], "mouth", draft=draft["id"])
    assert visible.name == "0.png"


def test_gpu_generation_entrypoints_are_never_called(generated, monkeypatch):
    from image2live2d import pipeline
    from image2live2d.app import server

    def forbidden(*args, **kwargs):
        raise AssertionError("Editor must not start generation")
    monkeypatch.setattr(server, "_run_pipeline", forbidden)
    monkeypatch.setattr(server, "resume_job", forbidden)
    monkeypatch.setattr(pipeline, "prepare_meshes", forbidden)
    g = generated
    project = create(g)
    draft = changed(g, project)
    result = g.store.apply(project["id"], draft["id"], 1, True)
    assert g.store.download(project["id"], result["current"])


def test_scaled_atlas_restoration_keeps_layout_and_original_alpha(generated, monkeypatch):
    from image2live2d.app import editor
    from image2live2d.backends.live2d import moc3_emit

    g = generated
    pack = lambda rig, root: g.pack(rig, root, atlas_size=48)
    monkeypatch.setattr(editor, "build_atlas", pack)
    monkeypatch.setattr(moc3_emit, "build_atlas", pack)
    array = np.array(Image.open(g.assets / "mouth.png"))
    array[..., 3] = np.arange(32, dtype=np.uint8)[None, :] * 6 + 20
    Image.fromarray(array).save(g.assets / "mouth.png")
    Live2DEmitter(asset_root=g.assets).build(g.rig, g.bundle)
    project = create(g)
    p = g.store.project(project["id"])
    layout = read(p / "original/layout.json")["mouth"]
    assert layout["tile"][2]-layout["tile"][0] < 32
    draft = changed(g, project)
    edited = g.store.apply(project["id"], draft["id"], 1, True)
    draft = g.store.draft(project["id"], edited["current"], "mouth")
    draft = g.store.strokes(project["id"], draft["id"], 0,
                           [{**stroke("restore"), "radius": 32, "hardness": 1}])
    restored = g.store.apply(project["id"], draft["id"], 1, True)
    with (
        Image.open(p / "original/bundle/textures/atlas.png") as original,
        Image.open(p / "revisions" / restored["current"] / "bundle/textures/atlas.png") as after,
    ):
        assert np.array_equal(original, after)
