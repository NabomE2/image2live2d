"""PNG/PSD transport through the real web pipeline and validated model downloads.

The CPU test replaces only GPU inference with an HTTP contract fixture returning
a real layered PSD. The opt-in test uses a real PNG and real See-through service.
Neither test certifies the proprietary runtime or VTube Studio.
"""
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image
from psd_tools import PSDImage

from image2live2d.app import server
from image2live2d.backends.live2d.bundle import bundle_zip, validate_bundle
from image2live2d.samples import make_sample_layers


@contextmanager
def running_http(handler):
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)


def request(base, path, data=None):
    return urllib.request.urlopen(base + path, data=data, timeout=120)


def complete_job(base, png, timeout=60):
    with request(base, "/api/jobs?name=character.png", png) as response:
        jid = json.load(response)["job_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with request(base, "/api/jobs/" + jid) as response:
            job = json.load(response)
        if job["status"] != "running":
            assert job["status"] == "done", job
            return jid
        time.sleep(0.05)
    pytest.fail("PNG conversion timed out")


def download_model(base, jid, target):
    with request(base, f"/api/jobs/{jid}/live2d-status") as response:
        status = json.load(response)
    assert status == {"ready": True, "structural_validation": "passed",
                      "cubism_core": "unverified", "vtube_studio": "unverified"}
    with request(base, f"/api/jobs/{jid}/live2d-download") as response:
        assert response.headers["Content-Type"] == "application/zip"
        assert 'filename="character_vtube.zip"' in response.headers["Content-Disposition"]
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        assert all(name.startswith("model/") for name in archive.namelist())
        archive.extractall(target)
    model = target / "model/model.model3.json"
    validate_bundle(model)
    assert (target / "model/model.moc3").stat().st_size > 1024
    assert (target / "model/textures/atlas.png").is_file()
    return model


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    """Valid PNG upload and PSD response; no ML model or copyrighted art."""
    layers = make_sample_layers(tmp_path / "layers", size=128)
    psd = PSDImage.new("RGB", (128, 128))
    flat = Image.new("RGBA", (128, 128))
    for file in sorted(layers.glob("*.png")):
        with Image.open(file) as image:
            flat.alpha_composite(image.convert("RGBA"))
            psd.create_pixel_layer(image.convert("RGBA"), name=file.stem)
    png = io.BytesIO()
    flat.save(png, "PNG")
    data = io.BytesIO()
    psd.save(data)
    submitted = []

    class InferenceFixture(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/decompose"
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert body == png.getvalue()
            with Image.open(io.BytesIO(body)) as image:
                assert image.format == "PNG" and image.size == (128, 128)
            submitted.append(body)
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b'{"job_id":"fixture"}')

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path == "/jobs/fixture":
                self.wfile.write(b'{"status":"done"}')
            elif self.path == "/jobs/fixture/result":
                self.wfile.write(data.getvalue())
            else:
                raise AssertionError(self.path)

        def log_message(self, *args):
            pass

    with running_http(InferenceFixture) as url:
        monkeypatch.setenv("IMAGE2LIVE2D_DECOMPOSE_URL", url)
        monkeypatch.delenv("IMAGE2LIVE2D_DECOMPOSE_TOKEN", raising=False)
        yield png.getvalue(), submitted


def test_png_to_rig_to_live2d_zip_over_http(tmp_path, model_files):
    png, submitted = model_files
    with running_http(server._make_handler()) as base:
        jid = complete_job(base, png)
        model = download_model(base, jid, tmp_path / "download")
        assert len(submitted) == 1
        # Existing .inp download remains available.
        with request(base, f"/api/jobs/{jid}/download") as response:
            assert response.read().startswith(b"TRNSRTS")
        # A missing file after generation must not be reported as ready or downloadable.
        bdir = server.live2d_bundle_dir(server.get_job(jid))
        (bdir / "model.moc3").unlink()
        for route in ("live2d-status", "live2d-download"):
            with pytest.raises(urllib.error.HTTPError) as error:
                request(base, f"/api/jobs/{jid}/{route}")
            assert error.value.code == 422
            assert "Missing" in json.load(error.value)["error"]
    assert "model.moc3" in validate_bundle(model)


def test_png_public_api_produces_complete_zip(tmp_path, model_files):
    from image2live2d import convert_image

    png, submitted = model_files
    path = tmp_path / "input.png"
    path.write_bytes(png)
    result = convert_image(path, tmp_path / "api", service_url=os.environ["IMAGE2LIVE2D_DECOMPOSE_URL"])
    assert len(submitted) == 1
    assert result.live2d_zip_path.is_file()
    validate_bundle(result.live2d_path)


def test_png_cli_produces_complete_zip(tmp_path, model_files):
    from image2live2d.__main__ import main

    png, submitted = model_files
    path = tmp_path / "input.png"
    path.write_bytes(png)
    directory = tmp_path / "cli"
    assert main(["--image", str(path), "--decompose-url", os.environ["IMAGE2LIVE2D_DECOMPOSE_URL"],
                 "--work-dir", str(tmp_path / "cli_layers"), "-o", str(tmp_path / "cli.inp"),
                 "--live2d", str(directory)]) == 0
    assert len(submitted) == 1
    assert (tmp_path / "cli.zip").is_file()
    validate_bundle(next(directory.glob("*.model3.json")))


@pytest.mark.skipif(not (os.environ.get("I2L_TEST_PNG") and os.environ.get("I2L_TEST_DECOMPOSE_URL")),
                    reason="real PNG + CUDA service not configured; GPU inference UNVERIFIED")
def test_real_png_gpu_pipeline(tmp_path, monkeypatch):
    path = Path(os.environ["I2L_TEST_PNG"])
    with Image.open(path) as image:
        assert image.format == "PNG"
    monkeypatch.setenv("IMAGE2LIVE2D_DECOMPOSE_URL", os.environ["I2L_TEST_DECOMPOSE_URL"])
    monkeypatch.delenv("IMAGE2LIVE2D_GCP_INSTANCE", raising=False)
    with running_http(server._make_handler()) as base:
        jid = complete_job(base, path.read_bytes(), timeout=1900)
        download_model(base, jid, tmp_path / "real-download")


@pytest.fixture
def bundle(tmp_path):
    from image2live2d import convert_layers
    layers = make_sample_layers(tmp_path / "layers", size=128)
    return convert_layers(layers, tmp_path / "export", live2d=True).live2d_path


@pytest.mark.parametrize("damage", ["moc", "texture", "motion", "physics", "parameter",
                                   "texture_index", "traversal", "windows_path"])
def test_invalid_bundle_cannot_be_downloaded(bundle, damage):
    from image2live2d.backends.live2d.moc3_binary import read_moc3, write_moc3
    manifest = json.loads(bundle.read_text())
    refs = manifest["FileReferences"]
    if damage in ("moc", "texture", "motion", "physics"):
        rel = {"moc": refs["Moc"], "texture": refs["Textures"][0],
               "motion": next(iter(refs["Motions"].values()))[0]["File"],
               "physics": refs["Physics"]}[damage]
        (bundle.parent / rel).unlink()
    elif damage == "parameter":
        manifest["Groups"][0]["Ids"] = ["ParamAbsent"]
    elif damage == "texture_index":
        path = bundle.parent / refs["Moc"]
        moc = read_moc3(path.read_bytes())
        moc.get("artMeshes", "textureNos")[0] = 999
        path.write_bytes(write_moc3(moc))
    elif damage == "traversal":
        refs["Moc"] = "../outside.moc3"
    elif damage == "windows_path":
        refs["Textures"] = ["C:\\outside.png"]
    bundle.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle_zip(bundle)


def test_unfinished_job_cannot_download():
    job = server._Job(id="unfinished", name="incomplete")
    with server._LOCK:
        server._JOBS[job.id] = job
    try:
        with running_http(server._make_handler()) as base:
            with pytest.raises(urllib.error.HTTPError) as error:
                request(base, "/api/jobs/unfinished/live2d-download")
            assert error.value.code == 409
    finally:
        server._JOBS.pop(job.id, None)
