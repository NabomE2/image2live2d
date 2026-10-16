"""See-through decompose service — HTTP wrapper around See-through's PSD inference.

POST an image (raw bytes) to ``/decompose`` and get back a layered ``.psd`` (See-through's native
output), which the image2live2d client (``decompose.from_service``) turns into a LayerStack. Runs on a
GPU box (e.g. a GCP L4/T4 VM — see docs/DECOMPOSE_SERVICE.md).

The default subprocess mode uses isolated LayerDiff/depth/PSD stages with a shared deadline.
SEE_THROUGH_MEMORY_PROFILE=legacy retains the original single-process CLI for comparison.
Warm mode is retained only with the legacy profile; it cannot enforce a subprocess deadline.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from model_manager import ModelManager
from pipeline import WarmPipeline, WarmUnavailable
from process_runner import run_process

models = ModelManager()

log = logging.getLogger("seethrough")
logging.basicConfig(level=logging.INFO)

SEE_THROUGH_DIR = Path(os.environ.get("SEE_THROUGH_DIR", "/opt/see-through"))
PYTHON = os.environ.get("SEE_THROUGH_PYTHON", "python")
SCRIPT = os.environ.get("SEE_THROUGH_SCRIPT", "inference/scripts/inference_psd.py")
OUTPUT_DIR = Path(os.environ.get("SEE_THROUGH_OUTPUT", SEE_THROUGH_DIR / "workspace/layerdiff_output"))
TOKEN = os.environ.get("SEE_THROUGH_TOKEN")
RESOLUTION = os.environ.get("RESOLUTION")
TIMEOUT = int(os.environ.get("SEE_THROUGH_TIMEOUT", "900"))
if TIMEOUT <= 0:
    raise ValueError("SEE_THROUGH_TIMEOUT must be positive")
MEMORY_PROFILE = os.environ.get("SEE_THROUGH_MEMORY_PROFILE", "staged")
if MEMORY_PROFILE not in {"staged", "legacy"}:
    raise ValueError("SEE_THROUGH_MEMORY_PROFILE must be staged or legacy")
# Default to the known-good subprocess path. Warm in-process is opt-in (INFERENCE_MODE=warm) until
# See-through's apply_* API is confirmed on the box — its real signature is positional and our warm
# wrapper is best-effort, so leaving it default would silently fall back per request.
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "subprocess").lower()

app = FastAPI(title="see-through decompose service", version="0.3.0")
_subproc_lock = threading.Lock()  # one GPU -> serialize the subprocess path
_warm: WarmPipeline | None = None
_job_slot = threading.BoundedSemaphore(1)

# Async job registry: inference takes minutes, so /decompose returns a job id immediately and the
# client polls — no single HTTP connection is held open across the whole inference (which a NAT/idle
# timeout would drop). job_id -> {"status": running|done|error, "psd": path|None, "error": str|None}.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    global _warm
    if INFERENCE_MODE == "subprocess" or MEMORY_PROFILE == "staged":
        log.info("INFERENCE_MODE=subprocess (warm pipeline disabled)")
        return
    try:
        wp = WarmPipeline(SEE_THROUGH_DIR, resolution=int(RESOLUTION) if RESOLUTION else None)
        wp.load()
        _warm = wp
        log.info("warm in-process pipeline loaded")
    except WarmUnavailable as exc:
        log.warning("warm pipeline unavailable (%s) — falling back to subprocess", exc)
        _warm = None


@app.get("/setup")
def setup_status():
    return models.status()


@app.post("/setup/models")
def install_models(request: Request):
    if TOKEN and request.headers.get("X-Auth-Token") != TOKEN:
        raise HTTPException(401, "Invalid token")
    if not models.status()["gpu"]:
        raise HTTPException(409, "NVIDIA CUDA GPU is unavailable")
    models.install()
    return models.status()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "mode": "warm" if _warm else "subprocess",
        "see_through_dir": str(SEE_THROUGH_DIR),
        "exists": SEE_THROUGH_DIR.is_dir(),
        "memory_profile": MEMORY_PROFILE,
        "timeout_seconds": TIMEOUT,
    }


@app.post("/decompose", status_code=202)
async def decompose(request: Request) -> JSONResponse:
    """Start an async decompose job; returns {"job_id"} immediately. Poll GET /jobs/{id}."""
    if TOKEN and request.headers.get("X-Auth-Token") != TOKEN:
        raise HTTPException(status_code=401, detail="missing/invalid X-Auth-Token")
    if os.environ.get("MANAGED_MODELS") and not models.status()["ready"]:
        raise HTTPException(409, "Install the local models in the app first")
    if not _job_slot.acquire(blocking=False):
        raise HTTPException(429, "GPU busy: wait for the current job before uploading again")
    try:
        job_id = uuid.uuid4().hex
        work = Path(tempfile.mkdtemp(prefix="st_job_"))
        src = work / "input.png"
        size = 0
        with src.open("wb") as stream:
            async for chunk in request.stream():
                size += len(chunk)
                stream.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="empty request body (POST the image bytes)")
        with _jobs_lock:
            _jobs[job_id] = {"status": "running", "psd": None, "error": None,
                             "error_code": None, "started": time.time(), "telemetry": None}
        threading.Thread(target=_worker, args=(job_id, src, work), daemon=True).start()
    except BaseException:
        _job_slot.release()
        raise
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    secs = round(job.get("finished", time.time()) - job["started"], 1)
    return {"status": job["status"], "error": job["error"], "elapsed_seconds": secs,
            "mode": "warm" if _warm else "subprocess",
            "error_code": job.get("error_code"), "telemetry": job.get("telemetry")}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "inference failed")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="job not finished")
    psd = Path(job["psd"])
    return FileResponse(psd, media_type="image/vnd.adobe.photoshop", filename=psd.name)


def _worker(job_id: str, src: Path, work: Path) -> None:
    """Run inference in a background thread, recording the result/error on the job."""
    try:
        def record(event):
            with _jobs_lock:
                _jobs[job_id]["telemetry"] = event
        psd = _infer(src, work / "out", on_event=record)
        with _jobs_lock:
            _jobs[job_id].update(status="done", psd=str(psd))
    except Exception as exc:  # noqa: BLE001 - report any failure to the client via the job
        log.warning("job %s failed: %s", job_id, exc)
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc),
                                 error_code=getattr(exc, "code", "inference_failed"))
    finally:
        with _jobs_lock:
            _jobs[job_id]["finished"] = time.time()
        _job_slot.release()


def _infer(src: Path, out_dir: Path, on_event=lambda event: None) -> Path:
    """Warm in-process inference if available, else the subprocess CLI. Returns the layered .psd."""
    if _warm is not None:
        try:
            return _warm.infer(src, out_dir)
        except WarmUnavailable as exc:
            log.warning("warm infer unavailable (%s) — using subprocess", exc)
    return _subprocess_infer(src, on_event=on_event)


def _subprocess_infer(src: Path, on_event=lambda event: None) -> Path:
    """Run See-through's CLI in a fresh process; return the newest layered .psd (raises on failure)."""
    cmd = [PYTHON, SCRIPT, "--srcp", str(src), "--save_to_psd"]
    output = src.parent / "output"
    cmd += ["--save_dir", str(output)]
    if os.environ.get("MANAGED_MODELS"):
        paths = models.paths()
        cmd += ["--repo_id_layerdiff", paths["layerdiff"], "--repo_id_depth", paths["depth"]]
    if RESOLUTION:
        cmd += ["--resolution", RESOLUTION]
    with _subproc_lock:
        before = set(output.rglob("*.psd")) if output.exists() else set()
        deadline = time.monotonic() + TIMEOUT
        if MEMORY_PROFILE == "staged":
            runner = str(Path(__file__).with_name("staged_inference.py"))
            for stage in ("layerdiff", "depth", "psd"):
                run_process([PYTHON, runner, stage, *cmd[1:]], cwd=str(SEE_THROUGH_DIR),
                            deadline=deadline, stage=stage, on_event=on_event)
        else:
            run_process(cmd, cwd=str(SEE_THROUGH_DIR), deadline=deadline,
                        stage="legacy", on_event=on_event)
        after = set(output.rglob("*.psd")) if output.exists() else set()
    pool = after - before
    # See-through writes both a layered "input.psd" AND a "input_depth.psd"; we want the layers one.
    pool = {p for p in pool if not p.name.endswith("_depth.psd")}
    if not pool:
        raise RuntimeError("see-through produced no .psd")
    return max(pool, key=lambda p: p.stat().st_mtime)
