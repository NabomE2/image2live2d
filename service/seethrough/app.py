"""See-through decompose service — HTTP wrapper around See-through's PSD inference.

POST an image (raw bytes) to ``/decompose`` and get back a layered ``.psd`` (See-through's native
output), which the image2live2d client (``decompose.from_service``) turns into a LayerStack. Runs on a
GPU box (e.g. a GCP L4/T4 VM — see docs/DECOMPOSE_SERVICE.md).

The default subprocess mode uses isolated LayerDiff/depth/PSD stages with independent deadlines and durable checkpoints.
SEE_THROUGH_MEMORY_PROFILE=legacy retains the original single-process CLI for comparison.
Warm mode is retained only with the legacy profile; it cannot enforce a subprocess deadline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from checkpoints import STAGES, Checkpoints, atomic_json, fingerprint
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
STAGE_TIMEOUTS = {stage: int(os.environ.get(f"SEE_THROUGH_{stage.upper()}_TIMEOUT", str(TIMEOUT)))
                  for stage in STAGES}
RETENTION = int(os.environ.get("SEE_THROUGH_RETENTION_SECONDS", "604800"))
JOB_ROOT = Path(os.environ.get("SEE_THROUGH_JOB_DIR", "/tmp/seethrough-jobs"))
if any(value <= 0 for value in STAGE_TIMEOUTS.values()) or RETENTION <= 0:
    raise ValueError("Stage timeouts and retention must be positive")
MEMORY_PROFILE = os.environ.get("SEE_THROUGH_MEMORY_PROFILE", "staged")
if MEMORY_PROFILE not in {"staged", "low_vram", "legacy"}:
    raise ValueError("SEE_THROUGH_MEMORY_PROFILE must be staged, low_vram or legacy")
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
    restore_jobs()
    threading.Thread(target=_janitor, daemon=True).start()
    if INFERENCE_MODE == "subprocess" or MEMORY_PROFILE in {"staged", "low_vram"}:
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


def _save_job(job_id):
    job = _jobs[job_id]
    if job.get("work"):
        atomic_json(Path(job["work"]) / "job.json", job)


def restore_jobs():
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    with _jobs_lock:
        for path in JOB_ROOT.glob("*/job.json"):
            jid = path.parent.name
            if len(jid) != 32 or any(c not in "0123456789abcdef" for c in jid):
                continue
            try:
                job = json.loads(path.read_text())
                if not isinstance(job, dict):
                    raise TypeError("Invalid job record")
                job["work"] = str(path.parent)
                if job["status"] == "running":
                    job.update(status="error", error="Service restarted; resume completed checkpoints",
                               error_code="interrupted", finished=time.time())
                    for entry in job.get("stages", {}).values():
                        if entry.get("status") == "running":
                            entry.update(status="error", error="Service interrupted")
                _jobs[jid] = job
                _save_job(jid)
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("Ignoring unreadable job %s", jid)
    cleanup_jobs()


def cleanup_jobs():
    now = time.time()
    with _jobs_lock:
        for jid, job in list(_jobs.items()):
            if job["status"] != "running" and now - job.get("finished", job["started"]) > RETENTION:
                if job.get("work"):
                    shutil.rmtree(job["work"], ignore_errors=True)
                del _jobs[jid]
        # Uploads interrupted before registration have no manifest and no checkpoint.
        if JOB_ROOT.exists():
            for directory in JOB_ROOT.iterdir():
                if (directory.is_dir() and directory.name not in _jobs
                        and now - directory.stat().st_mtime > RETENTION):
                    shutil.rmtree(directory, ignore_errors=True)


def _janitor():
    while True:
        time.sleep(min(3600, RETENTION))
        cleanup_jobs()


def _authorize(request):
    if TOKEN and request.headers.get("X-Auth-Token") != TOKEN:
        raise HTTPException(401, "Invalid token")


@app.post("/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str, request: Request, stage: str | None = None,
               stop_after: str | None = None):
    _authorize(request)
    if stage not in {None, *STAGES} or stop_after not in {None, *STAGES}:
        raise HTTPException(400, "Unknown stage")
    if stage and stop_after and STAGES.index(stop_after) < STAGES.index(stage):
        raise HTTPException(400, "stop_after precedes requested stage")
    if MEMORY_PROFILE == "legacy":
        raise HTTPException(409, "Stage resume requires staged or low_vram")
    if not _job_slot.acquire(False):
        raise HTTPException(429, "GPU busy")
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "unknown job id or retention expired")
            if job["status"] == "running":
                raise HTTPException(409, "Job is already running")
            work = Path(job["work"])
            if not (work / "input.png").is_file():
                raise HTTPException(409, "Input expired; upload again")
            job.update(status="running", error=None, error_code=None, psd=None,
                       started=time.time(), telemetry=None)
            job.pop("finished", None)
            _save_job(job_id)
        threading.Thread(target=_worker, args=(job_id, work / "input.png", work),
                         kwargs={"force": stage, "stop_after": stop_after}, daemon=True).start()
    except BaseException:
        if "work" in locals():
            with _jobs_lock:
                job.update(status="error", error="Unable to start resumed worker", finished=time.time())
        _job_slot.release()
        raise
    return {"job_id": job_id}


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
        "stage_timeouts": STAGE_TIMEOUTS, "retention_seconds": RETENTION,
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
        cleanup_jobs()
        work = JOB_ROOT / job_id
        work.mkdir(parents=True)
        src = work / "input.png"
        size = 0
        with src.open("wb") as stream:
            async for chunk in request.stream():
                size += len(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if not size:
            raise HTTPException(status_code=400, detail="empty request body (POST the image bytes)")
        with _jobs_lock:
            _jobs[job_id] = {"status": "running", "psd": None, "error": None,
                             "error_code": None, "started": time.time(), "telemetry": None,
                             "work": str(work), "stages": {stage: {"status": "pending"} for stage in STAGES}}
            _save_job(job_id)
        threading.Thread(target=_worker, args=(job_id, src, work), daemon=True).start()
    except BaseException:
        if "work" in locals():
            shutil.rmtree(work, ignore_errors=True)
            with _jobs_lock:
                _jobs.pop(job_id, None)
        _job_slot.release()
        raise
    return JSONResponse({"job_id": job_id, "checkpoint_protocol": 1}, status_code=202)


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    secs = round(job.get("finished", time.time()) - job["started"], 1)
    return {"status": job["status"], "error": job["error"], "elapsed_seconds": secs,
            "mode": "warm" if _warm else "subprocess",
            "error_code": job.get("error_code"), "telemetry": job.get("telemetry"),
            "stages": job.get("stages", {}), "stage_timeouts": STAGE_TIMEOUTS,
            "resumable": bool(job.get("work")) and MEMORY_PROFILE != "legacy",
            "expires_at": None if job["status"] == "running" else job.get("finished", job["started"]) + RETENTION}


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


def _worker(job_id: str, src: Path, work: Path, *, force=None, stop_after=None) -> None:
    """Run inference in a background thread, recording the result/error on the job."""
    try:
        def record(event):
            with _jobs_lock:
                _jobs[job_id]["telemetry"] = event
                if event.get("event") == "checkpoint":
                    _jobs[job_id]["stages"] = event["stages"]
                    _save_job(job_id)
        if force or stop_after:
            psd = _subprocess_infer(src, on_event=record, force=force, stop_after=stop_after)
        else:
            psd = _infer(src, work / "out", on_event=record)
        with _jobs_lock:
            _jobs[job_id].update(status="done" if psd else "paused", psd=str(psd) if psd else None)
    except Exception as exc:  # noqa: BLE001 - report any failure to the client via the job
        log.warning("job %s failed: %s", job_id, exc)
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=str(exc),
                                 error_code=getattr(exc, "code", "inference_failed"))
    finally:
        try:
            with _jobs_lock:
                _jobs[job_id]["finished"] = time.time()
                try:
                    _save_job(job_id)
                except OSError as exc:
                    _jobs[job_id].update(status="error", error=str(exc), error_code="storage_error")
                    log.error("Unable to persist job %s: %s", job_id, exc)
        finally:
            _job_slot.release()


def _infer(src: Path, out_dir: Path, on_event=lambda event: None) -> Path:
    """Warm in-process inference if available, else the subprocess CLI. Returns the layered .psd."""
    if _warm is not None:
        try:
            return _warm.infer(src, out_dir)
        except WarmUnavailable as exc:
            log.warning("warm infer unavailable (%s) — using subprocess", exc)
    return _subprocess_infer(src, on_event=on_event)


def _subprocess_infer(src: Path, on_event=lambda event: None, *, force=None, stop_after=None) -> Path:
    """Run See-through's CLI in a fresh process; return the newest layered .psd (raises on failure)."""
    cmd = [PYTHON, SCRIPT, "--srcp", str(src), "--save_to_psd"]
    output = src.parent / "output"
    cmd += ["--save_dir", str(output)]
    if os.environ.get("MANAGED_MODELS"):
        paths = models.paths()
        cmd += ["--repo_id_layerdiff", paths["layerdiff"], "--repo_id_depth", paths["depth"]]
    if RESOLUTION:
        cmd += ["--resolution", RESOLUTION]
    elif MEMORY_PROFILE == "low_vram":
        cmd += ["--resolution", "768"]
    if MEMORY_PROFILE == "low_vram":
        cmd += ["--inference_steps", "20"]
    with _subproc_lock:
        before = set(output.rglob("*.psd")) if output.exists() else set()
        if MEMORY_PROFILE in {"staged", "low_vram"}:
            runner = str(Path(__file__).with_name("staged_inference.py"))
            profile_args = ["--low-vram"] if MEMORY_PROFILE == "low_vram" else []
            model_paths = models.paths() if os.environ.get("MANAGED_MODELS") else {}
            if not model_paths:
                # Pin already cached remote snapshots to local paths for reproducible retries.
                try:
                    from huggingface_hub import snapshot_download
                except ImportError:
                    snapshot_download = None
                for name, repo in (("layerdiff", "layerdifforg/seethroughv0.0.2_layerdiff3d"),
                                   ("depth", "24yearsold/seethroughv0.0.1_marigold")):
                    try:
                        if snapshot_download is not None:
                            model_paths[name] = snapshot_download(repo, local_files_only=True)
                    except OSError:
                        pass  # unknown identity deliberately cannot reuse a checkpoint
                if len(model_paths) == 2:
                    cmd += ["--repo_id_layerdiff", model_paths["layerdiff"], "--repo_id_depth", model_paths["depth"]]
            store = Checkpoints(src.parent)
            on_event({"stage": "checkpoints", "event": "validating"})
            identities = fingerprint(src, [*cmd[1:], MEMORY_PROFILE], SEE_THROUGH_DIR, model_paths)
            def record(stage, entry):
                public = {name: {k: v for k, v in value.items()
                                 if k not in {"files", "identity", "directory"}}
                          for name, value in store.state.items()}
                public[stage]["reused"] = entry.get("reused", False)
                on_event({"stage": stage, "event": "checkpoint", "stages": public})
            def run(stage, destination, deadline):
                stage_cmd = cmd.copy()
                stage_cmd[stage_cmd.index("--save_dir") + 1] = str(destination)
                run_process([PYTHON, runner, *profile_args, stage, *stage_cmd[1:]],
                            cwd=str(SEE_THROUGH_DIR), deadline=deadline, stage=stage, on_event=on_event)
            return store.execute(identities, STAGE_TIMEOUTS, run, record, force, stop_after)
        else:
            run_process(cmd, cwd=str(SEE_THROUGH_DIR), deadline=time.monotonic() + TIMEOUT,
                        stage="legacy", on_event=on_event)
        after = set(output.rglob("*.psd")) if output.exists() else set()
    pool = after - before
    # See-through writes both a layered "input.psd" AND a "input_depth.psd"; we want the layers one.
    pool = {p for p in pool if not p.name.endswith("_depth.psd")}
    if not pool:
        raise RuntimeError("see-through produced no .psd")
    return max(pool, key=lambda p: p.stat().st_mtime)
