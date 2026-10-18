"""Persistent orchestration contracts; no CUDA/model performance claims."""
import importlib
import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def cp(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "service/seethrough"))
    return importlib.import_module("checkpoints")


@pytest.fixture
def identities():
    return {name: name for name in ("layerdiff", "depth", "psd")}


def test_failed_partial_depth_restarts_without_layerdiff(cp, tmp_path, stage_output, identities):
    calls, events = [], []
    def run(stage, path, deadline):
        calls.append(stage)
        stage_output(path, stage)
        if stage == "depth":
            (path / "input/src_img.png").write_bytes(b"partially overwritten")
            raise TimeoutError("depth timed out")
    store = cp.Checkpoints(tmp_path)
    budgets = dict.fromkeys(identities, 12)
    with pytest.raises(TimeoutError):
        store.execute(identities, budgets, run, lambda *a: events.append(a))
    assert store.valid("layerdiff", "layerdiff")
    assert not store.valid("depth", "depth")
    assert not list(store.root.glob("attempt-*"))
    store = cp.Checkpoints(tmp_path)  # simulate process restart
    calls.clear()
    def finish(stage, path, deadline):
        calls.append(stage)
        assert 11 < deadline - time.monotonic() <= 12
        stage_output(path, stage)
    result = store.execute(identities, budgets, finish, lambda *a: None)
    assert result.name == "input.psd" and calls == ["depth", "psd"]
    calls.clear()
    store.execute(identities, budgets, finish, lambda *a: None)
    assert not calls
    store.execute(identities, budgets, finish, lambda *a: None, force="depth", stop_after="depth")
    assert calls == ["depth"] and store.state["psd"]["status"] == "pending"


def test_corrupt_checkpoint_invalidates_descendants(cp, tmp_path, stage_output, identities):
    store = cp.Checkpoints(tmp_path)
    calls = []
    def run(stage, path, deadline):
        calls.append(stage)
        stage_output(path, stage)
    budgets = dict.fromkeys(identities, 30)
    store.execute(identities, budgets, run, lambda *a: None)
    source = store.root / store.state["layerdiff"]["directory"] / "input/src_img.png"
    source.write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="layerdiff checkpoint is invalid"):
        store.execute(identities, budgets, run, lambda *a: None, force="depth")
    assert not calls
    store.execute(identities, budgets, run, lambda *a: None)
    assert calls == list(identities)


def test_zero_exit_without_required_artifacts_is_not_done(cp, tmp_path, identities):
    store = cp.Checkpoints(tmp_path)
    with pytest.raises(ValueError, match="src_img"):
        store.execute(identities, dict.fromkeys(identities, 10), lambda *a: None, lambda *a: None)
    assert store.state["layerdiff"]["status"] == "error"


def test_unpublished_directory_is_never_reused(cp, tmp_path, stage_output, identities):
    store = cp.Checkpoints(tmp_path)
    orphan = store.root / "layerdiff-orphan"
    stage_output(orphan, "layerdiff")
    assert not store.valid("layerdiff", identities["layerdiff"])
    store.prune()
    assert not orphan.exists()


def test_fingerprint_content_and_dependency_invalidation(cp, tmp_path):
    src = tmp_path / "input.png"
    src.write_bytes(b"original")
    models = {}
    for name in ("layerdiff", "depth"):
        root = tmp_path / name
        root.mkdir()
        (root / "weights.bin").write_bytes(b"weights")
        models[name] = str(root)
    def snapshot(command=None):
        return cp.fingerprint(src, command or ["script.py", "low_vram"], tmp_path, models)
    before = snapshot()
    assert snapshot() == before
    (Path(models["depth"]) / "weights.bin").write_bytes(b"changed")
    after = snapshot()
    assert before["layerdiff"] == after["layerdiff"]
    assert before["depth"] != after["depth"] and before["psd"] != after["psd"]
    assert snapshot(["script.py", "staged"])["layerdiff"] != after["layerdiff"]
    src.write_bytes(b"different")
    assert snapshot()["layerdiff"] != after["layerdiff"]
    assert cp.fingerprint(src, ["script.py"], tmp_path, {}) != cp.fingerprint(src, ["script.py"], tmp_path, {})


def test_job_restart_recovery_and_retention(cp, monkeypatch, tmp_path):
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "JOB_ROOT", tmp_path)
    monkeypatch.setattr(app, "_jobs", {})
    jid = "a" * 32
    root = tmp_path / jid
    cp.atomic_json(root / "job.json", {"status": "running", "started": time.time(),
                   "stages": {"layerdiff": {"status": "done"}, "depth": {"status": "running"}}})
    app.restore_jobs()
    job = app._jobs[jid]
    assert job["status"] == "error" and job["error_code"] == "interrupted"
    assert job["stages"]["layerdiff"]["status"] == "done"
    assert job["stages"]["depth"]["status"] == "error"
    job["finished"] = time.time() - app.RETENTION - 1
    app.cleanup_jobs()
    assert not root.exists() and jid not in app._jobs


def test_resume_authorization_busy_and_paused(cp, monkeypatch, tmp_path):
    from types import SimpleNamespace

    app = importlib.import_module("app")
    monkeypatch.setattr(app, "TOKEN", "test-token")
    with pytest.raises(app.HTTPException) as exc:
        app.resume_job("unknown", SimpleNamespace(headers={}))
    assert exc.value.status_code == 401
    request = SimpleNamespace(headers={"X-Auth-Token": "test-token"})
    assert app._job_slot.acquire(False)
    try:
        with pytest.raises(app.HTTPException) as exc:
            app.resume_job("unknown", request)
        assert exc.value.status_code == 429
    finally:
        app._job_slot.release()


def test_independent_timeouts(cp, tmp_path, stage_output, identities, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(cp.time, "monotonic", lambda: clock[0])
    observed = []
    def run(stage, path, deadline):
        observed.append(deadline - clock[0])
        clock[0] += 9
        stage_output(path, stage)
    cp.Checkpoints(tmp_path).execute(identities, {"layerdiff": 10, "depth": 20, "psd": 30},
                                    run, lambda *a: None)
    assert observed == [10, 20, 30]


def test_worker_api_resume_depth_and_preserve_result(cp, monkeypatch, tmp_path, stage_output):
    from types import SimpleNamespace

    app = importlib.import_module("app")
    monkeypatch.setattr(app, "MEMORY_PROFILE", "low_vram")
    monkeypatch.setattr(app, "TOKEN", None)
    monkeypatch.setattr(app, "_jobs", {})
    monkeypatch.setenv("MANAGED_MODELS", "1")
    monkeypatch.setattr(app.models, "paths", lambda: {"layerdiff": str(tmp_path / "model-l"),
                                                    "depth": str(tmp_path / "model-d")})
    for name in ("model-l", "model-d"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "weights").write_bytes(b"fixed model")
    jid = "b" * 32
    work = tmp_path / jid
    work.mkdir()
    src = work / "input.png"
    src.write_bytes(b"input")
    app._jobs[jid] = {"status": "running", "started": time.time(), "work": str(work),
                      "error": None, "psd": None}
    calls = []
    fail = [True]
    def run(cmd, **kw):
        stage = kw["stage"]
        calls.append(stage)
        stage_output(Path(cmd[cmd.index("--save_dir") + 1]), stage)
        if stage == "depth" and fail[0]:
            raise RuntimeError("interrupted depth")
    monkeypatch.setattr(app, "run_process", run)
    assert app._job_slot.acquire(False)
    app._worker(jid, src, work)
    status = app.job_status(jid)
    assert status["status"] == "error"
    assert status["stages"]["layerdiff"]["status"] == "done"
    assert status["stages"]["depth"]["status"] == "error"
    fail[0] = False
    pending = []
    class Thread:
        def __init__(self, target, args, kwargs, daemon):
            pending.append(lambda: target(*args, **kwargs))
        def start(self):
            pass
    monkeypatch.setattr(app.threading, "Thread", Thread)
    request = SimpleNamespace(headers={})
    assert app.resume_job(jid, request, stage="depth")["job_id"] == jid
    with pytest.raises(app.HTTPException) as error:
        app.resume_job(jid, request)
    assert error.value.status_code == 429
    pending.pop()()
    assert calls == ["layerdiff", "depth", "depth", "psd"]
    status = app.job_status(jid)
    assert status["status"] == "done" and status["stages"]["layerdiff"]["reused"]
    assert Path(app.job_result(jid).path).is_file()
    assert json.loads((work / "job.json").read_text())["status"] == "done"


def test_atomic_manifest_failure_does_not_publish_checkpoint(cp, monkeypatch, tmp_path, stage_output, identities):
    store = cp.Checkpoints(tmp_path)
    original = cp.atomic_json
    failed = [False]
    def interrupted(path, value):
        if value.get("layerdiff", {}).get("status") == "done" and not failed[0]:
            failed[0] = True
            raise OSError("disk write failed")
        original(path, value)
    monkeypatch.setattr(cp, "atomic_json", interrupted)
    with pytest.raises(OSError, match="disk write failed"):
        store.execute(identities, dict.fromkeys(identities, 10),
                      lambda stage, path, deadline: stage_output(path, stage), lambda *a: None)
    assert not cp.Checkpoints(tmp_path).valid("layerdiff", "layerdiff")


def test_persistence_failure_always_releases_gpu_slot(cp, monkeypatch, tmp_path):
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "_jobs", {"disk": {"started": time.time(), "status": "running"}})
    monkeypatch.setattr(app, "_infer", lambda *a, **k: tmp_path / "output.psd")
    def fail(job_id):
        raise OSError("disk full")
    monkeypatch.setattr(app, "_save_job", fail)
    assert app._job_slot.acquire(False)
    app._worker("disk", tmp_path / "input.png", tmp_path)
    assert app._jobs["disk"]["error_code"] == "storage_error"
    assert app._job_slot.acquire(False)
    app._job_slot.release()


@pytest.mark.parametrize("content", ['partial JSON', '[]', '{"layerdiff":"bad"}'])
def test_corrupt_manifest_is_not_reused(cp, tmp_path, content):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "manifest.json").write_text(content)
    assert not cp.Checkpoints(tmp_path).valid("layerdiff", "key")
