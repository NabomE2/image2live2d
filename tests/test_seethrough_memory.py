"""CPU contract tests; these do not establish GPU fit, speed, or model quality."""
import asyncio
import importlib
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE = Path(__file__).resolve().parents[1] / "service" / "seethrough"


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVICE))
    return importlib.import_module("process_runner")


def test_stream_events_and_bounded_failure_output(backend, tmp_path):
    events = []
    script = tmp_path / "child.py"
    script.write_text('print(\'SEETHROUGH_EVENT {"stage":"load", "event":"start"}\', flush=True)\n'
                      'print("x" * 200000)\nraise RuntimeError("last failure")\n')
    with pytest.raises(backend.InferenceError) as error:
        backend.run_process([sys.executable, str(script)], cwd=tmp_path,
                            deadline=time.monotonic() + 10, stage="test", on_event=events.append)
    assert events == [{"stage": "load", "event": "start"}]
    assert error.value.code == "inference_failed"
    assert "last failure" in str(error.value)
    assert len(str(error.value)) < 6200


def test_timeout_kills_process_group(backend, tmp_path):
    script = tmp_path / "child.py"
    child_pid = tmp_path / "pid"
    script.write_text('import subprocess, sys, time\nfrom pathlib import Path\n'
                      'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
                      'Path(sys.argv[1]).write_text(str(p.pid))\ntime.sleep(60)\n')
    with pytest.raises(backend.InferenceError, match="timeout"):
        backend.run_process([sys.executable, str(script), str(child_pid)], cwd=tmp_path,
                            deadline=time.monotonic() + 1, stage="depth")
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/status")
    # A killed orphan can remain a zombie until container init reaps it.
    assert not status.exists() or "State:\tZ" in status.read_text()


@pytest.mark.parametrize("rc,tail,before,after,reported,expected", [
    (1, "CUDA out of memory", {}, {}, None, "cuda_oom"),
    (1, "MemoryError", {}, {}, None, "host_memory_exhausted"),
    (-signal.SIGKILL, "", {"cgroup_memory_events": {"oom_kill": 2}},
     {"cgroup_memory_events": {"oom_kill": 3}}, None, "host_memory_exhausted"),
    (-signal.SIGKILL, "", {}, {}, None, "process_killed"),
    (1, "", {}, {}, "cuda_oom", "cuda_oom"),
    (1, "bad input", {}, {}, None, "inference_failed"),
])
def test_failure_classification(backend, rc, tail, before, after, reported, expected):
    assert backend.classify_failure(rc, tail, before, after, reported) == expected


def test_expired_shared_deadline_does_not_start_next_stage(backend, tmp_path):
    marker = tmp_path / "started"
    with pytest.raises(backend.InferenceError, match="before depth"):
        backend.run_process([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                            cwd=tmp_path, deadline=time.monotonic() - 1, stage="depth")
    assert not marker.exists()


@pytest.fixture
def app_module(backend, monkeypatch):
    monkeypatch.delenv("MANAGED_MODELS", raising=False)
    return importlib.import_module("app")


def test_three_stages_share_deadline_and_model_arguments(app_module, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app_module, "MEMORY_PROFILE", "staged")
    monkeypatch.setattr(app_module, "RESOLUTION", None)
    monkeypatch.setenv("MANAGED_MODELS", "1")
    monkeypatch.setattr(app_module.models, "paths", lambda: {"layerdiff": "/models/l", "depth": "/models/d"})
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if kwargs["stage"] == "psd":
            output = tmp_path / "output"
            output.mkdir()
            (output / "input.psd").write_bytes(b"PSD")
            (output / "input_depth.psd").write_bytes(b"depth")
    monkeypatch.setattr(app_module, "run_process", run)
    assert app_module._subprocess_infer(tmp_path / "input.png").name == "input.psd"
    assert [kw["stage"] for _, kw in calls] == ["layerdiff", "depth", "psd"]
    assert len({kw["deadline"] for _, kw in calls}) == 1
    for cmd, _ in calls:
        assert cmd[cmd.index("--repo_id_layerdiff") + 1] == "/models/l"
        assert cmd[cmd.index("--repo_id_depth") + 1] == "/models/d"
        assert "--resolution" not in cmd and "--inference_steps" not in cmd


def test_busy_rejected_before_reading_upload(app_module):
    async def body():
        pytest.fail("busy service must not read image into RAM")
    request = SimpleNamespace(headers={}, body=body)
    assert app_module._job_slot.acquire(False)
    try:
        with pytest.raises(app_module.HTTPException) as exc:
            asyncio.run(app_module.decompose(request))
        assert exc.value.status_code == 429
    finally:
        app_module._job_slot.release()


def test_worker_error_is_exposed_and_slot_released(app_module, backend, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise backend.InferenceError("cuda_oom", "test exhaustion")
    monkeypatch.setattr(app_module, "_infer", fail)
    app_module._jobs["test"] = {"started": time.time()}
    assert app_module._job_slot.acquire(False)
    app_module._worker("test", tmp_path / "input.png", tmp_path)
    status = app_module.job_status("test")
    assert status["status"] == "error" and status["error_code"] == "cuda_oom"
    assert app_module._job_slot.acquire(False)
    app_module._job_slot.release()


def test_loader_dtype_and_prompt_cache_order(backend, monkeypatch):
    runner = importlib.import_module("staged_inference")
    events = []
    class Component:
        def to(self, **kw):
            events.append(("encoder.to", kw))
    def create_class(name):
        class Model:
            text_encoder = Component()
            text_encoder_2 = Component()
            @classmethod
            def from_pretrained(cls, *a, **kw):
                events.append((name, kw))
                return cls()
            def cache_tag_embeds(self):
                events.append(("cache", {}))
            def __call__(self):
                return "same output"
        Model.__name__ = name
        return Model
    names = ["TransparentVAE", "UNetFrameConditionModel", "KDiffusionStableDiffusionXLPipeline", "MarigoldDepthPipeline"]
    iu = SimpleNamespace(**{name: create_class(name) for name in names})
    from contextlib import nullcontext
    torch = SimpleNamespace(bfloat16="bf16", inference_mode=nullcontext)
    monkeypatch.setattr(runner, "emit", lambda *a, **kw: None)
    runner.adapt(iu, torch)
    obj = iu.KDiffusionStableDiffusionXLPipeline.from_pretrained("repo", scheduler=None)
    assert events == [(names[2], {"scheduler": None, "torch_dtype": "bf16", "low_cpu_mem_usage": True}),
                      ("encoder.to", {"device": "cuda", "dtype": "bf16"}),
                      ("encoder.to", {"device": "cuda", "dtype": "bf16"}), ("cache", {})]
    assert obj() == "same output"


def test_real_png_through_staged_cli_adapter(backend, tmp_path):
    """Real PNG + actual adapter/process runner, fake ML backend; no GPU quality claim."""
    from PIL import Image
    Image.new("RGBA", (16, 16), (100, 20, 30, 255)).save(tmp_path / "input.png")
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "__init__.py").write_text("")
    (tmp_path / "torch.py").write_text('class Cuda:\n def is_initialized(self): return False\n'
                                       ' class OutOfMemoryError(Exception): pass\ncuda=Cuda()\nbfloat16="bf16"\n')
    (tmp_path / "utils" / "inference_utils.py").write_text('''
from pathlib import Path
from PIL import Image
class Model:
    @classmethod
    def from_pretrained(cls, *a, **kw): return cls()
    def __call__(self): pass
TransparentVAE = type('TransparentVAE', (Model,), {})
UNetFrameConditionModel = type('UNetFrameConditionModel', (Model,), {})
KDiffusionStableDiffusionXLPipeline = type('KDiffusionStableDiffusionXLPipeline', (Model,), {})
MarigoldDepthPipeline = type('MarigoldDepthPipeline', (Model,), {})
def apply_layerdiff():
    Image.open('input.png').save('parts.png')
def apply_marigold():
    assert Path('parts.png').exists()
    Path('depth.json').write_text('{}')
def further_extr():
    assert Path('depth.json').exists()
    from psd_tools import PSDImage
    PSDImage.new('RGBA', (16,16)).save('input.psd')
''')
    script = tmp_path / "upstream.py"
    script.write_text('from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr\n'
                      'apply_layerdiff()\napply_marigold()\nfurther_extr()\n')
    # Child imports mock dependencies from PYTHONPATH, while running the production adapter.
    import os
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(tmp_path)
    events = []
    try:
        deadline = time.monotonic() + 20
        for stage in ("layerdiff", "depth", "psd"):
            backend.run_process([sys.executable, str(SERVICE / "staged_inference.py"), stage, str(script)],
                                cwd=tmp_path, deadline=deadline, stage=stage, on_event=events.append)
    finally:
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old
    from psd_tools import PSDImage
    assert PSDImage.open(tmp_path / "input.psd").size == (16, 16)
    assert [e["stage"] for e in events if e["event"] == "process_start"] == ["layerdiff", "depth", "psd"]
    assert all("elapsed_seconds" in e for e in events if e["event"] == "end")
    assert json.dumps(events)


def test_failed_upload_releases_slot(app_module):
    async def stream():
        yield b"partial image"
        raise OSError("upload disconnected")
    request = SimpleNamespace(headers={}, stream=stream)
    with pytest.raises(OSError, match="disconnected"):
        asyncio.run(app_module.decompose(request))
    assert app_module._job_slot.acquire(False)
    app_module._job_slot.release()
