"""Timing and memory-lifetime contracts on CPU; no hardware speed claim."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "service/seethrough"))
    return importlib.import_module("low_vram_metrics")


def test_span_synchronizes_and_separates_prior_queue_wait(metrics, monkeypatch):
    events = []
    clock = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(metrics, "emit", lambda stage, event, **kw: events.append((event, kw)))
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True,
                                                synchronize=lambda: events.append(("sync", {}))))
    with metrics.span(torch, "layerdiff.unet", step=1) as result:
        events.append(("call", {}))
    assert [event for event, _ in events] == ["start", "sync", "call", "sync", "end"]
    assert result["elapsed_seconds"] == 3
    assert events[-1][1]["boundary_wait_seconds"] == 2
    assert events[-1][1]["peak_scope"] == "process"


def test_error_does_not_resynchronize_or_mask_original_exception(metrics, monkeypatch):
    calls = []

    def emit(stage, event, **kw):
        calls.append(event)
        if event == "error":
            raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(metrics, "emit", emit)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True,
                                                synchronize=lambda: calls.append("sync")))
    with pytest.raises(MemoryError, match="original"), metrics.span(torch, "layerdiff.unet"):
        raise MemoryError("original")
    assert calls == ["start", "sync", "error"]


def test_cpu_span_never_initializes_cuda(metrics, monkeypatch):
    monkeypatch.setattr(metrics, "emit", lambda *a, **kw: None)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False,
                                                synchronize=lambda: pytest.fail("CUDA init")))
    with metrics.span(torch, "cpu") as result:
        pass
    assert result["elapsed_seconds"] >= 0


def test_inventory_includes_parameters_and_buffers(metrics):
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(8, 4, dtype=torch.float32)
    module.register_buffer("counter", torch.zeros(3, dtype=torch.int64))
    record = metrics.inventory(module)
    assert record == {"device_bytes": {"cpu": 168},
                      "dtype_bytes": {"torch.float32": 144, "torch.int64": 24},
                      "weight_bytes": 168}


def test_placement_detects_failed_eviction(metrics, monkeypatch):
    low = importlib.import_module("low_vram")
    monkeypatch.setattr(metrics, "emit", lambda *a, **kw: None)
    tensor = SimpleNamespace(device="cuda:0", dtype="bf16", numel=lambda: 2,
                             element_size=lambda: 2)
    # A broken/hooked .to that leaves weights on CUDA must not be reported as offloaded.
    module = SimpleNamespace(parameters=lambda: [tensor], buffers=list, to=lambda **kw: None)
    pipe = SimpleNamespace(unet=module, vae=module, trans_vae=SimpleNamespace(decoder=module))
    torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(is_initialized=lambda: False))
    with pytest.raises(RuntimeError, match="placement mismatch for unet"):
        low.Placement(pipe, torch).activate("vae")


def test_ff_releases_each_chunk_and_avoids_cat(metrics, monkeypatch):
    torch = pytest.importorskip("torch")
    import weakref

    low = importlib.import_module("low_vram")
    refs = []

    class FF(torch.nn.Module):
        def forward(self, data):
            assert all(ref() is None for ref in refs)
            output = data * 2
            refs.append(weakref.ref(output))
            return output

    module = FF()
    low.chunk_feed_forward(module, torch, tokens=17)
    data = torch.randn(13, 9, 8).transpose(0, 1)
    monkeypatch.setattr(torch, "cat", lambda *a, **kw: pytest.fail("full output duplication"))
    with torch.inference_mode():
        output = module(data)
        torch.testing.assert_close(output, data * 2)
    assert len(refs) == 7 and all(ref() is None for ref in refs)


def test_ff_gradient_path_preserved(metrics):
    torch = pytest.importorskip("torch")
    from copy import deepcopy

    low = importlib.import_module("low_vram")
    module = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.GELU(), torch.nn.Linear(12, 8))
    reference = deepcopy(module)
    low.chunk_feed_forward(module, torch, tokens=7)
    data = torch.randn(11, 3, 8, requires_grad=True)
    other = data.detach().clone().requires_grad_()
    module(data).sum().backward()
    reference(other).sum().backward()
    torch.testing.assert_close(data.grad, other.grad)
    for actual, expected in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_span_logs_start_end_cuda_memory_without_reset(metrics, monkeypatch, capsys):
    import json
    import sys

    telemetry = importlib.import_module("telemetry")
    memory = {"allocated": 100, "reserved": 200}
    cuda = SimpleNamespace(is_initialized=lambda: True, synchronize=lambda: None,
                           memory_allocated=lambda: memory["allocated"],
                           memory_reserved=lambda: memory["reserved"],
                           max_memory_allocated=lambda: 900,
                           max_memory_reserved=lambda: 1000)
    torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(telemetry, "memory_snapshot", dict)
    with metrics.span(torch, "layerdiff.step", step=1, steps=20, group_index=0):
        memory.update(allocated=150, reserved=300)
    rows = [json.loads(line.split(telemetry.PREFIX)[1])
            for line in capsys.readouterr().out.splitlines()]
    assert [row["event"] for row in rows] == ["start", "end"]
    assert [row["cuda_allocated_bytes"] for row in rows] == [100, 150]
    assert [row["cuda_reserved_bytes"] for row in rows] == [200, 300]
    assert all(row["cuda_peak_allocated_bytes"] == 900 for row in rows)
    assert all(row["cuda_peak_reserved_bytes"] == 1000 for row in rows)
