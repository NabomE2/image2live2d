"""Synchronized wall-clock spans for low_vram only; no CUDA initialization."""
from __future__ import annotations

import time
from contextlib import contextmanager, suppress

from telemetry import emit


def synchronize(torch):
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


@contextmanager
def span(torch, stage, **fields):
    """Separate earlier queued work from this operation; preserve original failures.

    These are wall times, including CPU work, allocations and CUDA completion,
    not GPU kernel-only timings. Never reset the process-wide CUDA peak counters.
    """
    emit(stage, "start", **fields)
    started = time.perf_counter()
    ready = started
    result = {}
    try:
        synchronize(torch)
        ready = time.perf_counter()
        yield result
        synchronize(torch)
    except BaseException:
        # In particular, don't synchronize again or mask OOM during unwinding.
        # Telemetry is best effort only while preserving an active inference error.
        with suppress(Exception):
            emit(stage, "error", **fields, timing_incomplete=True,
                 elapsed_seconds=round(time.perf_counter() - ready, 6))
        raise
    else:
        result["elapsed_seconds"] = round(time.perf_counter() - ready, 6)
        emit(stage, "end", **fields, **result,
             boundary_wait_seconds=round(ready - started, 6), peak_scope="process")


def inventory(module):
    """Logical parameter+buffer bytes, deduplicated by tensor identity, not storage."""
    devices, dtypes, seen = {}, {}, set()
    for tensor in (*module.parameters(), *module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        size = tensor.numel() * tensor.element_size()
        device, dtype = str(tensor.device), str(tensor.dtype)
        devices[device] = devices.get(device, 0) + size
        dtypes[dtype] = dtypes.get(dtype, 0) + size
    return {"device_bytes": devices, "dtype_bytes": dtypes,
            "weight_bytes": sum(devices.values())}


def transfer(torch, tensor, *, device, dtype=None, role, group_index=None, **fields):
    source = str(tensor.device)
    with span(torch, "layerdiff.tensor_transfer", role=role, group_index=group_index,
              source=source, target=str(device), tensor_bytes=tensor.numel() * tensor.element_size(),
              **fields):
        return tensor.to(device=device, dtype=dtype or tensor.dtype)
