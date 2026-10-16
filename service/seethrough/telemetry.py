"""Best-effort Linux/container memory observations; no CUDA initialization for monitoring."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PREFIX = "SEETHROUGH_EVENT "


def memory_snapshot(pid: int | None = None) -> dict:
    values = {}
    for path, keys in ((f"/proc/{pid or os.getpid()}/status", {"VmRSS": "rss_bytes", "VmHWM": "peak_rss_bytes"}),
                       ("/proc/meminfo", {"MemAvailable": "system_available_bytes", "SwapFree": "swap_free_bytes"})):
        try:
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition(":")
                if key in keys:
                    values[keys[key]] = int(value.split()[0]) * 1024
        except (OSError, ValueError):
            pass
    # Docker uses a private cgroup namespace by default. Missing v2 counters are omitted.
    for name in ("current", "max", "events"):
        try:
            raw = Path(f"/sys/fs/cgroup/memory.{name}").read_text().strip()
            values[f"cgroup_memory_{name}"] = (
                {k: int(v) for k, v in (line.split() for line in raw.splitlines())}
                if name == "events" else (None if raw == "max" else int(raw))
            )
        except (OSError, ValueError):
            pass
    return values


def emit(stage: str, event: str, **extra) -> None:
    data = {"stage": stage, "event": event, "time": time.time(), **memory_snapshot(), **extra}
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_initialized():
        data.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(PREFIX + json.dumps(data), flush=True)
