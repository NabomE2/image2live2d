"""Bounded streaming logs, a shared deadline, and process-group cleanup for inference."""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque

from telemetry import PREFIX, memory_snapshot

log = logging.getLogger("seethrough")


class InferenceError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def classify_failure(returncode, tail, before, after, reported=None):
    if reported in {"cuda_oom", "host_memory_exhausted"}:
        return reported
    if "CUDA out of memory" in tail or "torch.OutOfMemoryError" in tail:
        return "cuda_oom"
    if "MemoryError" in tail or "DefaultCPUAllocator: can't allocate memory" in tail:
        return "host_memory_exhausted"
    previous = before.get("cgroup_memory_events", {}).get("oom_kill", 0)
    current = after.get("cgroup_memory_events", {}).get("oom_kill", 0)
    if returncode == -signal.SIGKILL and current > previous:
        return "host_memory_exhausted"
    return "process_killed" if returncode < 0 else "inference_failed"


def run_process(cmd, *, cwd, deadline, stage, on_event=lambda event: None):
    if time.monotonic() >= deadline:
        raise InferenceError("timeout", f"Total inference deadline exceeded before {stage}")
    before = memory_snapshot()
    tail = deque(maxlen=32)  # never retain unbounded inference stdout in service RAM
    reported = []
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})

    def read_output():
        assert proc.stdout is not None
        # readline(size) also bounds a single enormous line from a third-party library.
        while chunk := proc.stdout.readline(4096):
            line = chunk.decode("utf-8", errors="replace").rstrip()
            tail.append(line)
            log.info("[%s] %s", stage, line)
            if line.startswith(PREFIX):
                try:
                    event = json.loads(line[len(PREFIX):])
                    if event.get("error_code"):
                        reported.append(event["error_code"])
                    on_event(event)
                except (ValueError, TypeError, AttributeError):
                    log.warning("Invalid telemetry event")

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    timed_out = False
    try:
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                proc.wait(timeout=min(remaining, 1))
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Also reap descendants that inherited the output pipe, even if the leader exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        reader.join(timeout=5)
        if not reader.is_alive():
            proc.stdout.close()
    if timed_out:
        raise InferenceError("timeout", f"Total inference deadline exceeded during {stage}; process group stopped")
    if proc.returncode:
        output = "\n".join(tail)[-6000:]
        code = classify_failure(proc.returncode, output, before, memory_snapshot(),
                                reported[-1] if reported else None)
        raise InferenceError(code, f"{stage} failed (exit {proc.returncode}): {output}")
