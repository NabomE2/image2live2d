"""Run one stage of the pinned upstream CLI, preserving its arguments and output layout.

Each invocation exits before the next model is loaded. No quantization, resolution,
step count, scheduler or attention changes. This adapter targets upstream 58a1cb11.
"""
from __future__ import annotations

import functools
import os
import runpy
import sys
import threading
import time
from contextlib import contextmanager

from telemetry import emit


@contextmanager
def measured(name):
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    start = time.monotonic()
    emit(name, "start")
    try:
        yield
    finally:
        # Do not mask an OOM with another CUDA call during error unwinding.
        torch = sys.modules.get("torch")
        if sys.exc_info()[0] is None and torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        emit(name, "end", elapsed_seconds=round(time.monotonic() - start, 3))


def adapt(iu, torch):
    """Load in inference dtype and retire prompt encoders before allocating the UNet on CUDA."""
    for cls in (iu.TransparentVAE, iu.UNetFrameConditionModel,
                iu.KDiffusionStableDiffusionXLPipeline, iu.MarigoldDepthPipeline):
        original = cls.from_pretrained

        def load(*args, _original=original, _name=cls.__name__, **kwargs):
            with measured(f"load.{_name}"):
                kwargs.update(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
                obj = _original(*args, **kwargs)
            if _name == "KDiffusionStableDiffusionXLPipeline":
                with measured("layerdiff.prompt_cache"):
                    obj.text_encoder.to(device="cuda", dtype=torch.bfloat16)
                    obj.text_encoder_2.to(device="cuda", dtype=torch.bfloat16)
                    with torch.inference_mode():
                        obj.cache_tag_embeds()
            return obj

        cls.from_pretrained = staticmethod(load)

    for cls in (iu.KDiffusionStableDiffusionXLPipeline, iu.MarigoldDepthPipeline):
        original_call = cls.__call__

        @functools.wraps(original_call)
        def call(self, *args, _original=original_call, _name=cls.__name__, **kwargs):
            # LayerDiff calls this once for body and once for head (unless head is empty).
            with measured(f"infer.{_name}"):
                return _original(self, *args, **kwargs)

        cls.__call__ = call


def main():
    stage, script, *args = sys.argv[1:]
    if stage not in {"layerdiff", "depth", "psd"}:
        raise ValueError(f"Unknown stage: {stage}")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "8"
    emit(stage, "process_start")
    stopped = threading.Event()

    def sample():
        while not stopped.wait(5):
            emit(stage, "sample")

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        with measured("imports"):
            import torch
            from utils import inference_utils as iu
        adapt(iu, torch)
        selected = {"layerdiff": "apply_layerdiff", "depth": "apply_marigold", "psd": "further_extr"}[stage]
        for name in ("apply_layerdiff", "apply_marigold", "further_extr"):
            if name != selected:
                setattr(iu, name, lambda *a, **kw: None)
        sys.argv = [script, *args]
        with measured(stage):
            runpy.run_path(script, run_name="__main__")
    except Exception as exc:
        code = "inference_failed"
        if isinstance(exc, MemoryError):
            code = "host_memory_exhausted"
        elif "torch" in sys.modules and isinstance(exc, sys.modules["torch"].cuda.OutOfMemoryError):
            code = "cuda_oom"
        emit(stage, "error", error_code=code, error=str(exc))
        raise
    finally:
        stopped.set()
        sampler.join(timeout=1)


if __name__ == "__main__":
    main()
