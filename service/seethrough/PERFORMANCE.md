# low_vram LayerDiff timeout investigation

Base: image2live2d `167242f` (PR #3 merged), See-through
`58a1cb11d13f85acec9bbddb8cd4b6487843d4cf`.

Reported hardware: RTX 4060 8GB, RAM 32GB, WSL2/Docker. The report says
`low_vram`, `sdpa=true`, `ff_chunk_tokens=1024`, `vae_tile_pixels=512`, normal
loading/prompt caching, a 1800-second inference timeout, approximately 7.75GiB
allocated at inference start and a maximum of approximately 8.91GiB, without
an OOM kill. These are user observations, not measurements reproduced here.
**No CUDA device or model weights are available in this development environment.
RTX 4060 completion, speed, peak VRAM/RAM and visual quality remain unverified.**

## Findings from code

1. Offloading is between phases, not between UNet blocks or denoising steps.
   `Placement` captures the original bound `.to` methods before bootstrap VAE
   moves are deferred. Those original methods evict inactive weights to CPU
   before moving the active weights to CUDA. The deferred `.to` wrapper does
   not intercept the controller's moves. Model construction/prompt caching alone
   does not prove these transitions actually succeeded on the user's machine.
2. Pinned `apply_layerdiff` initially puts the UNet on CUDA so it can construct
   the seeded CUDA generator. At first `_encode`, the controller moves it back
   to CPU, then reloads it for denoising. The existing outer inference-start
   event occurs **before** that eviction. High allocation at that event is not
   proof of failed offload. This initial round trip is retained to preserve the
   generator/device contract, and is now timed explicitly.
3. At 768px, the latent grid is 96×96. The body processes 13 frames together;
   the head processes 11. Upstream `UNetFrameConditionModel.forward` reshapes
   batch×frames and retains `down_block_res_samples` until the up blocks consume
   them. For scale, a single BF16 `[13,320,96,96]` activation is 73.125MiB;
   simultaneous residuals, attention and FF intermediates add to resident UNet
   weights. This is an illustrative tensor size, not a measured model peak.
4. 20 steps reduces repeated work, not the peak of a single step. FF chunking
   bounds token-local intermediate work, not attention/residual tensors or
   weights. 512px tiling affects the SD VAE, not the UNet or custom transparency
   decoder. SDPA configuration does not identify which kernel PyTorch selects;
   it is not proof that Flash Attention executed.
5. The previous FF implementation kept **every** chunk output in a list and
   then allocated the full concatenated output. During concatenation both
   output sets were live. In inference mode this avoidable duplication is now
   replaced by one preallocated output plus one live chunk. Chunk size, dtype,
   layer ordering, attention, resolution, step count, seed and scheduler are
   unchanged. The autograd path retains concatenation. CPU tests check numeric
   parity, chunk release, absence of final concatenation and gradients.

Points 1–5 are source findings. They do **not** establish which component caused
this user's 1800-second timeout or quantify any GPU speed/memory improvement.

## New diagnostics (low_vram only)

Rebuild and recreate the container; `restart` alone does not update its code.
Keep the same input, model revisions and existing timeout for the first comparison.
No timeout, quality or staged-mode defaults are changed by this instrumentation.

```bash
SEE_THROUGH_MEMORY_PROFILE=low_vram docker compose --profile gpu up -d --build
# Confirm the profile via the companion, whose port remains unpublished:
docker compose exec seethrough python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')))"
# Start recording before uploading a PNG; stop the log follower after the job ends.
docker compose logs -f --no-color seethrough > seethrough-profile.log
```

Every event retains the `SEETHROUGH_EVENT` JSON prefix, timestamps, Linux memory
observations and, once CUDA is initialized, PyTorch allocated/reserved/current
process-peak bytes. No CUDA peak counters are reset. `group_index=0` is body,
`group_index=1` is head; an empty head can legitimately skip the second call.

| Stage/event | Meaning |
| --- | --- |
| `layerdiff.low_vram/configured` | Settings and initial logical parameter/buffer bytes, grouped by device/dtype; `ff_output=preallocated_inference` identifies this implementation. |
| `layerdiff.bootstrap_deferred` | Requested VAE CUDA move and effective CPU residency. |
| `layerdiff.bootstrap_unet` | Initial UNet move; separate from later offloading. |
| `layerdiff.inference/start` | Group, resolution, frames, requested steps, torch version, CUDA device name and total device bytes. |
| `layerdiff.model_transfer` | Model name, target, before/after device and dtype byte counts, elapsed time. A post-move device mismatch fails explicitly. No-op CPU moves are also visible. |
| `layerdiff.placement/active` | Actual parameter/buffer residency after the transition, alongside requested active models. In denoise, only UNet should have CUDA weights; VAE and transparency decoder should be CPU. |
| `layerdiff.tensor_transfer` | Adapter-managed copy/cast, its role, source/target, source tensor bytes, group and optional frame, elapsed time. Same-device transfers can be no-ops. |
| `layerdiff.denoise/start` | Actual latent shape, frame count, scheduler timestep count and requested steps. |
| `layerdiff.step/start,end` | 1-based `step`, `steps`, group, `progress` from 0 to 1, wall time plus `unet_seconds`, `prepare_seconds`, `scheduler_seconds`, current/peak GPU memory. |
| `layerdiff.step_prepare`, `layerdiff.unet`, `layerdiff.scheduler` | Separate boundaries for scheduler input scaling/concatenation, the UNet call, and scheduler update. A start with no end locates a stalled/incomplete operation. |
| `layerdiff.encode`, `layerdiff.decode` | VAE compute phases; decode is per output frame. The upstream decoder's internal output-to-CPU conversions are included in decode time, not separately attributed. |

Timing spans synchronize CUDA at boundaries when already initialized. They do
not initialize CUDA on CPU-only tests. `elapsed_seconds` is synchronized wall
clock time (CPU work, allocations, driver waits and CUDA completion), **not
kernel-only time or isolated PCIe bandwidth**. `boundary_wait_seconds` records
work queued before the span. Step totals include nested logging/synchronization
overhead and need not equal the sum of the component times. Profiling perturbs
execution; compare runs with the same instrumentation.

On failure, an `error` event marks timing incomplete. There is no extra CUDA
synchronization while unwinding that failure. SIGKILL/deadline termination may
leave only a `start` event and the existing runner timeout message. `peak_scope`
is `process`: peak values include earlier loading, encoding and steps, not just
that step. CUDA tensor allocation counters are not physical dedicated VRAM,
Windows shared-memory usage, or total GPU use by other processes.

The adapter's explicit condition/prompt/final-latent copies occur outside the
step loop. A long `unet` span includes anything the upstream UNet, PyTorch or
WSL driver does internally; it does **not** prove there were no hidden transfers
or driver paging. This patch does not enable block offload, NVTX, CUPTI, or a
kernel profiler, and does not claim to attribute those hidden operations.

## Hypotheses requiring the next GPU run

- If UNet weights alone leave little space, activations can create pressure
  despite successful VAE offload. Compare `placement` weight bytes with allocated,
  reserved and device total bytes; they measure different things.
- WSL/WDDM shared-memory paging, host RAM pressure or competing workloads could
  explain long wall times without an OOM kill. None is established by the report.
  Observe Windows dedicated/shared GPU memory, RAM and WSL swap alongside the log.
- Many small FF chunks can add dispatch overhead; SDPA may select a slower or
  more memory-intensive backend for some shapes. Changing chunk size or forcing
  an attention backend without measurements can trade latency for OOM risk, so
  neither is changed here.
- The reported 8.91GiB needs the original field name, byte value, device total,
  units and event time checked. It must not be presented as proven dedicated
  physical VRAM usage, nor as proof of an offload failure.

Collect complete logs through timeout/success, not only 5-second memory samples.
Locate the last start/end pair, compare model/tensor transfer times against per
step UNet times and body/head progression, and confirm the measured residency.
An unfinished step has no trustworthy completion duration. Repeat the same PNG
and cached models to distinguish cold loading from steady inference. If further
optimization requires quantization, block offload, smaller resolution or fewer
steps, evaluate that as a separate quality/memory change against these results.

## Validation and preserved paths

CPU tests cover phase order and actual device-inventory reporting, failed
eviction, timing synchronization/error preservation, step progress and memory
fields, full body/head tensor paths, and streaming FF parity/lifetimes/gradients.
The full suite also covers staged/legacy routing, PNG/PSD conversion and Live2D
export. Staged inference, depth inference and Live2D implementation are unchanged.
The full existing Ruff baseline remains a separate issue; no blanket lint fixes
or suppressions are introduced for that baseline.
