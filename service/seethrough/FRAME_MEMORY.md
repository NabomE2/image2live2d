# Frame-aware UNet memory scheduling (low_vram)

Base: main `30ba3d5` (PR #4), See-through
`58a1cb11d13f85acec9bbddb8cd4b6487843d4cf`, torch 2.8.0, diffusers 0.37.0.
This supersedes the phase-only residency description in [PR #4 analysis](PERFORMANCE.md).
No CUDA device or pretrained weights were available here. **RTX 4060 8GB speed,
completion within 1800 seconds, peak VRAM/RAM and rendered quality are unverified.**

## Why there may be no completed step log

In PR #4, `layerdiff.step/start` is emitted only after VAE encoding, UNet placement,
prompt lookup/transfers, time IDs, scheduler setup and initial noise creation.
`layerdiff.step/end` requires input preparation, the entire UNet, scheduler update
and CUDA synchronization to finish. Its absence does not prove that no computation
started. Each timing span also synchronizes before its body and after it; a missing
`end` previously could not distinguish a Python call from a driver completion wait.
The provided observations contain no last-stage trace, so the exact runtime stall
cannot be identified from them. High GPU utilization alone does not identify it.

The new events distinguish these cases:

| Last event | Interpretation |
| --- | --- |
| `denoise_prepare/enter` | Adapter entered denoise, before model placement. |
| `prompt_lookup`, `time_ids`, `schedule_setup`, `noise_setup` | Setup spans before the first step. |
| Any span `start` without `ready` | Waiting in its initial CUDA synchronization (or interrupted logging). |
| `ready` without `submitted` | Python body has not returned; may include CUDA dispatch, allocation or synchronous copies. |
| `submitted` without `end` | Body returned; final CUDA synchronization has not finished. |
| `unet_module/enter,submitted` | First use of UNet/conditioning/convolution/normalization/attention/FF blocks in each body's/head's first UNet call. Unsynchronized: submitted is NOT GPU completion. |
| `partition/enter,submitted` | Current independent slice and axis within that first UNet call; all temporal frames retained. |
| `block_load`, `block_evict` | Synchronized transformer weight transfers inside every UNet call. These are included in UNet/step time, not additional compute time. |
| `unet_memory/configured`, `unet_residency/ready` | Partition sizes, logical weight bytes and actual idle CPU/GPU residency. |

All names above have the `layerdiff.` prefix. Existing step progress, elapsed time,
GPU memory fields and failure events remain. Logging/synchronization adds overhead.
A last marker narrows the location, but cannot establish which CUDA kernel or WSL
paging mechanism is responsible. Process peak counters include earlier phases.

## Why whole-UNet frame splitting is invalid

Let the hidden tensor be `X[b,f,s,c]` (sample, frame, spatial token, channel).
Pinned spatial BasicTransformerBlock computes `Y[b,f,:,:] = A(X[b,f,:,:], E[b,f])`.
LayerNorm is over channels, attention over all spatial tokens or conditioning
text tokens, and FF is token-local. There is no reduction over the batch/frame
axis. We run one frame at a time **within this block**, retaining its entire
spatial sequence and slicing frame-specific encoder states/masks consistently.

Pinned CrossFrameTransformerBlock reshapes to `[B*S,F,C]` and computes
`Z[b,:,s,:] = T(Y[b,:,s,:])`. Attention is over **all F frames**; different spatial
positions are independent. We partition S into at most 128 positions, retaining
all frames and original order, then write back the corresponding output slices.
Its `attn2` must be None, as in the pinned constructor. Unsupported conditioning
or normalization fails explicitly rather than silently changing the model.

These factorizations preserve each block's mathematical function; by composition,
interleaving the original spatial/temporal blocks preserves dependencies. Floating
point kernels may differ with batch shapes, so bitwise identity is not promised.
Frame group embeddings, shared initial noise, scheduler, residual skip tensors,
convolutions and whole-UNet frame ordering are unchanged. Naive independent frame
UNet calls would remove temporal attention and are deliberately not implemented.
Training/gradient execution bypasses partitioning; weight offload is inference-only.

## Memory changes and tradeoffs

- Only convolution/conditioning and other non-transformer weights remain on CUDA.
  Each spatial/temporal transformer block is loaded before use and evicted afterward.
  One block is active at a time; all of its partitions reuse that loaded block.
  Bootstrap also uses this placement, preserving the CUDA generator/device contract
  without initially loading every transformer weight onto CUDA.
- Logical GPU weight budget becomes resident weights plus the largest active block,
  rather than all UNet weights. Actual allocation also includes activations,
  allocator cache/workspaces, and other process allocations. CPU-held weights and
  synchronous PCIe transfers increase host memory/traffic; this can slow inference.
  No fixed VRAM bound or speedup is claimed.
- Spatial/temporal intermediate work is bounded by these independent partitions.
  A full input/output and UNet skip tensors still exist; this is not complete
  activation offload. Outputs use preallocation rather than a list plus concatenate.
- BF16 policy, SDPA, FF chunking, VAE tiling, 768px/20 steps, prompts and seeds remain
  unchanged. No timeout extension, forced attention kernel, quantization or reduction
  in output frames is introduced. Staged and Live2D paths are unchanged.

20 steps limits repetition, not the memory peak inside one call. A BF16
`[13,320,96,96]` activation alone is 73.125 MiB at 768px; several residuals and
attention/FF tensors coexist with weights. VAE tiling does not shrink UNet weights.
The reported 7.75/8.91 GiB values therefore warrant reducing UNet residency, even
when VAE offload works. WSL shared-memory paging remains a hypothesis, not a finding.

## Reproduce validation and GPU diagnosis

CI fetches the exact upstream revision above without weights. CPU regression tests
compare real pinned temporal and interleaved transformer implementations (FP32 and
BF16), frame-specific masks/conditioning, body/head group-conditioned tiny UNets,
remaining slices and multi-sample grouping. They include a counterexample showing
that naive frame splitting changes temporal outputs, placement ordering and trace
cleanup. These are small random-weight numerical tests, not pretrained image or
CUDA-transfer tests. Full tests retain staged and Live2D coverage.

```bash
# Use an upstream checkout at the revision above:
SEE_THROUGH_TEST_SOURCE=/path/to/see-through .venv/bin/python -m pytest -q -ra
```

For the RTX run, follow the rebuild/health/log capture commands in
[PERFORMANCE.md](PERFORMANCE.md#new-diagnostics-low_vram-only). Keep the same PNG,
model revisions and timeout. Confirm `unet_memory/configured`, CPU-resident idle
transformer blocks and GPU-resident convolution weights; compare transfer spans,
first-UNet module/partition markers and completed step times. Observe Windows
shared/dedicated GPU memory and host RAM alongside CUDA allocated/reserved values.
A successful run and image comparison on the actual GPU remain required before
claiming performance improvement or visual equivalence with pretrained weights.
