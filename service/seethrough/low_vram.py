"""Opt-in LayerDiff adapter for See-through 58a1cb11 (Apache-2.0).

The denoising equations and output contract follow upstream
common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py. Only the CLI's
fullpage, batch=1, guidance=1 path is supported; unsupported calls fail explicitly.
No model weights are quantized. GPU fit and quality need hardware validation.
"""
from __future__ import annotations

import gc
import types

from low_vram_metrics import inventory, span, transfer
from telemetry import emit


class Placement:
    """Evict inactive weights *before* loading the next phase's weights."""

    def __init__(self, pipe, torch):
        self.torch = torch
        self.models = {"unet": pipe.unet, "vae": pipe.vae, "decoder": pipe.trans_vae.decoder}
        self.moves = {name: model.to for name, model in self.models.items()}
        self.group_index = None

    def move(self, name, device):
        before = inventory(self.models[name])
        with span(self.torch, "layerdiff.model_transfer", model=name, target=device,
                  group_index=self.group_index, before=before) as timing:
            self.moves[name](device=device, dtype=self.torch.bfloat16)
            after = inventory(self.models[name])
            timing["after"] = after
            if any(d.split(":")[0] != device for d in after["device_bytes"]):
                raise RuntimeError(f"low_vram placement mismatch for {name}: {after}")

    def activate(self, *names):
        for name in self.models:
            if name not in names:
                self.move(name, "cpu")
        gc.collect()
        self.torch.cuda.empty_cache()
        for name in names:
            self.move(name, "cuda")
        emit("layerdiff.placement", "active", models=list(names), group_index=self.group_index,
             residency={name: inventory(model) for name, model in self.models.items()})

    def park(self):
        self.activate()


def chunk_feed_forward(module, torch, tokens=1024):
    """Bound intermediate GEGLU activations across both spatial and frame axes.

    FeedForward is token-local. Flattening leading dimensions avoids upstream's
    divisibility restriction (body=13 frames, head=11) and accepts a short tail.
    Attention still sees all frames/tokens; cross-frame attention is not split.
    """
    original = module.forward

    def forward(self, hidden_states, *args, **kwargs):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        if flat.shape[0] <= tokens:
            return original(hidden_states, *args, **kwargs)
        if torch.is_grad_enabled():
            # Keep the existing autograd path for callers outside inference_mode.
            chunks = [original(part, *args, **kwargs) for part in flat.split(tokens, dim=0)]
            result = torch.cat(chunks, dim=0)
        else:
            # Retain one chunk, not every output plus a second full-size cat buffer.
            result = None
            for offset in range(0, flat.shape[0], tokens):
                chunk = original(flat[offset:offset + tokens], *args, **kwargs)
                if result is None:
                    result = chunk.new_empty((flat.shape[0], chunk.shape[-1]))
                result[offset:offset + chunk.shape[0]].copy_(chunk)
                del chunk
        return result.reshape(*shape[:-1], result.shape[-1])

    module.forward = types.MethodType(forward, module)


def configure(pipe, torch):
    """Called after prompt caching, before upstream's unconditional CUDA moves."""
    from diffusers.models.attention import FeedForward
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    if hasattr(pipe, "_low_vram_placement"):
        return
    for root in (pipe.unet, pipe.vae, pipe.trans_vae):
        root.eval()
        for module in root.modules():
            if isinstance(module, Attention):
                module.set_processor(AttnProcessor2_0())
            if isinstance(module, FeedForward):
                chunk_feed_forward(module, torch)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    # Explicit 512px SD-VAE tiles; the custom transparency decoder remains
    # whole-image, one layer at a time. Tiling can change pixels near seams.
    pipe.vae.tile_sample_min_size = 512
    pipe.vae.tile_latent_min_size = 512 // pipe.vae_scale_factor
    pipe._low_vram_placement = Placement(pipe, torch)

    # Pinned apply_layerdiff moves all three models to CUDA after loading.
    # Defer ONLY the VAE moves; keep the UNet CUDA placement so upstream creates
    # a CUDA RNG with seed 42, as in staged. Placement retains the real .to.
    for model in (pipe.vae, pipe.trans_vae):
        original_to = model.to

        def defer(self, *, device=None, dtype=None, _move=original_to):
            with span(torch, "layerdiff.bootstrap_deferred", model=type(self).__name__,
                      requested_device=str(device), effective_device="cpu") as timing:
                result = _move(device="cpu", dtype=dtype)
                timing["residency"] = inventory(self)
                return result

        model.to = types.MethodType(defer, model)
    original_unet_to = pipe.unet.to

    def bootstrap_unet(self, *args, **kwargs):
        with span(torch, "layerdiff.bootstrap_unet") as timing:
            result = original_unet_to(*args, **kwargs)
            timing["residency"] = inventory(self)
            return result

    pipe.unet.to = types.MethodType(bootstrap_unet, pipe.unet)
    emit("layerdiff.low_vram", "configured", sdpa=True, ff_chunk_tokens=1024,
         vae_tile_pixels=512, ff_output="preallocated_inference",
         residency={name: inventory(model)
                    for name, model in pipe._low_vram_placement.models.items()})


def _encode(pipe, fullpage, torch, ops):
    import numpy as np

    pipe._low_vram_placement.activate("vae")
    # Upstream uses an opaque alpha channel for conditioning and use_offset=False.
    # Avoid its unused transparency-offset network and RGB-padding allocations.
    rgb = fullpage[..., :3]
    argb = np.concatenate([np.full_like(rgb[..., :1], 255), rgb], axis=2)
    group = pipe._low_vram_placement.group_index
    tensor = transfer(torch, ops.img2tensor(argb, normalize=True), device=pipe.vae.device,
                      dtype=pipe.vae.dtype, role="encode_input", group_index=group)
    with span(torch, "layerdiff.encode", group_index=group):
        feed = (tensor[:, 1:] * 2 - 1) * tensor[:, :1]
        latent = pipe.vae.encode(feed).latent_dist.sample() * pipe.vae.config.scaling_factor
    return transfer(torch, latent, device="cpu", role="condition_out", group_index=group)


def _denoise(pipe, condition, prompt, steps, generator, group_index, torch, ops):
    pipe._low_vram_placement.activate("unet")
    device, dtype = pipe.unet.device, pipe.unet.dtype
    frames = len(prompt)
    condition = transfer(torch, condition, device=device, dtype=dtype,
                         role="condition_in", group_index=group_index)
    h, w = condition.shape[-2:]
    condition = condition[:, None].expand(-1, frames, -1, -1, -1)
    embeds, pooled = pipe.encode_cropped_prompt_77tokens_cached(prompt)
    embeds = transfer(torch, embeds, device=device, role="prompt_embeds", group_index=group_index)
    pooled = transfer(torch, pooled, device=device, role="pooled_embeds", group_index=group_index)
    size = (h * pipe.vae_scale_factor, w * pipe.vae_scale_factor)
    time_ids = torch.tensor([list(size + (0, 0) + size)], dtype=dtype, device=device)
    time_ids = time_ids.expand(frames, -1)
    timesteps, count = ops.retrieve_timesteps(pipe.scheduler, steps, device,
                                             timesteps=None, sigmas=None)
    # Preserve upstream's shared noise across frames and CUDA generator.
    noise = ops.randn_tensor((1, 1, 4, h, w), generator=generator, device=device, dtype=dtype)
    latents = noise.expand(-1, frames, -1, -1, -1) * pipe.scheduler.init_noise_sigma
    del noise
    emit("layerdiff.denoise", "start", group_index=group_index, frames=frames,
         latent_shape=list(latents.shape), steps=len(timesteps), requested_steps=steps,
         adapter_transfers="outside_step_loop")
    with pipe.progress_bar(total=count) as progress:
        for i, t in enumerate(timesteps):
            fields = {"group_index": group_index, "step": i + 1, "steps": len(timesteps)}
            with span(torch, "layerdiff.step", **fields) as step_timing:
                with span(torch, "layerdiff.step_prepare", **fields) as prepare:
                    model_input = pipe.scheduler.scale_model_input(latents, t)
                    unet_input = torch.cat([model_input, condition], dim=-3)
                with span(torch, "layerdiff.unet", **fields) as unet:
                    prediction = pipe.unet(
                        unet_input, t, encoder_hidden_states=embeds,
                        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                        return_dict=False, group_index=group_index,
                    )[0]
                del unet_input
                with span(torch, "layerdiff.scheduler", **fields) as scheduler:
                    latents = pipe.scheduler.step(prediction, t, latents, return_dict=False)[0]
                    del model_input, prediction
                step_timing.update(unet_seconds=unet["elapsed_seconds"],
                                   prepare_seconds=prepare["elapsed_seconds"],
                                   scheduler_seconds=scheduler["elapsed_seconds"],
                                   progress=(i + 1) / len(timesteps))
            if i == len(timesteps) - 1 or (i + 1) % pipe.scheduler.order == 0:
                progress.update()
    # Conditioning and loop tensors die before the VAE is loaded.
    return transfer(torch, latents[0], device="cpu", role="denoised_latents",
                    group_index=group_index)


def _decode(pipe, latents, fullpage, torch, ops):
    pipe._low_vram_placement.activate("vae", "decoder")
    group = pipe._low_vram_placement.group_index
    alpha = transfer(torch, ops.img2tensor(fullpage[..., -1] / 255., dtype=pipe.vae.dtype),
                     device=pipe.vae.device, role="alpha_mask", group_index=group)[0][..., None]
    images, previews = [], []
    for index, latent in enumerate(latents):
        latent = transfer(torch, latent[None], device=pipe.vae.device, dtype=pipe.vae.dtype,
                          role="decode_latent", group_index=group, frame=index + 1)
        with span(torch, "layerdiff.decode", group_index=group, frame=index + 1,
                  frames=len(latents), includes_internal_output_transfers=True):
            latent = latent / pipe.vae.config.scaling_factor
            result, preview = pipe.trans_vae.decoder(pipe.vae, latent, mask=alpha)
        images.extend(result)
        previews.extend(preview)
        del latent, result, preview
    return ops.LayerdiffPipelineOutput(images=images, vis_list=previews)


def infer(pipe, *, fullpage, prompt, num_inference_steps=20, generator=None,
          guidance_scale=1.0, batch_size=1, strength=1.0, negative_prompt="",
          group_index=None, **kwargs):
    import torch
    from modules.layerdiffuse import diffusers_kdiffusion_sdxl as ops

    if (kwargs or fullpage is None or not prompt or batch_size != 1
            or guidance_scale != 1.0 or strength != 1.0 or negative_prompt not in (None, "")):
        raise ValueError("low_vram supports the pinned CLI's fullpage/batch=1/guidance=1 path only")
    pipe._guidance_scale = guidance_scale
    pipe._low_vram_placement.group_index = group_index
    hardware = {}
    if torch.cuda.is_initialized():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware = {"cuda_device": props.name, "cuda_total_bytes": props.total_memory}
    emit("layerdiff.inference", "start", group_index=group_index, frames=len(prompt),
         resolution=list(fullpage.shape[:2]), steps=num_inference_steps,
         torch_version=torch.__version__, **hardware)
    with torch.inference_mode():
        condition = _encode(pipe, fullpage, torch, ops)
        latents = _denoise(pipe, condition, prompt, num_inference_steps, generator,
                           group_index, torch, ops)
        del condition
        result = _decode(pipe, latents, fullpage, torch, ops)
        del latents
    # On failure the stage process exits. Do not mask OOM with cleanup CUDA calls.
    pipe._low_vram_placement.park()
    return result
