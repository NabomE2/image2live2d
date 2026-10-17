"""Inference-only memory scheduling for pinned See-through's frame UNet.

Never split the UNet's frame sequence. Spatial transformer batches are independent;
temporal attention keeps every frame and only partitions independent pixel positions.
"""
from __future__ import annotations

import inspect
import types
from contextlib import contextmanager

from low_vram_metrics import inventory, span
from telemetry import emit


def _slice_batch(value, start, stop, size, torch):
    if value is None or not torch.is_tensor(value) or value.ndim == 0:
        return value
    if value.shape[0] == size:
        return value[start:stop]
    if value.shape[0] == 1:
        return value
    raise ValueError(f"Unsupported spatial conditioning batch {value.shape[0]} (expected {size})")


def partition(module, torch, *, temporal=False, size=1, trace=None, name=""):
    original = module.forward
    signature = inspect.signature(original)

    def forward(self, hidden_states, *args, **kwargs):
        if torch.is_grad_enabled() or self.training:
            return original(hidden_states, *args, **kwargs)
        bound = signature.bind(hidden_states, *args, **kwargs)
        bound.apply_defaults()
        if temporal:
            if self.attn2 is not None:
                raise ValueError("Temporal partitioning requires pinned attn2=None")
            frames = bound.arguments["num_frames"]
            if frames <= 0 or hidden_states.shape[0] % frames:
                raise ValueError("Invalid temporal frame grouping")
            axis = 1  # all frames retained, partition only spatial positions
        else:
            if self.norm_type != "layer_norm" or bound.arguments.get("cross_attention_kwargs"):
                raise ValueError("Spatial partitioning supports pinned layer_norm/no custom attention kwargs")
            axis = 0
        length = hidden_states.shape[axis]
        if length <= size:
            return original(hidden_states, *args, **kwargs)
        output = None
        for start in range(0, length, size):
            stop = min(start + size, length)
            arguments = dict(bound.arguments)
            arguments["hidden_states"] = hidden_states.narrow(axis, start, stop - start)
            if not temporal:
                for key in ("attention_mask", "encoder_hidden_states", "encoder_attention_mask",
                            "timestep", "class_labels"):
                    if key in arguments:
                        arguments[key] = _slice_batch(arguments[key], start, stop, length, torch)
                if arguments.get("added_cond_kwargs"):
                    raise ValueError("Additional spatial conditioning is not supported by pinned adapter")
            detail = {"module": name, "axis": axis, "start": start, "stop": stop,
                      "total": length, "group_index": (trace or {}).get("group_index")}
            if trace and trace["enabled"]:
                emit("layerdiff.partition", "enter", **detail)
            chunk = original(**arguments)
            if output is None:
                shape = list(chunk.shape)
                shape[axis] = length
                output = chunk.new_empty(shape)
            output.narrow(axis, start, stop - start).copy_(chunk)
            del chunk
            if trace and trace["enabled"]:
                emit("layerdiff.partition", "submitted", **detail)
        return output

    module.forward = types.MethodType(forward, module)


class TransformerOffload:
    """Keep small/conv modules resident; load one transformer block at a time.

    Calls retain full inputs and original execution order. Weight transfers are
    synchronous and byte-preserving; no quantization or activation offload.
    """

    def __init__(self, unet, blocks, torch, trace, device="cuda"):
        self.unet, self.blocks, self.torch, self.trace = unet, dict(blocks), torch, trace
        self.device = device
        self.original_to = unet.to
        self.enabled = False
        self.active = None
        blocked = {id(module) for module in self.blocks.values()}
        self.resident = []

        def collect(module):
            if id(module) in blocked:
                return
            self.resident.append(module)
            for child in module.children():
                collect(child)

        collect(unet)
        for name, module in self.blocks.items():
            module.register_forward_pre_hook(self._load(name))
            module.register_forward_hook(self._unload(name))

    def _load(self, name):
        def hook(module, args):
            if not self.enabled:
                return
            if self.torch.is_grad_enabled() or self.active is not None:
                raise RuntimeError("Transformer offload requires sequential inference_mode execution")
            with span(self.torch, "layerdiff.block_load", module=name,
                      group_index=self.trace.get("group_index")):
                module.to(device=self.device)
            self.active = name
        return hook

    def _unload(self, name):
        def hook(module, args, output):
            if not self.enabled:
                return
            with span(self.torch, "layerdiff.block_evict", module=name,
                      group_index=self.trace.get("group_index")):
                module.to(device="cpu")
            self.active = None
        return hook

    def place(self, device, dtype):
        self.enabled = False
        self.original_to(device="cpu", dtype=dtype)
        self.active = None
        if device == "cuda":
            # PyTorch 2.8 Module._apply(recurse=False) moves only each resident
            # node's own parameters/buffers, never its offloaded descendants.
            def move(tensor):
                return tensor.to(device=self.device,
                                 dtype=dtype if tensor.is_floating_point() else tensor.dtype)
            for module in self.resident:
                module._apply(move, recurse=False)
            self.enabled = True
        self.validate(device)
        emit("layerdiff.unet_residency", "ready", policy="transformer_block_offload",
             resident_device=self.device if self.enabled else "cpu", blocks=len(self.blocks),
             residency=inventory(self.unet))
        return self.unet

    def validate(self, device):
        resident_device = self.device if device == "cuda" else "cpu"
        for name, block in self.blocks.items():
            if any(d.split(":")[0] != "cpu" for d in inventory(block)["device_bytes"]):
                raise RuntimeError(f"Transformer block was not evicted: {name}")
        for module in self.resident:
            for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
                if tensor.device.type != resident_device.split(":")[0]:
                    raise RuntimeError("Resident UNet tensor has incorrect device")


def install(unet, torch):
    from diffusers.models.attention import BasicTransformerBlock

    trace = {"enabled": False, "group_index": None}
    blocks = []
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock):
            if module.norm_type != "layer_norm":
                raise ValueError("Unsupported spatial transformer normalization")
            partition(module, torch, size=1, trace=trace, name=name)
            blocks.append((name, module))
        elif (type(module).__name__ == "CrossFrameTransformerBlock"
              and type(module).__module__ == "modules.layerdiffuse.transformer3d"):
            if module.attn2 is not None:
                raise ValueError("Unsupported temporal cross conditioning")
            partition(module, torch, temporal=True, size=128, trace=trace, name=name)
            blocks.append((name, module))
    unet._low_vram_trace = trace
    # Small test/different models with no transformer blocks need no weight scheduler.
    offload = TransformerOffload(unet, blocks, torch, trace) if blocks else None
    unet._low_vram_offload = offload
    offloaded_bytes = sum(inventory(module)["weight_bytes"] for _, module in blocks)
    emit("layerdiff.unet_memory", "configured", spatial_batch=1, temporal_positions=128,
         transformer_offload_blocks=len(blocks), all_temporal_frames_retained=True,
         offloaded_weight_bytes=offloaded_bytes,
         resident_weight_bytes=inventory(unet)["weight_bytes"] - offloaded_bytes)
    return offload


@contextmanager
def first_call_trace(unet, group_index):
    """Unsynchronized first-use markers: submitted is not CUDA completion."""
    state = getattr(unet, "_low_vram_trace", {"enabled": False})
    state.update(enabled=True, group_index=group_index)
    handles, entered = [], set()

    def before(name):
        def hook(module, args):
            if name not in entered:
                entered.add(name)
                emit("layerdiff.unet_module", "enter", module=name, group_index=group_index)
        return hook

    def after(name):
        def hook(module, args, output):
            if name in entered:
                entered.remove(name)
                # Remove this pair after first use; each attention/FF can be called
                # repeatedly by partitioning without flooding internal leaf logs.
                for handle in pairs[name]:
                    handle.remove()
                emit("layerdiff.unet_module", "submitted", module=name, group_index=group_index)
        return hook

    pairs = {}
    try:
        for name, module in unet.named_modules():
            kind = type(module).__name__
            if (not name or kind in {"BasicTransformerBlock", "CrossFrameTransformerBlock",
                                    "ResnetBlock2D", "Attention", "FeedForward",
                                    "GroupEmbedding", "TimestepEmbedding", "Conv2d", "GroupNorm"}
                    or name in {"conv_in", "conv_out", "mid_block"}):
                pairs[name] = [module.register_forward_pre_hook(before(name)),
                               module.register_forward_hook(after(name))]
                handles.extend(pairs[name])
        yield
    finally:
        state["enabled"] = False
        for handle in handles:
            handle.remove()
