"""CPU equivalence against pinned upstream code, not an RTX speed/VRAM claim."""
import importlib
import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "service/seethrough"))
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    return importlib.import_module("unet_memory")


@pytest.fixture
def upstream(monkeypatch):
    root = os.environ.get("SEE_THROUGH_TEST_SOURCE")
    if not root:
        pytest.skip("set SEE_THROUGH_TEST_SOURCE to pinned upstream for equivalence tests")
    path = Path(root) / "common/modules/layerdiffuse/transformer3d.py"
    assert path.is_file(), "upstream test source missing"
    spec = importlib.util.spec_from_file_location("modules.layerdiffuse.transformer3d", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("frames", [11, 13])
def test_spatial_partition_preserves_frame_conditioning(memory, frames):
    import torch
    from diffusers.models.attention import BasicTransformerBlock

    torch.manual_seed(13)
    block = BasicTransformerBlock(8, 2, 4, cross_attention_dim=12).eval()
    reference = deepcopy(block)
    memory.partition(block, torch, size=1)
    hidden = torch.randn(frames, 9, 8)
    condition = torch.randn(frames, 7, 12)
    mask = torch.zeros(frames, 1, 7)
    mask[:, :, -1] = -10000
    with torch.inference_mode():
        actual = block(hidden, encoder_hidden_states=condition, encoder_attention_mask=mask)
        expected = reference(hidden, encoder_hidden_states=condition, encoder_attention_mask=mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("frames,batches", [(11, 1), (13, 2)])
def test_pinned_temporal_partition_retains_all_frames(memory, upstream, frames, batches):
    import torch

    torch.manual_seed(7)
    block = upstream.CrossFrameTransformerBlock(8, 8, 2, 4).eval()
    reference = deepcopy(block)
    memory.partition(block, torch, temporal=True, size=5)
    hidden = torch.randn(batches * frames, 13, 8)  # remainder, multiple sample groups
    with torch.inference_mode():
        actual = block(hidden, num_frames=frames)
        expected = reference(hidden, num_frames=frames)
        # Counterexample: independent per-frame calls lose temporal dependence.
        incorrect = torch.cat([reference(frame[None], num_frames=1) for frame in hidden])
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(incorrect, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("frames,dtype", [(11, "float32"), (13, "bfloat16")])
def test_pinned_interleaved_transformer_parity(memory, upstream, frames, dtype):
    import torch

    torch.manual_seed(42)
    dtype = getattr(torch, dtype)
    model = upstream.Transformer3DModel(
        num_attention_heads=2, attention_head_dim=4, in_channels=8,
        num_layers=2, norm_num_groups=4, cross_attention_dim=12,
    ).eval().to(dtype=dtype)
    reference = deepcopy(model)
    offload = memory.install(model, torch)
    # Exercise the same placement/hooks on CPU; this does not validate CUDA movement.
    offload.device = "cpu"
    offload.place("cuda", dtype)
    hidden = torch.randn(frames, 8, 12, 12, dtype=dtype)
    condition = torch.randn(frames, 7, 12, dtype=dtype)
    with torch.inference_mode():
        expected = reference(hidden, num_frames=frames, encoder_hidden_states=condition).sample
        actual = model(hidden, num_frames=frames, encoder_hidden_states=condition).sample
    torch.testing.assert_close(actual, expected, atol=2e-2 if dtype == torch.bfloat16 else 1e-5,
                               rtol=2e-2 if dtype == torch.bfloat16 else 1e-5)
    assert offload.active is None
    offload.validate("cuda")


def test_offload_never_moves_two_blocks_on_device_together(memory, monkeypatch):
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    reference = deepcopy(model)
    events = []
    trace = {"group_index": 0, "enabled": False}
    controller = memory.TransformerOffload(model, list(model.named_children()), torch, trace, device="cpu")
    monkeypatch.setattr(memory, "emit", lambda *a, **k: None)
    for name, child in model.named_children():
        move = child.to
        def record(*a, _name=name, _move=move, **kw):
            events.append((_name, controller.active, kw["device"]))
            return _move(*a, **kw)
        monkeypatch.setattr(child, "to", record)
    controller.place("cuda", torch.float32)
    data = torch.randn(3, 4)
    with torch.inference_mode():
        torch.testing.assert_close(model(data), reference(data))
    assert events == [("0", None, "cpu"), ("0", "0", "cpu"),
                      ("1", None, "cpu"), ("1", "1", "cpu")]
    controller.place("cpu", torch.float32)
    assert not controller.enabled and controller.active is None


def test_trace_logs_before_unet_failure_and_removes_hooks(memory, monkeypatch):
    import torch

    class Broken(torch.nn.Module):
        def forward(self, data):
            raise RuntimeError("first op")

    unet = Broken()
    rows = []
    monkeypatch.setattr(memory, "emit", lambda stage, event, **kw: rows.append((stage, event, kw)))
    with pytest.raises(RuntimeError, match="first op"), memory.first_call_trace(unet, 0):
        unet(torch.zeros(1))
    assert rows[0] == ("layerdiff.unet_module", "enter", {"module": "", "group_index": 0})
    assert not unet._forward_hooks and not unet._forward_pre_hooks
    assert len(rows) == 1  # no false submission/completion after failure


def test_unsupported_temporal_conditioning_rejected(memory):
    import torch

    class Temporal(torch.nn.Module):
        attn2 = object()

        def forward(self, hidden_states, num_frames, encoder_hidden_states=None):
            pytest.fail("unsupported operation must not run")

    block = Temporal().eval()
    memory.partition(block, torch, temporal=True, size=2)
    with torch.inference_mode(), pytest.raises(ValueError, match="attn2=None"):
        block(torch.zeros(13, 5, 8), num_frames=13)


@pytest.mark.parametrize("group,frames", [(0, 13), (1, 11)])
def test_pinned_full_unet_group_conditioning_and_skips(memory, upstream, monkeypatch, tmp_path, group, frames):
    import ast
    import sys

    import torch

    root = Path(os.environ["SEE_THROUGH_TEST_SOURCE"]) / "common"
    # Load the actual pure helper without importing unrelated training dependencies.
    path = root / "utils/torch_utils.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "zero_module")
    helper_path = tmp_path / "torch_utils.py"
    helper_path.write_text(ast.unparse(function))
    helper_spec = importlib.util.spec_from_file_location("utils.torch_utils", helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    monkeypatch.setitem(sys.modules, "utils.torch_utils", helper)
    spec = importlib.util.spec_from_file_location(
        "modules.layerdiffuse.layerdiff3d", root / "modules/layerdiffuse/layerdiff3d.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    torch.manual_seed(44)
    model = module.UNetFrameConditionModel(
        in_channels=8, out_channels=4, block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossFrameAttnDownBlock"),
        up_block_types=("CrossFrameAttnUpBlock", "UpBlock2D"), layers_per_block=1,
        mid_block_type="UNetMidBlockCrossFrameAttn",
        norm_num_groups=4, attention_head_dim=2, cross_attention_dim=2048,
        group_embedding=True, addition_embed_type="text_time", addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=1292,
    ).eval()
    reference = deepcopy(model)
    offload = memory.install(model, torch)
    offload.device = "cpu"
    offload.place("cuda", torch.float32)
    hidden = torch.randn(1, frames, 8, 8, 8)
    condition = torch.randn(frames, 3, 2048)
    added = {"text_embeds": torch.randn(frames, 1280), "time_ids": torch.zeros(frames, 6)}
    with torch.inference_mode():
        expected = reference(hidden, torch.tensor(12), encoder_hidden_states=condition,
                             added_cond_kwargs=added, group_index=group).sample
        actual = model(hidden, torch.tensor(12), encoder_hidden_states=condition,
                       added_cond_kwargs=added, group_index=group).sample
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    offload.validate("cuda")
