"""Adapter contracts and CPU tensor parity; not an 8GB/GPU performance claim."""
import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVICE = Path(__file__).resolve().parents[1] / "service" / "seethrough"


@pytest.fixture
def low(monkeypatch):
    monkeypatch.syspath_prepend(str(SERVICE))
    return importlib.import_module("low_vram")


@pytest.mark.parametrize("profile,resolution,expected", [
    ("low_vram", None, "768"), ("low_vram", "1024", "1024"),
    ("staged", None, None), ("legacy", None, None),
])
def test_profile_commands(low, monkeypatch, tmp_path, stage_output, profile, resolution, expected):
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "MEMORY_PROFILE", profile)
    monkeypatch.setattr(app, "RESOLUTION", resolution)
    monkeypatch.delenv("MANAGED_MODELS", raising=False)
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        if profile == "legacy":
            output = tmp_path / "output"
            output.mkdir(exist_ok=True)
            (output / "input.psd").write_bytes(b"PSD")
        else:
            stage_output(Path(cmd[cmd.index("--save_dir") + 1]), kw["stage"])

    monkeypatch.setattr(app, "run_process", run)
    (tmp_path / "input.png").write_bytes(b"input")
    assert app._subprocess_infer(tmp_path / "input.png").name == "input.psd"
    assert len(calls) == (1 if profile == "legacy" else 3)
    assert len({kw['deadline'] for _, kw in calls}) == len(calls)
    for cmd, _ in calls:
        if expected:
            assert cmd[cmd.index("--resolution") + 1] == expected
        else:
            assert "--resolution" not in cmd
        if profile == "low_vram":
            assert "--low-vram" in cmd
            assert cmd[cmd.index("--inference_steps") + 1] == "20"
        else:
            assert "--low-vram" not in cmd and "--inference_steps" not in cmd
        # Depth quality, seeds and empty-head/missing-PNG patches are unchanged.
        assert "--inference_steps_depth" not in cmd and "--seed" not in cmd


def test_low_vram_disables_warm_even_if_requested(low, monkeypatch):
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "MEMORY_PROFILE", "low_vram")
    monkeypatch.setattr(app, "INFERENCE_MODE", "warm")
    monkeypatch.setattr(app, "WarmPipeline", lambda *a, **k: pytest.fail("warm must not load"))
    app._startup()
    assert app.health()["memory_profile"] == "low_vram"


def test_placement_evicts_before_loading_and_parks(low, monkeypatch):
    events, devices = [], {name: "cpu" for name in ("unet", "vae", "decoder")}

    class Model:
        def __init__(self, name):
            self.name = name

        def parameters(self):
            return [SimpleNamespace(device=devices[self.name], dtype="bf16",
                                    numel=lambda: 8, element_size=lambda: 2)]

        def buffers(self):
            return []

        def to(self, device, **kwargs):
            devices[self.name] = device
            events.append((self.name, device))
            assert not (devices["unet"] == "cuda" and
                        any(devices[n] == "cuda" for n in ("vae", "decoder")))

    pipe = SimpleNamespace(unet=Model("unet"), vae=Model("vae"),
                           trans_vae=SimpleNamespace(decoder=Model("decoder")))
    torch = SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(empty_cache=lambda: None,
                                                                  is_initialized=lambda: False))
    monkeypatch.setattr(low, "emit", lambda *a, **k: None)
    placement = low.Placement(pipe, torch)
    # Body then head: the same weights must be reused and each cycle must evict.
    for _ in range(2):
        placement.activate("vae")
        placement.activate("unet")
        placement.activate("vae", "decoder")
        placement.park()
        assert set(devices.values()) == {"cpu"}
    assert events.count(("unet", "cuda")) == 2


def test_runner_routes_only_layerdiff_to_low_vram(low, monkeypatch):
    runner = importlib.import_module("staged_inference")
    events = []

    def model(name):
        class Model:
            text_encoder = SimpleNamespace(to=lambda **k: events.append("encoder"))
            text_encoder_2 = text_encoder

            @classmethod
            def from_pretrained(cls, *a, **kw):
                assert kw["torch_dtype"] == "bf16"
                return cls()

            def cache_tag_embeds(self):
                events.append("cache")

            def __call__(self, **kw):
                return "original"
        Model.__name__ = name
        return Model

    names = ("TransparentVAE", "UNetFrameConditionModel",
             "KDiffusionStableDiffusionXLPipeline", "MarigoldDepthPipeline")
    iu = SimpleNamespace(**{name: model(name) for name in names})
    monkeypatch.setattr(low, "configure", lambda *a: events.append("configure"))
    monkeypatch.setattr(low, "infer", lambda *a, **kw: "low")
    monkeypatch.setattr(runner, "emit", lambda *a, **kw: None)
    runner.adapt(iu, SimpleNamespace(bfloat16="bf16", inference_mode=nullcontext), "low_vram")
    pipe = iu.KDiffusionStableDiffusionXLPipeline.from_pretrained("repo")
    assert events == ["encoder", "encoder", "cache", "configure"]
    assert pipe() == "low"
    assert iu.MarigoldDepthPipeline.from_pretrained("repo")() == "original"


@pytest.mark.parametrize("frames", [11, 13])
def test_chunked_feed_forward_tensor_parity_with_tail(low, frames):
    torch = pytest.importorskip("torch")
    from copy import deepcopy
    module = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.GELU(), torch.nn.Linear(32, 8))
    module.eval()
    reference = deepcopy(module)
    low.chunk_feed_forward(module, torch, tokens=17)
    # Noncontiguous input and a nondivisible token count cover both frame counts.
    data = torch.randn(frames, 7, 8).transpose(0, 1)
    with torch.inference_mode():
        torch.testing.assert_close(module(data), reference(data))


def test_configure_real_diffusers_modules(low, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKL
    from diffusers.models.attention import FeedForward
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    unet = torch.nn.Sequential(Attention(query_dim=8, heads=2, dim_head=4), FeedForward(8))
    vae = AutoencoderKL(block_out_channels=(32, 32),
                        down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2,
                        norm_num_groups=8, sample_size=32)
    trans = torch.nn.Module()
    trans.decoder = torch.nn.Linear(8, 8)
    pipe = SimpleNamespace(unet=unet, vae=vae, trans_vae=trans, vae_scale_factor=2)
    monkeypatch.setattr(low, "emit", lambda *a, **k: None)
    low.configure(pipe, torch)
    placement = pipe._low_vram_placement
    low.configure(pipe, torch)
    assert pipe._low_vram_placement is placement  # no nested patches
    assert isinstance(unet[0].processor, AttnProcessor2_0)
    assert vae.use_tiling and vae.use_slicing
    assert vae.tile_sample_min_size == 512 and vae.tile_latent_min_size == 256
    vae.to(device="cuda", dtype=torch.bfloat16)  # bootstrap MUST stay on CPU
    trans.to(device="cuda", dtype=torch.bfloat16)
    assert vae.device.type == "cpu" and next(trans.parameters()).device.type == "cpu"
    # Actual SDPA + patched Diffusers GEGLU, including a short final chunk.
    data = torch.randn(2, 519, 8)
    with torch.inference_mode():
        result = unet(data)
    assert result.shape == data.shape and torch.isfinite(result).all()


@pytest.mark.parametrize("frames,group", [(13, 0), (11, 1)])
def test_full_tensor_pipeline_preserves_frames_alpha_and_steps(low, monkeypatch, frames, group):
    torch = pytest.importorskip("torch")
    from types import ModuleType

    import numpy as np

    events = []
    logs = []
    metrics = importlib.import_module("low_vram_metrics")
    monkeypatch.setattr(metrics, "emit", lambda stage, event, **kw: logs.append(
        {"stage": stage, "event": event, **kw}))
    cpu = torch.device("cpu")
    dtype = torch.float32

    class VAE:
        device = cpu
        config = SimpleNamespace(scaling_factor=0.5)

        def encode(self, feed):
            # Opaque RGB conditioning despite original page alpha=128.
            torch.testing.assert_close(feed, torch.ones_like(feed))
            return SimpleNamespace(latent_dist=SimpleNamespace(
                sample=lambda: torch.ones(1, 4, 2, 2)))

    vae = VAE()
    vae.dtype = dtype

    class UNet:
        device = cpu

        def __call__(self, data, timestep, **kw):
            assert data.shape == (1, frames, 8, 2, 2)
            assert kw["group_index"] == group
            assert kw["encoder_hidden_states"].shape[0] == frames
            assert kw["added_cond_kwargs"]["time_ids"].shape == (frames, 6)
            events.append("step")
            return (data[:, :, 4:],)

    unet = UNet()
    unet.dtype = dtype

    def decoder(vae, latent, mask):
        torch.testing.assert_close(latent, torch.full_like(latent, -2.0))
        torch.testing.assert_close(mask, torch.full_like(mask, 128 / 255))
        events.append("decode")
        return [np.zeros((16, 16, 4), dtype=np.uint8)], [np.zeros((16, 16, 3), dtype=np.uint8)]

    def img2tensor(data, normalize=False, device="cpu", dtype=torch.float32):
        if data.ndim == 2:
            data = data[..., None]
        result = torch.from_numpy(data.copy()).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
        return result / 255 if normalize else result

    def timesteps(scheduler, count, device, **kw):
        assert count == 20
        return torch.arange(count), count

    ops = ModuleType("modules.layerdiffuse.diffusers_kdiffusion_sdxl")
    ops.img2tensor = img2tensor
    ops.retrieve_timesteps = timesteps
    ops.randn_tensor = lambda shape, **kw: torch.zeros(shape, device=kw['device'], dtype=kw['dtype'])
    ops.LayerdiffPipelineOutput = SimpleNamespace
    parent = ModuleType("modules.layerdiffuse")
    parent.diffusers_kdiffusion_sdxl = ops
    monkeypatch.setitem(sys.modules, "modules", ModuleType("modules"))
    monkeypatch.setitem(sys.modules, "modules.layerdiffuse", parent)
    pipe = SimpleNamespace(
        vae=vae, unet=unet, vae_scale_factor=8,
        trans_vae=SimpleNamespace(decoder=decoder),
        scheduler=SimpleNamespace(init_noise_sigma=1, order=1,
                                  scale_model_input=lambda x, t: x,
                                  step=lambda prediction, t, x, **kw: (x - 0.1 * prediction,)),
        encode_cropped_prompt_77tokens_cached=lambda prompt: (torch.ones(frames, 2, 4),
                                                             torch.ones(frames, 4)),
        progress_bar=lambda **kw: nullcontext(SimpleNamespace(update=lambda: None)),
        _low_vram_placement=SimpleNamespace(activate=lambda *a: events.append(a),
                                            park=lambda: events.append("park")),
    )
    page = np.full((16, 16, 4), 255, dtype=np.uint8)
    page[..., -1] = 128
    output = low.infer(pipe, fullpage=page, prompt=["tag"] * frames, group_index=group)
    assert len(output.images) == len(output.vis_list) == frames
    assert events == [("vae",), ("unet",)] + ["step"] * 20 + [
        ("vae", "decoder")] + ["decode"] * frames + ["park"]
    completed = [e for e in logs if e["stage"] == "layerdiff.step" and e["event"] == "end"]
    assert [e["step"] for e in completed] == list(range(1, 21))
    assert all(e["steps"] == 20 and e["group_index"] == group for e in completed)
    assert completed[-1]["progress"] == 1.0
    assert all(e["elapsed_seconds"] >= e["unet_seconds"] for e in completed)
    starts = [e for e in logs if e["event"] == "start"]
    first_step = next(i for i, e in enumerate(logs) if e["stage"] == "layerdiff.step")
    for stage in ("prompt_lookup", "time_ids", "schedule_setup", "noise_setup"):
        assert any(e["stage"] == f"layerdiff.{stage}" and e["event"] == "end"
                   for e in logs[:first_step])
    for step in range(1, 21):
        assert [e["stage"] for e in starts if e.get("step") == step] == [
            "layerdiff.step", "layerdiff.step_prepare", "layerdiff.unet", "layerdiff.scheduler"]
    transfers = [e for e in starts if e["stage"] == "layerdiff.tensor_transfer"]
    assert {e["role"] for e in transfers} == {"encode_input", "condition_out", "condition_in",
                                              "prompt_embeds", "pooled_embeds", "denoised_latents",
                                              "alpha_mask", "decode_latent"}
    assert sum(e["role"] == "decode_latent" for e in transfers) == frames
    assert all("step" not in e for e in transfers)  # no adapter transfer inside the step loop
    with pytest.raises(ValueError, match="guidance=1"):
        low.infer(pipe, fullpage=page, prompt=["tag"], guidance_scale=5)
