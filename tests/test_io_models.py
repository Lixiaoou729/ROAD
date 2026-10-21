import json

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from road.data import Images, digest
from road.evaluation.protocol import crop_resize, select_checkpoint, validate_prompts
from road.models.sd35_generator_backend import make_flow_state, predict_clean_from_flow
from road.models.sd35_text_condition import TextCondition
from road.text_cache import TextCache


def test_native_flow_equation_and_gradient():
    x = torch.randn(2, 16, 2, 2, requires_grad=True)
    v = torch.randn_like(x, requires_grad=True)
    y = predict_clean_from_flow(x, v, 1.0)
    torch.testing.assert_close(y, x - v)
    y.sum().backward()
    torch.testing.assert_close(v.grad, -torch.ones_like(v))
    torch.testing.assert_close(make_flow_state(x.detach(), v.detach(), 1), v.detach())


def test_text_condition_validation():
    c = TextCondition(torch.randn(1, 154, 4096), torch.randn(1, 2048))
    assert c.validate() is c
    with pytest.raises(ValueError):
        TextCondition(torch.randn(1, 77, 2048), torch.randn(1, 2048)).validate()


def test_cache_integrity(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"id": "1", "caption": "one", "image": "x.png"}) + "\n"
    )
    tensors = {
        "tokens": torch.randn(1, 3, 4),
        "pooled": torch.randn(1, 2),
        "mask": torch.ones(1, 3, dtype=torch.bool),
    }
    save_file(tensors, tmp_path / "000000.safetensors")
    meta = {
        "schema": 1,
        "manifest_sha256": digest(manifest),
        "family": "sdxl",
        "base_fingerprint": "base",
        "count": 1,
        "shard_size": 64,
        "shards": {"000000.safetensors": digest(tmp_path / "000000.safetensors")},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    cache = TextCache(tmp_path, manifest, "sdxl", "base")
    cache.verify_assets()
    actual = cache.batch([0], "cpu", torch.float32)
    for k in tensors:
        torch.testing.assert_close(actual[k], tensors[k])
    with pytest.raises(ValueError):
        TextCache(tmp_path, manifest, "sd35m", "base")
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError):
        TextCache(tmp_path, manifest, "sdxl", "base")


def test_data_rgb_and_preprocessing(tmp_path):
    im = Image.fromarray(
        np.random.default_rng(10).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    )
    im.save(tmp_path / "im.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"id": "1", "caption": "a", "image": "im.png"}) + "\n"
    )
    x = Images(manifest, "sdxl").read(0, 9)
    expected = torch.tensor(np.asarray(im).copy()).permute(2, 0, 1) / 127.5 - 1
    torch.testing.assert_close(x, expected)
    m = Images(manifest, "sd35m")
    torch.testing.assert_close(m.read(0, 9), m.read(0, 9))
    assert m.read(0, 9).shape == (3, 1024, 1024)
    assert crop_resize(Image.new("RGB", (100, 80)), 512).size == (512, 512)


def test_protocol_selection_and_counts():
    with pytest.raises(ValueError):
        validate_prompts([{"id": "1", "caption": "a"}], "coco10k")
    rows = [
        {"fid": v, "completed_steps": s, "protocol_hash": "same"}
        for v, s in [(11.1, 500), (10.9, 1000)]
    ]
    assert select_checkpoint(rows)["completed_steps"] == 1000
    rows[1]["protocol_hash"] = "other"
    with pytest.raises(ValueError):
        select_checkpoint(rows)


def test_sdxl_real_diffusers_backend(tmp_path):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    from road.models.sdxl import Backend

    g = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=40,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        latent_channels=4,
        norm_num_groups=8,
    )
    g.save_pretrained(tmp_path / "unet")
    vae.save_pretrained(tmp_path / "vae")
    DDPMScheduler().save_pretrained(tmp_path / "scheduler")
    b = Backend(str(tmp_path), torch.float32, False, False)
    noise = torch.randn(1, 4, 8, 8)
    condition = {"tokens": torch.randn(1, 5, 32), "pooled": torch.randn(1, 16)}
    latent = b.predict(noise, condition)
    rgb = b.decode(latent)
    assert latent.shape == noise.shape
    assert rgb.shape == (1, 3, 8, 8)
    rgb.square().mean().backward()
    assert any(p.grad is not None for p in b.generator.parameters())
    assert all(p.grad is None for p in b.vae.parameters())


def test_standalone_inference_contract(tmp_path, monkeypatch):
    from road import infer

    class Enc:
        def encode(self, prompts):
            return {
                "tokens": torch.zeros(len(prompts), 2, 3),
                "pooled": torch.zeros(len(prompts), 2),
                "mask": torch.ones(len(prompts), 2, dtype=torch.bool),
            }

    class Backend:
        generator = torch.nn.Linear(1, 1)
        vae = torch.nn.Identity()
        latent_shape = (3, 4, 4)

        def predict(self, x, c):
            return x

        def decode(self, x):
            return x

    monkeypatch.setattr(infer, "asset_fingerprint", lambda p: "asset")
    monkeypatch.setattr(infer, "encoder", lambda *a: Enc())
    monkeypatch.setattr(infer, "make_backend", lambda *a, **kw: Backend())
    (tmp_path / "generator").mkdir()
    (tmp_path / "generator/config.json").write_text("{}")
    (tmp_path / "road.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "checkpoint_type": "generator_only",
                "files": {
                    "generator/config.json": digest(tmp_path / "generator/config.json")
                },
                "family": "sdxl",
                "identity": {
                    "base": "asset",
                    "config": {"training": {"vae_dtype": "fp32"}},
                },
            }
        )
    )
    pipe = infer.Pipeline(tmp_path, "unused", "cpu", "fp32")
    a = pipe(["a", "b"], [1, 2])
    b = pipe(["b"], [2])
    assert np.array_equal(a[1], b[0])
    assert a[0].mode == "RGB"


def test_dataset_seal_and_white_alpha(tmp_path):
    Image.new("RGBA", (512, 512), (0, 0, 0, 0)).save(tmp_path / "im.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({"id": "1", "caption": "a", "image": "im.png"}) + "\n"
    )
    meta = {
        "count": 1,
        "manifest_sha256": digest(manifest),
        "shards": {"im.png": digest(tmp_path / "im.png")},
    }
    manifest.with_suffix(".assets.json").write_text(json.dumps(meta))
    images = Images(manifest, "sdxl")
    assert images.verify_assets()
    assert images.read(0, 0).min() == 1
    Image.new("RGB", (512, 512)).save(tmp_path / "im.png")
    with pytest.raises(ValueError, match="Dataset bytes changed"):
        images.verify_assets()


def test_prepared_tar_random_access(tmp_path, monkeypatch):
    import io
    import sys
    import tarfile

    from road.prepare_data import main

    image = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(image, format="PNG")
    with tarfile.open(tmp_path / "source.tar", "w") as tar:
        for name, data in [("one.png", image.getvalue()), ("one.txt", b"a red image")]:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--shards",
            str(tmp_path / "source.tar"),
            "--output",
            str(tmp_path / "train.jsonl"),
        ],
    )
    main()
    images = Images(tmp_path / "train.jsonl", "sdxl")
    images.verify_assets()
    assert "offset" in images.rows[0]
    torch.testing.assert_close(
        images.read(0, 1)[:, 0, 0], torch.tensor([1.0, -1.0, -1.0])
    )


def test_dinov3_real_model_taps_and_image_jacobian(monkeypatch):
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel

    from road.critic.dinov3 import DinoFeatures

    # Native B/16 dimensions and 12 blocks; narrow FFNs keep CPU smoke inexpensive.
    config = DINOv3ViTConfig(
        hidden_size=768,
        intermediate_size=32,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=4,
        patch_size=16,
    )
    config._attn_implementation = "eager"
    model = DINOv3ViTModel(config)
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: model)
    features = DinoFeatures("unused", checkpoint_features=True)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    taps = features(x)
    assert len(taps) == 4 and all(t.shape == (1, 197, 768) for t in taps)
    scale = torch.nn.Parameter(torch.tensor(1.0))
    score = sum(t[:, 0, :4].sum() for t in taps) * scale
    grad = torch.autograd.grad(score, x, create_graph=True)[0]
    grad.square().mean().backward()
    assert torch.isfinite(scale.grad) and scale.grad.abs() > 0
    assert all(p.grad is None for p in model.parameters())


def test_sd35_real_diffusers_transformer():
    from diffusers import SD3Transformer2DModel

    model = SD3Transformer2DModel(
        sample_size=4,
        patch_size=2,
        in_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=4096,
        caption_projection_dim=16,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=4,
        dual_attention_layers=(0,),
        qk_norm="rms_norm",
    )
    model.enable_gradient_checkpointing()
    model.train()
    x = torch.randn(1, 16, 4, 4)
    prediction = model(
        hidden_states=x,
        timestep=torch.tensor([1000.0]),
        encoder_hidden_states=torch.randn(1, 154, 4096),
        pooled_projections=torch.randn(1, 2048),
        return_dict=False,
    )[0]
    clean = predict_clean_from_flow(x, prediction, 1.0)
    assert clean.shape == x.shape
    clean.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_import_generator_strict_keys_and_export_hash(tmp_path):
    from types import SimpleNamespace

    from diffusers import DDPMScheduler, UNet2DConditionModel

    from road.checkpoints import export
    from road.import_generator import load_generator_state

    model = UNet2DConditionModel(
        sample_size=4,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        cross_attention_dim=32,
    )
    torch.save(model.state_dict(), tmp_path / "legacy.pt")
    load_generator_state(model, tmp_path / "legacy.pt")
    export(
        tmp_path / "export",
        SimpleNamespace(generator=model, scheduler=DDPMScheduler()),
        "sdxl",
        500,
        {"base": "test", "config": {"training": {"vae_dtype": "fp32"}}},
    )
    meta = json.loads((tmp_path / "export/road.json").read_text())
    assert meta["files"] and all(
        digest(tmp_path / "export" / k) == v for k, v in meta["files"].items()
    )
    torch.save({"unexpected": torch.zeros(1)}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):
        load_generator_state(model, tmp_path / "bad.pt")


def test_native_sd35_sigma_one_schedule():
    from diffusers import FlowMatchEulerDiscreteScheduler

    from road.models.sd35_generator_backend import (
        build_conditioning_schedule,
        resolve_conditioning_sigma,
    )

    for dynamic in (False, True):
        scheduler = FlowMatchEulerDiscreteScheduler(
            shift=3.0, use_dynamic_shifting=dynamic
        )
        schedule = build_conditioning_schedule(
            scheduler,
            latent_height=128,
            latent_width=128,
            patch_size=2,
            image_height=1024,
            image_width=1024,
        )
        result = resolve_conditioning_sigma(schedule)
        assert result.resolved_sigma == 1.0 and result.resolved_timestep == 1000.0


def test_geneval_task_mean_and_coverage():
    from road.evaluation.geneval import summarize_results
    from road.evaluation.protocol import TAGS

    rows = [
        {"filename": str(i), "tag": tag, "correct": i % 2 == 0}
        for i, tag in enumerate(TAGS)
    ]
    result = summarize_results(rows, [str(i) for i in range(6)])
    assert result["overall"] == 0.5
    with pytest.raises(ValueError):
        summarize_results(rows[:-1], [str(i) for i in range(6)])


def test_image_reward_native_bert_compatibility():
    import importlib.util

    if importlib.util.find_spec("ImageReward") is None:
        pytest.skip("Optional evaluation extra is not installed")
    from road.evaluation.image_reward import import_image_reward

    import_image_reward()
    from ImageReward.models.BLIP.med import BertConfig, BertModel

    config = BertConfig(
        hidden_size=24,
        num_attention_heads=3,
        num_hidden_layers=1,
        intermediate_size=32,
        encoder_width=24,
        vocab_size=30,
        add_cross_attention=True,
    )
    model = BertModel(config, add_pooling_layer=False).eval()
    with torch.no_grad():
        out = model(
            torch.tensor([[1, 2, 3]]),
            encoder_hidden_states=torch.randn(1, 4, 24),
            return_dict=True,
        )
    assert out.last_hidden_state.shape == (1, 3, 24)
    assert torch.isfinite(out.last_hidden_state).all()
