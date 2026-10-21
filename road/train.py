"""Shared orchestration; family entry points select native generator backends."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from . import checkpoints
from .artifacts import asset_fingerprint, digest, write_json
from .config import load
from .critic.dinov3 import DinoFeatures
from .critic.mixing import RandomMixing, refresh_if_due
from .critic.regularization import lazy_scale, penalty
from .critic.transformer import TransformerCritic, update_spectral_norm
from .data import Images, training_loader
from .distributed import local_call, rank_zero_call
from .models.factory import backend as make_backend
from .text_cache import TextCache
from .vjp import staged_backward


def autocast(device, precision):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
    )


def _run(family):
    parser = argparse.ArgumentParser(description=f"Train ROAD {family}")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--validate-config", action="store_true")
    a = parser.parse_args()
    cfg = load(a.config)
    t = cfg.training
    if cfg.family != family:
        raise ValueError("Configuration model family does not match entry point")
    if a.validate_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full ROAD training requires CUDA; use a CUDA training environment"
        )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != t.world_size:
        raise ValueError(
            "Launched world size differs from configuration; update accumulation/global batch explicitly"
        )
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    torch.manual_seed(t.seed)
    random.seed(t.seed)
    np.random.seed(t.seed % 2**32)
    out = Path(t.output)

    def prepare_output():
        if not a.resume and out.exists() and any(out.iterdir()):
            raise FileExistsError("Output is not empty; choose a fresh run directory")
        out.mkdir(parents=True, exist_ok=True)

    rank_zero_call(prepare_output)
    images = local_call(lambda: Images(cfg.assets.manifest, family))
    base_fp = rank_zero_call(lambda: asset_fingerprint(cfg.assets.base_model))
    identity = {
        "family": family,
        "base": base_fp,
        "dinov3": rank_zero_call(lambda: asset_fingerprint(cfg.assets.dinov3)),
        "data": images.fingerprint,
        "data_assets": rank_zero_call(images.verify_assets),
        "cache": local_call(lambda: digest(Path(cfg.assets.text_cache) / "metadata.json")),
        "config": cfg.to_dict(),
        "release_schema": 2,
    }

    def code_identity():
        source = hashlib.sha256(
            "".join(digest(f) for f in sorted(Path(__file__).parent.rglob("*.py"))).encode()
        ).hexdigest()
        versions = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "diffusers", "numpy", "Pillow")
        }
        return source, versions

    identity["source"], identity["versions"] = local_call(code_identity)
    # Operational changes (including a longer budget and worker count) do not
    # alter the resumed numerical state. Content hashes replace machine paths.
    identity["config"].pop("assets")
    for key in (
        "output", "steps", "checkpoint_interval", "export_interval",
        "log_interval", "data_workers", "prefetch_factor", "pin_memory",
    ):
        identity["config"]["training"].pop(key)
    authoritative_identity, launch_config = rank_zero_call(
        lambda: (identity, cfg.to_dict())
    )

    def check_identity():
        if identity != authoritative_identity or cfg.to_dict() != launch_config:
            raise ValueError("Ranks disagree on configuration, code or asset identity")

    local_call(check_identity)
    cache = local_call(lambda: TextCache(
        cfg.assets.text_cache, cfg.assets.manifest, family, base_fp
    ))
    rank_zero_call(cache.verify_assets)
    cache.verified = True  # shared, immutable asset tree verified once by rank zero

    def initialize_models():
        backend = make_backend(
            family,
            cfg.assets.base_model,
            torch.bfloat16 if t.vae_dtype == "bf16" else torch.float32,
            t.generator_checkpointing,
            t.vae_checkpointing,
        )
        backend.generator.to(device).train()
        backend.vae.to(device).eval()
        features = DinoFeatures(cfg.assets.dinov3, t.feature_checkpointing).to(device)
        mixing = RandomMixing().to(device)
        critic = TransformerCritic(
            family, theta=features.theta, checkpoint_blocks=t.critic_checkpointing
        ).to(device)
        return backend, features, mixing, critic

    backend, features, mixing, critic = local_call(initialize_models)
    if world > 1:
        backend.generator = DDP(
            backend.generator, device_ids=[local], gradient_as_bucket_view=True
        )
        critic = DDP(critic, device_ids=[local], broadcast_buffers=False)
    opt_kwargs = dict(
        lr=t.generator_lr, betas=(0.0, 0.99), weight_decay=0.0, foreach=False
    )
    if t.zero1 and world > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        opt_g = ZeroRedundancyOptimizer(
            backend.generator.parameters(),
            optimizer_class=torch.optim.AdamW,
            **opt_kwargs,
        )
    else:
        opt_g = torch.optim.AdamW(backend.generator.parameters(), **opt_kwargs)
    opt_d = torch.optim.AdamW(
        critic.parameters(),
        lr=t.critic_lr,
        betas=(0.0, 0.99),
        weight_decay=0.0,
        foreach=False,
    )
    # Rank-independent initialization; rank-dependent draws thereafter.
    torch.manual_seed(t.seed + rank)
    random.seed(t.seed + rank)
    np.random.seed((t.seed + rank) % 2**32)
    step = (
        checkpoints.load(
            a.resume, backend.generator, critic, mixing, opt_g, opt_d, identity
        )
        if a.resume
        else 0
    )
    if step >= t.steps:
        raise ValueError("Resume step is at/after the requested budget; increase training.steps")

    def record_run():
        previous = out / "run_identity.json"
        if previous.exists() and json.loads(previous.read_text()) != identity:
            raise ValueError("Output directory belongs to a different run; use a fresh path")
        recorded_steps = []
        for child in out.iterdir():
            prefix, _, number = child.name.partition("-")
            if prefix in ("state", "generator") and number.isdigit():
                recorded_steps.append(int(number))
        log = out / "metrics.jsonl"
        if log.exists():
            recorded_steps.extend(
                json.loads(line)["completed_steps"]
                for line in log.read_text().splitlines() if line.strip()
            )
        if recorded_steps and max(recorded_steps) > step:
            raise ValueError("Output contains later steps; resume into a fresh directory")
        write_json(out / "resolved_config.json", cfg.to_dict(), replace=True)
        write_json(previous, identity, replace=True)

    rank_zero_call(record_run)
    dtype = torch.bfloat16 if t.precision == "bf16" else torch.float32
    raw_d = checkpoints.unwrap(critic)
    loader = local_call(lambda: training_loader(images, cache, t, step, rank))
    batches = local_call(lambda: iter(loader))
    while step < t.steps:
        # CPU prefetch may run ahead, but only completed optimizer steps are saved.
        # Reconstructing tasks after resume discards unconsumed prefetched samples.
        def fetch_step():
            values = [next(batches) for _ in range(t.accumulation)]
            for micro, batch in enumerate(values):
                if not (batch["step"] == step).all() or not (batch["micro"] == micro).all():
                    raise RuntimeError("DataLoader returned an out-of-order training batch")
            return values

        microbatches = local_call(fetch_step)
        update_spectral_norm(raw_d)
        opt_g.zero_grad(set_to_none=True)
        raw_d.requires_grad_(False)
        g_value = 0.0
        # Math attention supports the critic's higher-order path. Generator retains
        # its own standard attention backend outside the objective callback.
        for micro in range(t.accumulation):
            batch = microbatches[micro]
            cond = {
                key: value.to(
                    device=device, dtype=torch.bool if key == "mask" else dtype,
                    non_blocking=t.pin_memory,
                )
                for key, value in batch["condition"].items()
            }
            noise = torch.randn(
                (t.microbatch, *backend.latent_shape), device=device, dtype=dtype
            )

            def predict(z):
                with autocast(device, t.precision):
                    return backend.predict(z, cond)

            def objective(rgb):
                with sdpa_kernel(SDPBackend.MATH), autocast(device, t.precision):
                    logits = raw_d(
                        mixing(features(rgb.float())), cond["tokens"], cond["mask"]
                    )
                    return F.softplus(-logits.float()).mean()

            sync = (
                backend.generator.no_sync()
                if world > 1 and micro < t.accumulation - 1
                else contextlib.nullcontext()
            )
            with sync:
                if t.generator_vjp == "three_stage":
                    value = staged_backward(
                        predict, backend.decode, objective, noise, 1 / t.accumulation
                    )
                else:
                    value = objective(backend.decode(predict(noise))) / t.accumulation
                    value.backward()
                    value = value.detach()
            g_value += value.detach()
        local_call(lambda: nn.utils.clip_grad_norm_(
            backend.generator.parameters(), t.generator_clip, error_if_nonfinite=True
        ))
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)
        raw_d.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        d_value = 0.0
        reg_scale = lazy_scale(
            cfg.regularization.gamma, cfg.regularization.lazy_interval, step
        )
        for micro in range(t.accumulation):
            batch = microbatches[micro]
            cond = {
                key: value.to(
                    device=device, dtype=torch.bool if key == "mask" else dtype,
                    non_blocking=t.pin_memory,
                )
                for key, value in batch["condition"].items()
            }
            real = batch["image"].to(device, non_blocking=t.pin_memory)
            real.requires_grad_(bool(reg_scale))
            with torch.no_grad(), autocast(device, t.precision):
                noise = torch.randn(
                    (t.microbatch, *backend.latent_shape), device=device, dtype=dtype
                )
                fake = backend.decode(backend.predict(noise, cond))
            sync = (
                critic.no_sync()
                if world > 1 and micro < t.accumulation - 1
                else contextlib.nullcontext()
            )
            with sync, sdpa_kernel(SDPBackend.MATH), autocast(device, t.precision):
                rf = mixing(features(real))
                ff = mixing(features(fake.float()))
                logits = critic(
                    [torch.cat([r, f]) for r, f in zip(rf, ff)],
                    torch.cat([cond["tokens"]] * 2),
                    torch.cat([cond["mask"]] * 2),
                )
                r, f = logits.chunk(2)
                loss = F.softplus(-r.float()).mean() + F.softplus(f.float()).mean()
                if reg_scale:
                    loss = loss + reg_scale * penalty(r, real, cfg.regularization.type)
                loss = loss / t.accumulation
                loss.backward()
                d_value += loss.detach()
            del rf, ff, logits, r, f, loss, real, fake
        local_call(lambda: nn.utils.clip_grad_norm_(
            critic.parameters(), t.critic_clip, error_if_nonfinite=True
        ))
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)
        step += 1
        refresh_if_due(mixing, step, cfg.mixing.refresh_interval)
        if step % t.log_interval == 0:
            losses = torch.stack([g_value, d_value])
            if world > 1:
                torch.distributed.all_reduce(losses)
                losses /= world
            g_mean, d_mean = losses.tolist()

            def log_metrics():
                with (out / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({
                        "completed_steps": step,
                        "g_loss": g_mean,
                        "d_loss": d_mean,
                        "reduction": "global_mean",
                    }) + "\n")
                print(f"step={step} g={g_mean:.5g} d={d_mean:.5g}", flush=True)

            rank_zero_call(log_metrics)
        if step % t.checkpoint_interval == 0 or step == t.steps:
            checkpoints.save(
                out / f"state-{step:07d}",
                backend.generator,
                critic,
                mixing,
                opt_g,
                opt_d,
                step,
                identity,
            )
        if step % t.export_interval == 0 or step == t.steps:
            rank_zero_call(lambda: checkpoints.export(
                out / f"generator-{step:07d}", backend, family, step, identity
            ))


def main(family):
    try:
        _run(family)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
