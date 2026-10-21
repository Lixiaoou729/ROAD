"""Versioned full-state continuation and independent generator exports."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .artifacts import checked_files, digest, object_digest, write_json
from .distributed import local_call, rank, rank_zero_call, world_size

FULL_SCHEMA = 2
EXPORT_SCHEMA = 2


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def _rng_state():
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached_gauss),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA training state cannot resume on CPU")
        torch.cuda.set_rng_state(state["cuda"])


def _create_staging(path):
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint or unfinished write exists: {path}")
    temporary.mkdir(parents=True)
    return temporary


def save(path, generator, critic, mixing, opt_g, opt_d, completed_steps, identity):
    """All ranks participate; output must be on their shared filesystem."""
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("Invalid completed step")
    path = Path(path)
    temporary = Path(rank_zero_call(lambda: str(_create_staging(path))))
    if hasattr(opt_g, "consolidate_state_dict"):
        opt_g.consolidate_state_dict(to=0)
    local_call(lambda: torch.save(_rng_state(), temporary / f"rng-{rank()}.pt"))

    def finish():
        torch.save(
            {
                "generator": unwrap(generator).state_dict(),
                "critic": unwrap(critic).state_dict(),
                "mixing": mixing.state_dict(),
                "optimizer_g": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
            },
            temporary / "state.pt",
        )
        files = {item.name: digest(item) for item in temporary.iterdir() if item.is_file()}
        write_json(temporary / "metadata.json", {
            "schema": FULL_SCHEMA,
            "checkpoint_type": "training_state",
            "completed_steps": completed_steps,
            "world_size": world_size(),
            "identity": identity,
            "files": files,
        })
        os.replace(temporary, path)

    rank_zero_call(finish)


def load(path, generator, critic, mixing, opt_g, opt_d, identity):
    path = Path(path)

    def preflight():
        if path.name.endswith(".partial"):
            raise ValueError("An unfinished checkpoint is not a resume target")
        meta = json.loads((path / "metadata.json").read_text())
        if meta.get("schema") != FULL_SCHEMA or meta.get("checkpoint_type") != "training_state":
            raise ValueError("Expected full-state schema 2. Schema 1/research training states require their original trainer; generator-only conversion is separate.")
        if meta.get("world_size") != world_size() or meta.get("identity") != identity:
            raise ValueError("Resume requires matching numerical configuration, code, assets and world size")
        if type(meta.get("completed_steps")) is not int or meta["completed_steps"] < 0:
            raise ValueError("Invalid checkpoint step")
        expected = {"state.pt"} | {f"rng-{i}.pt" for i in range(world_size())}
        if set(meta.get("files", {})) != expected:
            raise ValueError("Incomplete training-state inventory")
        checked_files(path, meta["files"])
        return meta

    meta = rank_zero_call(preflight)

    def restore():
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        unwrap(generator).load_state_dict(state["generator"], strict=True)
        unwrap(critic).load_state_dict(state["critic"], strict=True)
        mixing.load_state_dict(state["mixing"], strict=True)
        if int(mixing.last_refresh_step) > meta["completed_steps"]:
            raise ValueError("Refresh state is ahead of the saved training step")
        opt_g.load_state_dict(state["optimizer_g"])
        opt_d.load_state_dict(state["optimizer_d"])
        rng = torch.load(path / f"rng-{rank()}.pt", map_location="cpu", weights_only=True)
        _restore_rng(rng)

    local_call(restore)
    return meta["completed_steps"]


def export(path, backend, family, step, identity):
    """Single-rank I/O; caller handles collective error propagation if needed."""
    if family not in ("sdxl", "sd35m") or type(step) is not int or step < 0:
        raise ValueError("Invalid generator export family/step")
    path = Path(path)
    temporary = _create_staging(path)
    unwrap(backend.generator).save_pretrained(temporary / "generator", safe_serialization=True)
    backend.scheduler.save_pretrained(temporary / "scheduler")
    # Diffusers may remember local snapshot paths in otherwise portable configs.
    for config_file in temporary.rglob("*config.json"):
        config = json.loads(config_file.read_text())
        if "_name_or_path" in config:
            config.pop("_name_or_path")
            write_json(config_file, config, replace=True)
    training = identity["config"]["training"]
    vae_dtype = training["vae_dtype"]
    # Keep machine paths, dataset locations, full config and optimizer metadata
    # out of generator artifacts. A hash links back to the private run record.
    public_identity = {
        "base": identity["base"],
        "config": {"training": {"vae_dtype": vae_dtype}},
        "training_identity_sha256": object_digest(identity),
    }
    write_json(temporary / "road.json", {
        "schema": EXPORT_SCHEMA,
        "family": family,
        "completed_steps": step,
        "identity": public_identity,
        "checkpoint_type": "generator_only",
        "conversion": {
            key: identity[key]
            for key in ("imported_state_sha256", "conversion", "historical_result_reproduced")
            if key in identity
        },
        "inference": {
            "resolution": 1024,
            "num_generator_evaluations": 1,
            "classifier_free_guidance": False,
            "precision": training.get("precision", "bf16"),
            "vae_dtype": vae_dtype,
            "text_encoder_dtype": "fp32" if family == "sdxl" else "bf16",
            "noise_parameterization": "epsilon_t999" if family == "sdxl" else "flow_sigma1",
        },
        "files": {str(item.relative_to(temporary)): digest(item) for item in sorted(temporary.rglob("*")) if item.is_file()},
    })
    os.replace(temporary, path)
