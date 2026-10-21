"""Strictly import a flat research generator state dictionary for inference."""

import argparse
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from .artifacts import asset_fingerprint, digest
from .checkpoints import export


def load_generator_state(model, path):
    state = (
        load_file(str(path), device="cpu")
        if str(path).endswith(".safetensors")
        else torch.load(path, map_location="cpu", weights_only=True)
    )
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor)
            for k, v in state.items()
        )
    ):
        raise ValueError(
            "Expected a flat generator-only tensor dictionary; nested/full checkpoints are not guessed"
        )
    model.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["sdxl", "sd35m"], required=True)
    for name in ("base-model", "state-dict", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--completed-steps", type=int, required=True)
    parser.add_argument("--vae-dtype", choices=["bf16", "fp32"], required=True)
    args = parser.parse_args()
    if args.completed_steps < 0:
        raise ValueError("completed-steps must be nonnegative")
    from diffusers import (
        DDPMScheduler,
        FlowMatchEulerDiscreteScheduler,
        SD3Transformer2DModel,
        UNet2DConditionModel,
    )

    cls = UNet2DConditionModel if args.family == "sdxl" else SD3Transformer2DModel
    folder = "unet" if args.family == "sdxl" else "transformer"
    model = cls.from_config(
        cls.load_config(args.base_model, subfolder=folder, local_files_only=True)
    )
    load_generator_state(model, args.state_dict)
    scheduler_cls = (
        DDPMScheduler if args.family == "sdxl" else FlowMatchEulerDiscreteScheduler
    )
    scheduler = scheduler_cls.from_pretrained(
        args.base_model, subfolder="scheduler", local_files_only=True
    )
    identity = {
        "base": asset_fingerprint(args.base_model),
        "config": {"training": {"vae_dtype": args.vae_dtype}},
        "imported_state_sha256": digest(args.state_dict),
        "conversion": "strict_flat_native_generator",
        "historical_result_reproduced": False,
    }
    export(
        args.output,
        SimpleNamespace(generator=model, scheduler=scheduler),
        args.family,
        args.completed_steps,
        identity,
    )


if __name__ == "__main__":
    main()
