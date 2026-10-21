"""Standalone one-step inference; does not import DINO or the critic."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .artifacts import asset_fingerprint, checked_files
from .models.factory import backend as make_backend
from .models.factory import encoder


class Pipeline:
    def __init__(self, checkpoint, base_model, device="cuda", precision=None):
        self.device = torch.device(device)
        p = Path(checkpoint)
        if p.name.endswith(".partial"):
            raise ValueError("An unfinished export is not an inference checkpoint")
        self.meta = json.loads((p / "road.json").read_text())
        if (
            self.meta.get("schema") not in (1, 2)
            or self.meta.get("checkpoint_type") != "generator_only"
        ):
            raise ValueError("Expected ROAD generator export")
        files = self.meta.get("files", {})
        actual = {
            str(f.relative_to(p))
            for folder in ("generator", "scheduler")
            for f in (p / folder).rglob("*")
            if f.is_file()
        }
        if not files or set(files) != actual:
            raise ValueError("Export file inventory mismatch")
        checked_files(p, files)
        self.precision = precision or self.meta.get("inference", {}).get("precision", "bf16")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("Inference precision must be bf16 or fp32")
        self.dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        if self.device.type == "cuda" and (not torch.cuda.is_available() or (self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported())):
            raise RuntimeError("Requested CUDA/BF16 execution is unavailable")
        if self.meta["identity"]["base"] != asset_fingerprint(base_model):
            raise ValueError("Base model differs from export manifest")
        family = self.meta["family"]
        self.encoder = encoder(
            family,
            base_model,
            self.device,
            torch.float32 if family == "sdxl" else torch.bfloat16,
        )
        vae_policy = self.meta["identity"]["config"]["training"]["vae_dtype"]
        if vae_policy not in ("fp32", "bf16"):
            raise ValueError("Invalid exported VAE precision")
        self.backend = make_backend(
            family,
            base_model,
            torch.bfloat16 if vae_policy == "bf16" else torch.float32,
            False,
            False,
            str(p / "generator"),
            scheduler_path=str(p / "scheduler"),
        )
        self.backend.generator.to(self.device).eval()
        self.backend.vae.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, prompts, seeds):
        if len(prompts) != len(seeds) or not prompts or any(not isinstance(p, str) for p in prompts):
            raise ValueError("Provide one seed per prompt")
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("Seeds must be integers in [0, 2**63)")
        # CPU per-image RNG makes batching/sharding independent of noise identity.
        noise = torch.stack(
            [
                torch.randn(
                    self.backend.latent_shape,
                    generator=torch.Generator().manual_seed(int(s)),
                    dtype=torch.float32,
                )
                for s in seeds
            ]
        ).to(self.device, self.dtype)
        return self.generate(prompts, noise)

    @torch.inference_mode()
    def generate(self, prompts, noise, *, quantization="truncate"):
        """Native inference from explicit noise, also used by versioned evaluators."""
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Prompts must be a nonempty list of strings")
        if tuple(noise.shape) != (len(prompts), *self.backend.latent_shape):
            raise ValueError("Noise shape does not match prompts/native latent geometry")
        if not noise.is_floating_point() or not torch.isfinite(noise).all():
            raise ValueError("Noise must be a finite floating-point tensor")
        noise = noise.to(device=self.device, dtype=self.dtype)
        cond = {
            k: v.to(device=self.device, dtype=torch.bool if k == "mask" else self.dtype)
            for k, v in self.encoder.encode(prompts).items()
        }
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.dtype == torch.bfloat16
        ):
            latent = self.backend.predict(noise, cond)
        rgb = self.backend.decode(latent)
        if not torch.isfinite(rgb).all():
            raise FloatingPointError("Nonfinite decoded image; no output was written")
        pixels = (rgb.float().clamp(-1, 1) + 1) * 127.5
        if quantization == "round":
            pixels = pixels.round()
        elif quantization != "truncate":
            raise ValueError("Unknown image quantization")
        arrays = (
            pixels.to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        return [Image.fromarray(a) for a in arrays]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    image = Pipeline(a.checkpoint, a.base_model, a.device, a.precision)(
        [a.prompt], [a.seed]
    )[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


if __name__ == "__main__":
    main()
