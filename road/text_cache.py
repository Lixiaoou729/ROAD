"""Bounded safetensors shards; content-addressed metadata, no pickle cache."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import asset_fingerprint, digest, write_json
from .data import records


class TextCache:
    def __init__(self, path, manifest, family, base_fingerprint):
        self.path = Path(path)
        self.meta = json.loads((self.path / "metadata.json").read_text())
        self.count = len(records(manifest))
        expected = {
            "manifest_sha256": digest(manifest),
            "family": family,
            "base_fingerprint": base_fingerprint,
            "count": self.count,
        }
        for key, value in expected.items():
            if self.meta.get(key) != value:
                raise ValueError(f"Text cache identity mismatch: {key}")
        if self.meta.get("schema") != 1:
            raise ValueError("Unsupported text cache schema")
        size = self.meta.get("shard_size")
        if type(size) is not int or size < 1:
            raise ValueError("Invalid text-cache shard size")
        expected_names = {f"{i:06d}.safetensors" for i in range((self.count + size - 1) // size)}
        if set(self.meta.get("shards", {})) != expected_names:
            raise ValueError("Text cache shard inventory is incomplete")
        self.verified = False

    def verify_assets(self):
        """One sequential integrity scan, before sparse worker reads begin."""
        for name, expected in self.meta["shards"].items():
            if digest(self.path / name) != expected:
                raise ValueError(f"Corrupt text-cache shard: {name}")
        self.verified = True

    def row(self, index):
        if not self.verified:
            raise RuntimeError("Verify the immutable text cache before starting workers")
        if not 0 <= index < self.count:
            raise IndexError(index)
        shard, offset = divmod(index, self.meta["shard_size"])
        # mmap only requested rows, not an entire 64-caption shard per sample.
        with safe_open(self.path / f"{shard:06d}.safetensors", framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != {"tokens", "pooled", "mask"}:
                raise ValueError("Invalid text-cache tensor keys")
            return {key: tensors.get_slice(key)[offset].clone() for key in ("tokens", "pooled", "mask")}

    def batch(self, indices, device, dtype):
        # Also usable without DataLoader; explicit verification is required.
        rows = [self.row(index) for index in indices]
        return {
            key: torch.stack([row[key] for row in rows]).to(
                device=device, dtype=torch.bool if key == "mask" else dtype
            )
            for key in ("tokens", "pooled", "mask")
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["sdxl", "sd35m"], required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    if a.batch_size < 1:
        raise ValueError("batch-size must be positive")
    from .models.factory import encoder

    enc = encoder(
        a.family,
        a.base_model,
        a.device,
        torch.float32 if a.family == "sdxl" else torch.bfloat16,
    )
    rows = records(a.manifest)
    size = 64
    shards = {}
    for start in range(0, len(rows), size):
        chunks = []
        for b in range(start, min(start + size, len(rows)), a.batch_size):
            prompts = [
                r["caption"]
                for r in rows[b : min(b + a.batch_size, start + size, len(rows))]
            ]
            chunks.append({key: value.detach().cpu() for key, value in enc.encode(prompts).items()})
        tensors = {
            k: torch.cat([c[k] for c in chunks]).contiguous()
            for k in chunks[0]
        }
        name = f"{start // size:06d}.safetensors"
        save_file(tensors, out / name)
        shards[name] = digest(out / name)
    meta = {
        "schema": 1,
        "family": a.family,
        "manifest_sha256": digest(a.manifest),
        "base_fingerprint": asset_fingerprint(a.base_model),
        "count": len(rows),
        "shard_size": size,
        "shards": shards,
        "t5_length": 77,
    }
    write_json(out / "metadata.json", meta)


if __name__ == "__main__":
    main()
