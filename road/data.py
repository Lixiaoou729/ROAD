"""Manifest-indexed RGB input and exact step-addressed batches with stateless prefetch."""

import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler

from .artifacts import digest


def records(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Manifest must contain sample objects")
    if any(not isinstance(row.get("id"), str) or not row["id"] for row in rows):
        raise ValueError("Manifest needs nonempty string sample IDs")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    for row in rows:
        if not isinstance(row.get("caption"), str):
            raise ValueError("Each sample requires a caption")
        if not isinstance(row.get("image"), str) or not row["image"]:
            raise ValueError("Each sample requires an image/shard path")
        if ("offset" in row) != ("size" in row):
            raise ValueError("Tar byte offset and size must be supplied together")
        if "offset" in row and (
            "member" not in row
            or type(row["offset"]) is not int or row["offset"] < 0
            or type(row["size"]) is not int or row["size"] <= 0
        ):
            raise ValueError("Invalid tar byte range")
    return rows


class Images:
    def __init__(self, manifest, family):
        self.path = Path(manifest).resolve()
        self.rows = records(self.path)
        if family not in ("sdxl", "sd35m"):
            raise ValueError("Unknown image model family")
        self.family = family
        self.fingerprint = digest(self.path)

    def verify_assets(self):
        """Verify source bytes once before training; bind them into resume identity."""
        sidecar = self.path.with_suffix(".assets.json")
        meta = json.loads(sidecar.read_text())
        if meta["manifest_sha256"] != self.fingerprint or meta["count"] != len(
            self.rows
        ):
            raise ValueError("Dataset sidecar does not match manifest")
        expected = {row["image"] for row in self.rows}
        if set(meta["shards"]) != expected:
            raise ValueError("Dataset asset inventory does not match manifest")
        for name, sha in meta["shards"].items():
            if digest(self.path.parent / name) != sha:
                raise ValueError(f"Dataset bytes changed: {name}")
        return digest(sidecar)

    def read(self, index, seed):
        row = self.rows[index]
        p = self.path.parent / row["image"]
        if "member" in row:
            if "offset" in row:
                with p.open("rb") as source:
                    source.seek(row["offset"])
                    data = source.read(row["size"])
                if len(data) != row["size"]:
                    raise ValueError("Truncated tar member")
            else:
                with tarfile.open(p) as tar:
                    data = tar.extractfile(row["member"]).read()
            image = Image.open(io.BytesIO(data))
        else:
            image = Image.open(p)
        with image:
            image = ImageOps.exif_transpose(image)
            if self.family == "sdxl" and (
                "A" in image.getbands() or "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba)
            image = image.convert("RGB")
            rng = random.Random(seed)
            if self.family == "sdxl":
                # Reference XL input is already square 512 RGB. Fail instead of
                # silently applying a second crop to a mismatched dataset.
                if image.size != (512, 512):
                    raise ValueError(
                        "SDXL training requires prepared 512x512 RGB images"
                    )
            else:
                # Native SD3.5 RGBProcessor: shorter-edge Lanczos, random crop.
                w, h = image.size
                scale = 1024 / min(w, h)
                image = image.resize(
                    (round(w * scale), round(h * scale)), Image.Resampling.LANCZOS
                )
                w, h = image.size
                x = rng.randint(0, w - 1024)
                y = rng.randint(0, h - 1024)
                image = image.crop((x, y, x + 1024, y + 1024))
            if self.family == "sd35m" and rng.random() < 0.5:
                image = ImageOps.mirror(image)
            return (
                torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
                / 127.5
                - 1
            )


class StepBatches:
    """Global batch order determined solely by epoch, seed, and completed step.

    Each epoch drops its incomplete global batch. Resume does not replay data
    or need worker state. World size/accumulation are fixed across full resume.
    """

    def __init__(self, size, global_batch, seed):
        self.size = size
        self.batch = global_batch
        self.seed = seed
        self.steps_per_epoch = size // global_batch
        if not self.steps_per_epoch:
            raise ValueError("Dataset smaller than global batch")
        self.epoch = None
        self.order = None

    def indices(self, step, rank, microbatch, accumulation, world_size):
        if microbatch * accumulation * world_size != self.batch:
            raise ValueError("Invalid effective batch")
        epoch, offset = divmod(step, self.steps_per_epoch)
        if epoch != self.epoch:
            self.order = torch.randperm(
                self.size, generator=torch.Generator().manual_seed(self.seed + epoch)
            ).tolist()
            self.epoch = epoch
        batch = self.order[offset * self.batch : (offset + 1) * self.batch]
        local = microbatch * accumulation
        return batch[rank * local : (rank + 1) * local]


class TrainingSamples(Dataset):
    """Worker-local image and text reads; the sampler supplies all randomness."""

    def __init__(self, images, cache):
        self.images = images
        self.cache = cache

    def __len__(self):
        return len(self.images.rows)

    def __getitem__(self, key):
        index, seed, step, micro = key
        return {
            "image": self.images.read(index, seed),
            "condition": self.cache.row(index),
            "step": step,
            "micro": micro,
        }


class TrainingBatches(Sampler):
    """Microbatch tasks addressed by completed optimizer step, never prefetch position."""

    def __init__(self, size, settings, start_step, rank):
        self.order = StepBatches(size, settings.global_batch, settings.seed)
        self.settings = settings
        self.start_step = start_step
        self.rank = rank

    def __iter__(self):
        t = self.settings
        local_size = t.microbatch * t.accumulation
        for step in range(self.start_step, t.steps):
            indices = self.order.indices(step, self.rank, t.microbatch, t.accumulation, t.world_size)
            for micro in range(t.accumulation):
                begin = micro * t.microbatch
                yield [
                    (index, t.seed + step * t.global_batch + self.rank * local_size + begin + j, step, micro)
                    for j, index in enumerate(indices[begin:begin + t.microbatch])
                ]

    def __len__(self):
        return max(0, self.settings.steps - self.start_step) * self.settings.accumulation


def training_loader(images, cache, settings, start_step, rank):
    # A private generator keeps DataLoader worker seeding out of training RNG.
    options = dict(
        dataset=TrainingSamples(images, cache),
        batch_sampler=TrainingBatches(len(images.rows), settings, start_step, rank),
        num_workers=settings.data_workers,
        pin_memory=settings.pin_memory,
        generator=torch.Generator().manual_seed(settings.seed + rank),
    )
    if settings.data_workers:
        # Forking a process with a live CUDA/NCCL context is unsafe.
        options.update(multiprocessing_context="spawn", prefetch_factor=settings.prefetch_factor)
    return DataLoader(**options)
