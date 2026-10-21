"""Small strict YAML contract: only refresh and R1 are scientific switches."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Assets:
    base_model: str = ""
    dinov3: str = ""
    manifest: str = ""
    text_cache: str = ""


@dataclass
class Mixing:
    refresh_interval: int | None = 5000


@dataclass
class Regularization:
    type: str = "jr1"
    gamma: float = 0.0075
    lazy_interval: int = 16


@dataclass
class Training:
    output: str = "outputs/road"
    seed: int = 10
    steps: int = 100000
    microbatch: int = 2
    accumulation: int = 4
    world_size: int = 8
    global_batch: int = 64
    generator_lr: float = 5e-7
    critic_lr: float = 5e-6
    generator_clip: float = 10.0
    critic_clip: float = 10.0
    precision: str = "bf16"
    vae_dtype: str = "bf16"
    generator_checkpointing: bool = True
    vae_checkpointing: bool = True
    critic_checkpointing: bool = True
    feature_checkpointing: bool = True
    generator_vjp: str = "three_stage"
    zero1: bool = True
    checkpoint_interval: int = 5000
    export_interval: int = 500
    log_interval: int = 10
    data_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class Config:
    family: str = "sdxl"
    assets: Assets = field(default_factory=Assets)
    mixing: Mixing = field(default_factory=Mixing)
    regularization: Regularization = field(default_factory=Regularization)
    training: Training = field(default_factory=Training)

    def validate(self):
        t = self.training
        r = self.regularization
        if type(t.seed) is not int or t.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in (
            "generator_checkpointing",
            "vae_checkpointing",
            "critic_checkpointing",
            "feature_checkpointing",
            "zero1",
            "pin_memory",
        ):
            if type(getattr(t, key)) is not bool:
                raise ValueError(key + " must be a YAML boolean")
        if type(t.data_workers) is not int or t.data_workers < 0:
            raise ValueError("data_workers must be a nonnegative integer")
        positive(t.prefetch_factor, "prefetch_factor")
        for item in fields(self.assets):
            value = getattr(self.assets, item.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"assets.{item.name} requires a local path")
        if not isinstance(t.output, str) or not t.output.strip():
            raise ValueError("training.output requires a directory")
        if self.family not in ("sdxl", "sd35m"):
            raise ValueError("Unknown model family")
        if r.type not in ("jr1", "pooled_r1"):
            raise ValueError("Unknown R1 type")
        if type(r.gamma) not in (int, float) or not math.isfinite(r.gamma) or r.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        for key in (
            "steps",
            "microbatch",
            "accumulation",
            "world_size",
            "global_batch",
            "checkpoint_interval",
            "export_interval",
            "log_interval",
        ):
            positive(getattr(t, key), key)
        positive(r.lazy_interval, "lazy_interval")
        if self.mixing.refresh_interval is not None:
            positive(self.mixing.refresh_interval, "refresh_interval")
        if t.microbatch * t.accumulation * t.world_size != t.global_batch:
            raise ValueError(
                "global_batch must equal world_size * microbatch * accumulation"
            )
        for key in ("generator_lr", "critic_lr", "generator_clip", "critic_clip"):
            v = getattr(t, key)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValueError(key + " must be positive and finite")
        if t.precision not in ("fp32", "bf16") or t.vae_dtype not in ("fp32", "bf16"):
            raise ValueError("Only fp32/bf16 are supported")
        if t.generator_vjp not in ("direct", "three_stage"):
            raise ValueError("Invalid generator_vjp")
        return self

    def to_dict(self):
        return asdict(self)


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")


def strict(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**value)


def load(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    if data.get("family") == "sd35m":
        regularization = data.setdefault("regularization", {})
        if isinstance(regularization, dict):
            regularization.setdefault("gamma", 0.00075)
    for key, cls in [
        ("assets", Assets),
        ("mixing", Mixing),
        ("regularization", Regularization),
        ("training", Training),
    ]:
        if key in data:
            data[key] = strict(cls, data[key])
    return strict(Config, data).validate()
