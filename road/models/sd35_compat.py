"""Pinned-runtime and SD3.5 Medium checkpoint validation."""

from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

REQUIRED_DIFFUSERS_VERSION = "0.36.0"


def validate_diffusers_version() -> str:
    """Reject unvalidated Diffusers releases before model construction."""

    try:
        observed = metadata.version("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "diffusers is not installed. This project requires exactly "
            f"diffusers=={REQUIRED_DIFFUSERS_VERSION}."
        ) from exc
    if observed != REQUIRED_DIFFUSERS_VERSION:
        raise RuntimeError(
            f"Incompatible Diffusers version {observed!r}; expected exactly "
            f"{REQUIRED_DIFFUSERS_VERSION!r}. Install the pinned dependency."
        )
    return observed


def validate_sd35_api(transformer_cls: type) -> None:
    parameters = inspect.signature(transformer_cls.__init__).parameters
    missing = [
        name for name in ("qk_norm", "dual_attention_layers") if name not in parameters
    ]
    if missing:
        raise RuntimeError(
            "The installed SD3Transformer2DModel cannot represent SD3.5 Medium; "
            f"constructor fields missing: {missing}. Install diffusers=={REQUIRED_DIFFUSERS_VERSION}."
        )


def _config_value(config: Any, name: str) -> Any:
    if not hasattr(config, name):
        raise RuntimeError(
            f"SD3.5 transformer config is missing required field {name!r}."
        )
    return getattr(config, name)


def validate_sd35_transformer_config(config: Any) -> dict[str, Any]:
    expected = {
        "in_channels": 16,
        "out_channels": 16,
        "joint_attention_dim": 4096,
        "pooled_projection_dim": 2048,
    }
    observed: dict[str, Any] = {}
    for name, expected_value in expected.items():
        value = _config_value(config, name)
        observed[name] = value
        if int(value) != expected_value:
            raise RuntimeError(
                f"Incompatible SD3.5 Medium transformer config: {name}={value!r}, "
                f"expected {expected_value!r}."
            )

    qk_norm = _config_value(config, "qk_norm")
    if not isinstance(qk_norm, str) or not qk_norm.strip():
        raise RuntimeError(
            f"SD3.5 Medium requires checkpoint-specified QK normalization; got qk_norm={qk_norm!r}."
        )
    raw_dual_layers = _config_value(config, "dual_attention_layers")
    try:
        dual_layers = tuple(int(index) for index in raw_dual_layers)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed SD3.5 dual_attention_layers={raw_dual_layers!r}; expected integer indices."
        ) from exc
    if not dual_layers:
        raise RuntimeError(
            "SD3.5 Medium dual_attention_layers must be present and non-empty."
        )
    num_layers = int(_config_value(config, "num_layers"))
    invalid = [index for index in dual_layers if index < 0 or index >= num_layers]
    if invalid:
        raise RuntimeError(
            f"dual_attention_layers contains indices outside [0, {num_layers}): {invalid}."
        )
    patch_size = int(_config_value(config, "patch_size"))
    if patch_size <= 0 or num_layers <= 0:
        raise RuntimeError(
            f"Invalid SD3.5 transformer geometry: patch_size={patch_size}, num_layers={num_layers}."
        )
    observed.update(
        qk_norm=qk_norm,
        dual_attention_layers=list(dual_layers),
        patch_size=patch_size,
        num_layers=num_layers,
    )
    return observed
