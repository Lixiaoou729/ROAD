"""Model-native Stable Diffusion 3.5 Medium one-step generator backend."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .sd35_compat import (
    validate_diffusers_version,
    validate_sd35_api,
    validate_sd35_transformer_config,
)
from .sd35_text_condition import TextCondition


@dataclass(frozen=True)
class ConditioningSchedule:
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    source: str
    shift: float
    use_dynamic_shifting: bool
    resolution: tuple[int, int] | None
    dynamic_shift_mu: float | None
    image_sequence_length: int | None
    base_shift: float | None
    max_shift: float | None
    shift_type: str | None


@dataclass(frozen=True)
class SigmaResolution:
    resolved_sigma: float
    resolved_timestep: float
    grid_index: int
    schedule_source: str


@dataclass
class OneStepPrediction:
    model_output: torch.Tensor
    pred_clean_latent: torch.Tensor
    sigma: SigmaResolution


def predict_clean_from_flow(
    flow_state: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Convert SD3 velocity to x0: ``x0 = x_sigma - sigma * velocity``.

    Diffusers' FlowMatchEulerDiscreteScheduler.step upcasts the state update to
    FP32 and casts the result back to model-output dtype. This helper mirrors
    that behavior and centralizes the only clean-latent reconstruction used by
    training, logging, and evaluation.
    """

    if tuple(flow_state.shape) != tuple(model_output.shape):
        raise ValueError(
            f"Flow state and model output shapes differ: {tuple(flow_state.shape)} vs "
            f"{tuple(model_output.shape)}."
        )
    sigma_tensor = torch.as_tensor(sigma, device=flow_state.device, dtype=torch.float32)
    while sigma_tensor.ndim < flow_state.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)
    clean = flow_state.float() - sigma_tensor * model_output.float()
    return clean.to(dtype=model_output.dtype)


def make_flow_state(
    clean: torch.Tensor, noise: torch.Tensor, sigma: float | torch.Tensor
) -> torch.Tensor:
    if tuple(clean.shape) != tuple(noise.shape):
        raise ValueError(
            f"Clean/noise shapes differ: {tuple(clean.shape)} vs {tuple(noise.shape)}."
        )
    sigma_tensor = torch.as_tensor(sigma, device=clean.device, dtype=torch.float32)
    while sigma_tensor.ndim < clean.ndim:
        sigma_tensor = sigma_tensor.unsqueeze(-1)
    state = (1.0 - sigma_tensor) * clean.float() + sigma_tensor * noise.float()
    return state.to(dtype=clean.dtype)


def _calculate_dynamic_shift(scheduler_config: Any, *, image_seq_len: int) -> float:
    base_seq_len = int(scheduler_config.base_image_seq_len)
    max_seq_len = int(scheduler_config.max_image_seq_len)
    base_shift = float(scheduler_config.base_shift)
    max_shift = float(scheduler_config.max_shift)
    if max_seq_len <= base_seq_len:
        raise RuntimeError(
            f"Dynamic-shift max_image_seq_len must exceed base_image_seq_len: {max_seq_len} <= {base_seq_len}."
        )
    slope = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    return image_seq_len * slope + (base_shift - slope * base_seq_len)


def _initialized_training_arrays(scheduler: Any) -> tuple[torch.Tensor, torch.Tensor]:
    expected = int(scheduler.config.num_train_timesteps)
    timesteps = (
        scheduler.timesteps.detach().clone().to(device="cpu", dtype=torch.float32)
    )
    raw_sigmas = scheduler.sigmas.detach().clone().to(device="cpu", dtype=torch.float32)
    if timesteps.ndim != 1 or raw_sigmas.ndim != 1:
        raise RuntimeError(
            "Scheduler training timestep/sigma arrays must be one-dimensional."
        )
    if len(timesteps) != expected:
        raise RuntimeError(
            f"Initialized scheduler exposes {len(timesteps)} training timesteps; expected {expected}."
        )
    if len(raw_sigmas) == expected + 1:
        terminal = float(raw_sigmas[-1].item())
        if terminal not in (0.0, 1.0):
            raise RuntimeError(
                f"Scheduler has an unrecognized terminal-only sigma {terminal}; expected 0 or 1."
            )
        sigmas = raw_sigmas[:-1]
    elif len(raw_sigmas) == expected:
        sigmas = raw_sigmas
    else:
        raise RuntimeError(
            f"Initialized scheduler exposes {len(raw_sigmas)} sigmas for {expected} training steps."
        )
    if len(sigmas) != len(timesteps) or len(sigmas) == 0:
        raise RuntimeError(
            "Scheduler training sigma/timestep arrays are empty or misaligned."
        )
    if not bool(torch.isfinite(sigmas).all() and torch.isfinite(timesteps).all()):
        raise RuntimeError("Scheduler training arrays contain NaN/Inf.")
    if bool(torch.any(sigmas[:-1] < sigmas[1:])) or bool(
        torch.any(timesteps[:-1] < timesteps[1:])
    ):
        raise RuntimeError(
            "Scheduler training sigma/timestep arrays must be monotonically non-increasing."
        )
    return timesteps, sigmas


def build_conditioning_schedule(
    scheduler: Any,
    *,
    latent_height: int,
    latent_width: int,
    patch_size: int,
    image_height: int | None = None,
    image_width: int | None = None,
) -> ConditioningSchedule:
    """Build one immutable training grid without invoking the inference scheduler path."""

    if latent_height <= 0 or latent_width <= 0 or patch_size <= 0:
        raise ValueError(
            "Latent dimensions and transformer patch size must be positive."
        )
    if latent_height % patch_size or latent_width % patch_size:
        raise ValueError(
            f"Latent shape {(latent_height, latent_width)} is not divisible by patch size {patch_size}."
        )
    timesteps, sigmas = _initialized_training_arrays(scheduler)
    config = scheduler.config
    dynamic = bool(getattr(config, "use_dynamic_shifting", False))
    shift = float(getattr(config, "shift", getattr(scheduler, "shift", 1.0)))
    resolution = (
        int(image_height if image_height is not None else latent_height),
        int(image_width if image_width is not None else latent_width),
    )
    if not dynamic:
        expected_timesteps = sigmas * int(config.num_train_timesteps)
        if not torch.allclose(timesteps, expected_timesteps, rtol=0, atol=1e-4):
            raise RuntimeError(
                "Loaded fixed-shift scheduler training timesteps are not paired with its sigmas."
            )
        return ConditioningSchedule(
            timesteps=timesteps,
            sigmas=sigmas,
            source="loaded_scheduler_initialized_fixed_shift_training_arrays",
            shift=shift,
            use_dynamic_shifting=False,
            resolution=None,
            dynamic_shift_mu=None,
            image_sequence_length=None,
            base_shift=None,
            max_shift=None,
            shift_type=None,
        )

    unsupported = [
        name
        for name in (
            "invert_sigmas",
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
        )
        if bool(getattr(config, name, False))
    ]
    if unsupported:
        raise RuntimeError(
            "Dynamic training-grid construction does not support additional inference schedule transforms: "
            f"{unsupported}. Refusing to mix schedule paths."
        )
    image_seq_len = (latent_height // patch_size) * (latent_width // patch_size)
    mu = _calculate_dynamic_shift(config, image_seq_len=image_seq_len)
    shift_type = str(getattr(config, "time_shift_type", "exponential"))
    if shift_type not in ("exponential", "linear"):
        raise RuntimeError(
            f"Unsupported Diffusers dynamic time_shift_type {shift_type!r}."
        )
    # With dynamic shifting enabled, Diffusers 0.36.0 initializes raw training
    # sigmas and defers this exact transform to resolution-aware inference.
    shifted_sigmas = scheduler.time_shift(mu, 1.0, sigmas)
    if getattr(config, "shift_terminal", None):
        shifted_sigmas = scheduler.stretch_shift_to_terminal(shifted_sigmas)
    shifted_timesteps = shifted_sigmas * int(config.num_train_timesteps)
    if bool(torch.any(shifted_sigmas[:-1] < shifted_sigmas[1:])):
        raise RuntimeError(
            "Resolution-aware dynamically shifted sigmas are not monotonic."
        )
    return ConditioningSchedule(
        timesteps=shifted_timesteps.detach().clone(),
        sigmas=shifted_sigmas.detach().clone(),
        source="loaded_scheduler_initialized_raw_training_arrays+diffusers_0.36_dynamic_shift_once",
        shift=shift,
        use_dynamic_shifting=True,
        resolution=resolution,
        dynamic_shift_mu=float(mu),
        image_sequence_length=int(image_seq_len),
        base_shift=float(config.base_shift),
        max_shift=float(config.max_shift),
        shift_type=shift_type,
    )


def resolve_conditioning_sigma(scheduler: ConditioningSchedule) -> SigmaResolution:
    """ROAD uses pure Gaussian input at sigma=1; reject incompatible assets."""
    sigmas = scheduler.sigmas
    index = int(torch.argmin((sigmas - 1).abs()))
    sigma = float(sigmas[index])
    if abs(sigma - 1) > 1e-7:
        raise ValueError(
            "ROAD requires an actual sigma=1 point in the native training schedule"
        )
    return SigmaResolution(
        sigma, float(scheduler.timesteps[index]), index, scheduler.source
    )


class SD35GeneratorBackend:
    def __init__(
        self,
        *,
        transformer: torch.nn.Module,
        scheduler: Any,
        vae: torch.nn.Module,
        gradient_checkpointing: bool = True,
        vae_gradient_checkpointing: bool = False,
        vae_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.resolution = 1024
        self.model_config = validate_sd35_transformer_config(transformer.config)

        block_channels = tuple(int(value) for value in vae.config.block_out_channels)
        self.vae_scale_factor = 2 ** (len(block_channels) - 1)
        if self.resolution % self.vae_scale_factor:
            raise ValueError(
                f"Resolution {self.resolution} is not divisible by VAE scale factor {self.vae_scale_factor}."
            )
        self.latent_height = self.resolution // self.vae_scale_factor
        self.latent_width = self.latent_height
        self.latent_channels = int(transformer.config.in_channels)
        vae_latent_channels = int(vae.config.latent_channels)
        if vae_latent_channels != self.latent_channels:
            raise RuntimeError(
                f"Transformer/VAE latent-channel mismatch: {self.latent_channels} vs {vae_latent_channels}."
            )
        self.latent_shape = (
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )
        self.scaling_factor = float(vae.config.scaling_factor)
        shift = getattr(vae.config, "shift_factor", None)
        if shift is None:
            raise RuntimeError(
                "SD3.5 VAE config must define shift_factor; refusing SDXL-style normalization."
            )
        self.shift_factor = float(shift)
        if vae_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"VAE dtype must be float32 or bfloat16, got {vae_dtype}.")
        self.vae_dtype = vae_dtype
        self.last_vae_decode_input_dtype: str | None = None
        self.last_vae_decode_output_dtype: str | None = None

        self.conditioning_schedule = build_conditioning_schedule(
            scheduler,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            patch_size=int(transformer.config.patch_size),
            image_height=self.resolution,
            image_width=self.resolution,
        )
        self.sigma = resolve_conditioning_sigma(self.conditioning_schedule)
        if gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
            if not bool(getattr(transformer, "gradient_checkpointing", False)):
                raise RuntimeError(
                    "SD3 transformer did not enable gradient checkpointing."
                )
        if vae_gradient_checkpointing:
            vae.enable_gradient_checkpointing()
            if not bool(getattr(vae.decoder, "gradient_checkpointing", False)):
                raise RuntimeError(
                    "AutoencoderKL decoder did not enable gradient checkpointing."
                )
        self.vae_gradient_checkpointing = bool(vae_gradient_checkpointing)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        transformer_path: str | None = None,
        scheduler_path: str | None = None,
        transformer_dtype: torch.dtype = torch.float32,
        vae_dtype: torch.dtype = torch.float32,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> "SD35GeneratorBackend":
        validate_diffusers_version()
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            SD3Transformer2DModel,
        )

        validate_sd35_api(SD3Transformer2DModel)
        common: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
        }
        common = {key: value for key, value in common.items() if value is not None}
        if transformer_path:
            transformer = SD3Transformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=transformer_dtype,
                local_files_only=local_files_only,
            )
        else:
            transformer = SD3Transformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                torch_dtype=transformer_dtype,
                **common,
            )
        scheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(
                scheduler_path, local_files_only=local_files_only
            )
            if scheduler_path
            else FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler", **common
            )
        )
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            **common,
        )
        transformer.requires_grad_(True)
        vae.requires_grad_(False).eval()
        return cls(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            vae_dtype=vae_dtype,
            **kwargs,
        )

    def model_latent_from_vae(
        self,
        vae_latent: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self._validate_latent(vae_latent)
        # Centralize the SD3.5 normalization in configured VAE arithmetic.
        latent = vae_latent.to(dtype=self.vae_dtype)
        context = (
            torch.autocast(device_type=latent.device.type, enabled=False)
            if self.vae_dtype == torch.float32
            else contextlib.nullcontext()
        )
        with context:
            normalized = (latent - self.shift_factor) * self.scaling_factor
        return normalized if output_dtype is None else normalized.to(dtype=output_dtype)

    def vae_latent_from_model(self, model_latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(model_latent)
        # Cast before divide/add: casting only the result would leave BF16
        # rounding in the latent normalization even under the FP32 VAE policy.
        latent = model_latent.to(dtype=self.vae_dtype)
        context = (
            torch.autocast(device_type=latent.device.type, enabled=False)
            if self.vae_dtype == torch.float32
            else contextlib.nullcontext()
        )
        with context:
            return latent / self.scaling_factor + self.shift_factor

    def _validate_latent(self, latent: torch.Tensor) -> None:
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"SD3.5 latent must have shape [B, {self.latent_channels}, {self.latent_height}, "
                f"{self.latent_width}], got {tuple(latent.shape)}. Four-channel SDXL latents are incompatible."
            )

    def _validate_vae_policy(self) -> None:
        if self.vae.training:
            raise RuntimeError("Frozen SD3.5 VAE must remain in evaluation mode.")
        parameter_dtypes = {parameter.dtype for parameter in self.vae.parameters()}
        if parameter_dtypes and parameter_dtypes != {self.vae_dtype}:
            raise RuntimeError(
                "VAE parameter dtypes differ from the configured policy: "
                f"observed={sorted(map(str, parameter_dtypes))}, configured={self.vae_dtype}."
            )

    def one_step(
        self, state: torch.Tensor, condition: TextCondition
    ) -> OneStepPrediction:
        self._validate_latent(state)
        condition.validate(batch_size=state.shape[0])
        timestep = torch.full(
            (state.shape[0],),
            self.sigma.resolved_timestep,
            device=state.device,
            dtype=torch.float32,
        )
        model_output = self.transformer(
            hidden_states=state,
            timestep=timestep,
            encoder_hidden_states=condition.prompt_embeds.to(
                device=state.device, dtype=state.dtype
            ),
            pooled_projections=condition.pooled_prompt_embeds.to(
                device=state.device, dtype=state.dtype
            ),
            return_dict=False,
        )[0]
        self._validate_latent(model_output)
        if model_output.shape[0] != state.shape[0]:
            raise RuntimeError(
                "SD3.5 Transformer changed the runtime batch dimension: "
                f"input={state.shape[0]}, output={model_output.shape[0]}."
            )
        clean = predict_clean_from_flow(state, model_output, self.sigma.resolved_sigma)
        return OneStepPrediction(
            model_output=model_output, pred_clean_latent=clean, sigma=self.sigma
        )

    def scheduler_reference_clean(
        self,
        state: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler scheduler step from resolved sigma directly to sigma=0."""

        reference = copy.deepcopy(self.scheduler)
        device = state.device
        reference.timesteps = torch.tensor(
            [self.sigma.resolved_timestep], device=device, dtype=torch.float32
        )
        reference.sigmas = torch.tensor(
            [self.sigma.resolved_sigma, 0.0], device=device, dtype=torch.float32
        )
        reference._step_index = None
        reference._begin_index = None
        return reference.step(
            model_output,
            reference.timesteps[0],
            state,
            return_dict=False,
        )[0]

    def decode_model_latent(self, model_latent: torch.Tensor) -> torch.Tensor:
        self._validate_latent(model_latent)
        self._validate_vae_policy()
        vae_latent = self.vae_latent_from_model(model_latent)
        self.last_vae_decode_input_dtype = str(vae_latent.dtype).replace("torch.", "")
        context = (
            torch.autocast(device_type=vae_latent.device.type, enabled=False)
            if self.vae_dtype == torch.float32
            else contextlib.nullcontext()
        )
        with context:
            decoded = self.vae.decode(vae_latent, return_dict=False)[0]
        expected_shape = (model_latent.shape[0], 3, self.resolution, self.resolution)
        if tuple(decoded.shape) != expected_shape:
            raise RuntimeError(
                f"SD3.5 VAE decode returned {tuple(decoded.shape)}, expected {expected_shape}."
            )
        self.last_vae_decode_output_dtype = str(decoded.dtype).replace("torch.", "")
        return decoded

    def resolved_config(self) -> dict[str, Any]:
        scheduler_config = dict(self.scheduler.config)
        return {
            "model_family": "sd35",
            "resolution": self.resolution,
            "latent_shape": [
                self.latent_channels,
                self.latent_height,
                self.latent_width,
            ],
            "vae_scale_factor": self.vae_scale_factor,
            "vae_scaling_factor": self.scaling_factor,
            "vae_shift_factor": self.shift_factor,
            "vae_dtype": str(self.vae_dtype).replace("torch.", ""),
            "vae_latent_normalization_dtype": str(self.vae_dtype).replace("torch.", ""),
            "vae_decode_input_dtype": str(self.vae_dtype).replace("torch.", ""),
            "vae_decode_output_dtype_observed": self.last_vae_decode_output_dtype,
            "transformer": self.model_config,
            "sigma": asdict(self.sigma),
            "scheduler": scheduler_config,
            "conditioning_schedule": {
                key: value
                for key, value in asdict(self.conditioning_schedule).items()
                if key not in ("timesteps", "sigmas")
            },
            "vae_gradient_checkpointing": self.vae_gradient_checkpointing,
        }
