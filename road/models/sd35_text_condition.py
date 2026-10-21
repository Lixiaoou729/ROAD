"""SD3.5 three-encoder prompt semantics and cache I/O.

The tensor construction follows StableDiffusion3Pipeline in Diffusers v0.36.0:
the two penultimate CLIP hidden states are channel-concatenated, zero-padded to
T5 width, then sequence-concatenated with T5. The two projected CLIP pooled
outputs are channel-concatenated. SD3Transformer2DModel v0.36.0 has no text
attention-mask argument, so ``attention_mask`` is normally ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .sd35_compat import validate_diffusers_version

NATIVE_TOKEN_DIM = 4096

NATIVE_POOLED_DIM = 2048


@dataclass
class TextCondition:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor | None = None

    def validate(self, *, batch_size: int | None = None) -> "TextCondition":
        if (
            self.prompt_embeds.ndim != 3
            or self.prompt_embeds.shape[-1] != NATIVE_TOKEN_DIM
        ):
            raise ValueError(
                "SD3.5 prompt_embeds must have shape [B, S, 4096], got "
                f"{tuple(self.prompt_embeds.shape)}."
            )
        if (
            self.pooled_prompt_embeds.ndim != 2
            or self.pooled_prompt_embeds.shape[-1] != NATIVE_POOLED_DIM
        ):
            raise ValueError(
                "SD3.5 pooled_prompt_embeds must have shape [B, 2048], got "
                f"{tuple(self.pooled_prompt_embeds.shape)}."
            )
        if self.prompt_embeds.shape[0] != self.pooled_prompt_embeds.shape[0]:
            raise ValueError("Token and pooled condition batch sizes differ.")
        if batch_size is not None and self.prompt_embeds.shape[0] != batch_size:
            raise ValueError(
                f"Text condition batch is {self.prompt_embeds.shape[0]}, expected {batch_size}."
            )
        if self.attention_mask is not None and tuple(
            self.attention_mask.shape
        ) != tuple(self.prompt_embeds.shape[:2]):
            raise ValueError(
                f"attention_mask shape {tuple(self.attention_mask.shape)} does not match "
                f"prompt tokens {tuple(self.prompt_embeds.shape[:2])}."
            )
        return self

    def to(
        self, *, device: torch.device | str, dtype: torch.dtype | None = None
    ) -> "TextCondition":
        return TextCondition(
            prompt_embeds=self.prompt_embeds.to(
                device=device, dtype=dtype or self.prompt_embeds.dtype
            ),
            pooled_prompt_embeds=self.pooled_prompt_embeds.to(
                device=device, dtype=dtype or self.pooled_prompt_embeds.dtype
            ),
            attention_mask=None
            if self.attention_mask is None
            else self.attention_mask.to(device=device),
        )


class SD35OnlineTextEncoder:
    """Reference online encoder; intended for cache building and parity tests."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
    ) -> None:
        validate_diffusers_version()
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )

        common: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
        }
        common = {key: value for key, value in common.items() if value is not None}
        model_common = {**common, "torch_dtype": dtype}
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", **common
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2", **common
        )
        self.tokenizer_3 = T5TokenizerFast.from_pretrained(
            model_id, subfolder="tokenizer_3", **common
        )
        self.text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(
                model_id, subfolder="text_encoder", **model_common
            )
            .eval()
            .to(self.device)
        )
        self.text_encoder_2 = (
            CLIPTextModelWithProjection.from_pretrained(
                model_id, subfolder="text_encoder_2", **model_common
            )
            .eval()
            .to(self.device)
        )
        self.text_encoder_3 = (
            T5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder_3", **model_common
            )
            .eval()
            .to(self.device)
        )
        for module in (self.text_encoder, self.text_encoder_2, self.text_encoder_3):
            module.requires_grad_(False)
        self.tokenizer_max_length = int(self.tokenizer.model_max_length)

    def _encode_clip(
        self, prompts: Sequence[str], *, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.tokenizer if index == 0 else self.tokenizer_2
        encoder = self.text_encoder if index == 0 else self.text_encoder_2
        inputs = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        outputs = encoder(inputs.input_ids.to(self.device), output_hidden_states=True)
        pooled = outputs[0]
        tokens = outputs.hidden_states[-2]
        # Pipeline v0.36.0 normalizes both CLIP streams to text_encoder.dtype.
        tokens = tokens.to(device=self.device, dtype=self.text_encoder.dtype)
        pooled = pooled.to(device=self.device, dtype=self.text_encoder.dtype)
        return tokens, pooled

    def _encode_t5(
        self, prompts: Sequence[str], *, max_sequence_length: int
    ) -> torch.Tensor:
        inputs = self.tokenizer_3(
            list(prompts),
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return self.text_encoder_3(inputs.input_ids.to(self.device))[0].to(
            device=self.device, dtype=self.text_encoder_3.dtype
        )

    @torch.no_grad()
    def encode(
        self, prompts: Sequence[str], *, t5_max_sequence_length: int = 77
    ) -> TextCondition:
        if t5_max_sequence_length != 77:
            raise ValueError("ROAD uses t5_max_sequence_length=77.")
        normalized = [str(prompt) for prompt in prompts]
        clip_l, pooled_l = self._encode_clip(normalized, index=0)
        clip_g, pooled_g = self._encode_clip(normalized, index=1)
        t5 = self._encode_t5(normalized, max_sequence_length=t5_max_sequence_length)
        clip = torch.cat([clip_l, clip_g], dim=-1)
        if clip.shape[-1] > t5.shape[-1]:
            raise RuntimeError(
                f"Combined CLIP width {clip.shape[-1]} exceeds T5 width {t5.shape[-1]}."
            )
        clip = F.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
        condition = TextCondition(
            prompt_embeds=torch.cat([clip, t5], dim=-2),
            pooled_prompt_embeds=torch.cat([pooled_l, pooled_g], dim=-1),
            attention_mask=None,
        )
        return condition.validate(batch_size=len(normalized))
