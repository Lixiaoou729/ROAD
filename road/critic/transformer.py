"""Final four-level ROAD head, without historical architecture switches."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import (
    CrossAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    ScaledCosineAttention,
    _make_linear,
)


class Block(nn.Module):
    def __init__(self, dim, heads, depth, rope, cross=None):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.attn = ScaledCosineAttention(dim, heads, qk_dim=dim // 2)
        self.rope = rope
        self.ffn_in = _make_linear(dim, dim * 2)
        self.ffn_out = _make_linear(dim * 2, dim)
        self.act = nn.GELU()
        self.alpha_attn = nn.Parameter(torch.full((dim,), 1 / depth))
        self.alpha_ffn = nn.Parameter(torch.full((dim,), 1 / depth))
        self.cross_attn = cross
        self.gate_ca = nn.Parameter(torch.zeros(dim)) if cross is not None else None

    def forward(self, x, text, mask, n_patch):
        x = x + self.alpha_attn * self.attn(self.norm1(x), self.rope)
        if self.cross_attn is not None:
            update = self.cross_attn(x[:, :n_patch], text, mask) * self.gate_ca
            x = x + torch.cat([update, torch.zeros_like(x[:, n_patch:])], 1)
        return x + self.alpha_ffn * self.ffn_out(self.act(self.ffn_in(self.norm2(x))))


class Head(nn.Module):
    def __init__(
        self, text_dim, depth, dim=768, heads=6, theta=100.0, checkpoint_blocks=True
    ):
        super().__init__()
        self.out_proj = _make_linear(dim, 1)
        self.norm = RMSNorm(dim)
        rope = RotaryPositionEmbedding(dim // 2 // heads, theta=theta, learnable=True)
        cross = CrossAttention(
            dim,
            text_dim,
            heads,
            qk_dim=dim // 2,
            k_proj=_make_linear(text_dim, dim // 2),
            v_proj=_make_linear(text_dim, dim),
            kv_norm=RMSNorm(text_dim, elementwise_affine=False),
        )
        self.blocks = nn.ModuleList(
            [
                Block(dim, heads, depth, rope, cross if i == 0 else None)
                for i in range(depth)
            ]
        )
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, x, text, mask, dummy_registers=4):
        n_patch = x.shape[1] - 1 - dummy_registers
        for block in self.blocks:
            if self.checkpoint_blocks and torch.is_grad_enabled():
                # No running-state changes during recompute: SN power iteration is
                # updated once outside checkpointed blocks by the trainer.
                x = checkpoint(block, x, text, mask, n_patch, use_reentrant=False)
            else:
                x = block(x, text, mask, n_patch)
        return self.out_proj(self.norm(x)).squeeze(-1)[:, : n_patch + 1]


class TransformerCritic(nn.Module):
    def __init__(
        self,
        family,
        dim=768,
        heads=6,
        depths=(4, 3, 2, 1),
        theta=100.0,
        checkpoint_blocks=True,
    ):
        super().__init__()
        if family not in ("sdxl", "sd35m"):
            raise ValueError(family)
        # Active native SD3.5 conditioning adapter is retained. It is not an
        # optional normalization alternative in the Transformer heads.
        self.adapter = (
            nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 1024))
            if family == "sd35m"
            else nn.Identity()
        )
        text_dim = 1024 if family == "sd35m" else 2048
        self.heads = nn.ModuleList(
            [Head(text_dim, d, dim, heads, theta, checkpoint_blocks) for d in depths]
        )

    def forward(self, features, text, mask=None):
        if len(features) != len(self.heads):
            raise ValueError("Expected one feature tensor per critic head")
        if text.ndim != 3:
            raise ValueError("Text conditioning must be [batch, tokens, channels]")
        text = self.adapter(text)
        if mask is None:
            mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != text.shape[:2]:
            raise ValueError("Text mask must be boolean with matching batch/token axes")
        if not mask.any(1).all():
            raise ValueError("Each caption must contain at least one unmasked token")
        return torch.cat(
            [head(x, text, mask) for head, x in zip(self.heads, features)], 1
        )


@torch.no_grad()
def update_spectral_norm(module):
    """Advance SN estimates once per optimizer step, outside recomputation.

    Keep Linear modules in eval mode during forwards; their weights still
    receive gradients. This avoids checkpoint recomputation mutating u/v.
    """
    from torch.nn.utils.spectral_norm import SpectralNorm

    for layer in module.modules():
        for hook in layer._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm):
                hook.compute_weight(layer, do_power_iteration=True)
                layer.training = False
