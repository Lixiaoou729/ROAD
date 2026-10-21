"""Narrowed ROAD attention arithmetic from the SDXL implementation.
See THIRD_PARTY_NOTICES.md and docs/behavior_changes.md.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _make_linear(a, b, bias=True):
    return nn.utils.spectral_norm(nn.Linear(a, b, bias=bias))


class RMSNorm(nn.Module):
    """
    Diffusers-style RMSNorm with optional elementwise affine parameters.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        if self.weight is None:
            return x
        return x * self.weight.view(1, 1, -1)


def _rotate_even_odd(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, learnable: bool = False):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {dim}")
        self.dim = int(dim)
        inv_freq = 1.0 / (float(theta) ** (torch.arange(0, dim, 2).float() / dim))
        if learnable:
            self.inv_freq = nn.Parameter(inv_freq)
        else:
            self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(self, seq_len: int, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("n,d->nd", positions, self.inv_freq.float())
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]

    @staticmethod
    def apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return x * cos + _rotate_even_odd(x) * sin

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        num_summary_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del num_summary_tokens
        cos, sin = self.get_cos_sin(q.shape[-2], device=q.device, dtype=torch.float32)
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)


class ScaledCosineAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_dim: Optional[int] = None,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 100.0,
        num_summary_tokens: int = 0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.qk_dim = int(self.dim if qk_dim is None else qk_dim)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if self.qk_dim % self.num_heads != 0:
            raise ValueError(
                f"qk_dim={self.qk_dim} must be divisible by num_heads={num_heads}"
            )
        if logit_scale_init <= 0.0 or logit_scale_max <= 0.0:
            raise ValueError("logit_scale_init and logit_scale_max must be positive")

        self.v_head_dim = self.dim // self.num_heads
        self.qk_head_dim = self.qk_dim // self.num_heads
        self.num_summary_tokens = int(num_summary_tokens)

        self.q_proj = _make_linear(dim, self.qk_dim)
        self.k_proj = _make_linear(dim, self.qk_dim)
        self.v_proj = _make_linear(dim, dim)
        self.out_proj = _make_linear(dim, dim)
        self.logit_scale = nn.Parameter(
            torch.full((num_heads,), math.log(logit_scale_init))
        )
        self.logit_scale_max = float(math.log(logit_scale_max))

    def forward(
        self, x: torch.Tensor, rope: Optional[nn.Module] = None
    ) -> torch.Tensor:
        batch, num_tokens, _ = x.shape
        q_in = x
        k_in = x
        v_in = x
        q = (
            self.q_proj(q_in)
            .view(batch, num_tokens, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(k_in)
            .view(batch, num_tokens, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(v_in)
            .view(batch, num_tokens, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
        )

        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        if rope is not None:
            q, k = rope.apply_to_qk(q, k, num_summary_tokens=self.num_summary_tokens)

        q_attn = q * self.logit_scale.float().clamp(
            max=self.logit_scale_max
        ).exp().view(1, -1, 1, 1)
        out = F.scaled_dot_product_attention(
            q_attn, k, v.float(), dropout_p=0.0, scale=1.0
        )
        out = out.to(dtype=x.dtype)
        out = out.transpose(1, 2).contiguous().view(batch, num_tokens, self.dim)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Token cosine cross-attention; per-level K/V are shared by the head."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        *,
        qk_dim: Optional[int] = None,
        logit_scale_init: float = 10.0,
        logit_scale_max: float = 100.0,
        k_proj: nn.Module,
        v_proj: nn.Module,
        kv_norm: nn.Module,
    ):
        super().__init__()
        self.q_dim = int(q_dim)
        self.kv_dim = int(kv_dim)
        self.num_heads = int(num_heads)
        self.qk_dim = int(q_dim if qk_dim is None else qk_dim)
        if self.q_dim % self.num_heads != 0:
            raise ValueError(
                f"q_dim={q_dim} must be divisible by num_heads={num_heads}"
            )
        if self.qk_dim % self.num_heads != 0:
            raise ValueError(
                f"qk_dim={self.qk_dim} must be divisible by num_heads={num_heads}"
            )
        if logit_scale_init <= 0.0 or logit_scale_max <= 0.0:
            raise ValueError("logit_scale_init and logit_scale_max must be positive")

        self.v_head_dim = self.q_dim // self.num_heads
        self.qk_head_dim = self.qk_dim // self.num_heads

        # Pre-norm on the Q side, owned by this block. Defaults to non-affine.
        self.q_norm = RMSNorm(self.q_dim, elementwise_affine=False)

        # Pre-norm on the KV side and the K/V projections are externally
        # provided by this feature-level head.
        self.kv_norm = kv_norm
        self.k_proj = k_proj
        self.v_proj = v_proj

        # Spectrally normalized query projection.
        self.q_proj = _make_linear(self.q_dim, self.qk_dim)

        # Nonzero projection plus zero residual gate avoids a dead branch.
        # SN here is a deliberate correction to match the paper.
        self.out_proj = _make_linear(self.q_dim, self.q_dim)

        # Per-head bounded cosine-attention scale, same convention as self-attn.
        self.logit_scale = nn.Parameter(
            torch.full((self.num_heads,), math.log(logit_scale_init))
        )
        self.logit_scale_max = float(math.log(logit_scale_max))

    def forward(
        self,
        x_q: torch.Tensor,
        t_kv: torch.Tensor,
        kv_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_q:     [B, Nq, q_dim]
        t_kv:    [B, Nkv, kv_dim]
        kv_mask: [B, Nkv] bool  (True = keep, False = padded / dropped)
        """
        if kv_mask.shape[:2] != t_kv.shape[:2]:
            raise ValueError(
                f"kv_mask {tuple(kv_mask.shape)} does not align with t_kv "
                f"{tuple(t_kv.shape[:2])}"
            )

        batch, n_q, _ = x_q.shape
        n_kv = t_kv.shape[1]

        xn = self.q_norm(x_q)
        tn = self.kv_norm(t_kv)

        q = (
            self.q_proj(xn)
            .view(batch, n_q, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(tn)
            .view(batch, n_kv, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(tn)
            .view(batch, n_kv, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
        )

        # Cosine normalize Q, K in fp32.
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        v = v.float()

        scale = (
            self.logit_scale.float()
            .clamp(max=self.logit_scale_max)
            .exp()
            .view(1, -1, 1, 1)
        )
        q_attn = q * scale

        # Additive attention bias: 0 where kept, -1e4 where masked. -1e4 is
        # safe under fp16/bf16 if ever cast down, and strongly suppresses the
        # masked positions after softmax.
        neg = torch.tensor(-1e4, dtype=torch.float32, device=x_q.device)
        zero = torch.zeros((), dtype=torch.float32, device=x_q.device)
        attn_bias = torch.where(kv_mask.view(batch, 1, 1, n_kv), zero, neg)

        out = F.scaled_dot_product_attention(
            q_attn, k, v, attn_mask=attn_bias, dropout_p=0.0, scale=1.0
        )
        out = out.to(dtype=x_q.dtype)
        out = out.transpose(1, 2).contiguous().view(batch, n_q, self.q_dim)
        return self.out_proj(out)
