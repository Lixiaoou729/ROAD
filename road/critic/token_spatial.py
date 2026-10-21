import math

import torch


def infer_hw(num_patch_tokens: int) -> int:
    side = int(math.isqrt(max(int(num_patch_tokens), 0)))
    if side * side != int(num_patch_tokens):
        raise ValueError(
            f"Expected a square number of patch tokens, got {num_patch_tokens}"
        )
    return side


def select_levels(num_levels: int, num_heads: int):
    num_levels = int(num_levels)
    num_heads = int(num_heads)
    if num_levels <= 0 or num_heads <= 0:
        raise ValueError("num_levels and num_heads must be positive integers")
    if num_heads > num_levels:
        raise ValueError(f"num_heads={num_heads} must be <= num_levels={num_levels}")
    return [((i + 1) * num_levels) // num_heads - 1 for i in range(num_heads)]


def split_summary_patch_tokens(
    tokens: torch.Tensor,
    num_summary_tokens: int,
    *,
    summary_first: bool,
):
    num_summary_tokens = int(num_summary_tokens)
    if num_summary_tokens <= 0:
        return tokens[:, :0, :], tokens
    if tokens.shape[1] < num_summary_tokens:
        raise ValueError(
            f"num_summary_tokens={num_summary_tokens} exceeds sequence length {tokens.shape[1]}"
        )
    if summary_first:
        return tokens[:, :num_summary_tokens, :], tokens[:, num_summary_tokens:, :]
    return tokens[:, -num_summary_tokens:, :], tokens[:, :-num_summary_tokens, :]


def merge_summary_patch_tokens(
    summary_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    *,
    summary_first: bool,
) -> torch.Tensor:
    if summary_tokens.shape[1] == 0:
        return patch_tokens
    if summary_first:
        return torch.cat([summary_tokens, patch_tokens], dim=1)
    return torch.cat([patch_tokens, summary_tokens], dim=1)


def patch_tokens_to_map(patch_tokens: torch.Tensor) -> torch.Tensor:
    batch, num_patch_tokens, channels = patch_tokens.shape
    hw = infer_hw(num_patch_tokens)
    return patch_tokens.transpose(1, 2).reshape(batch, channels, hw, hw)


def patch_map_to_tokens(patch_map: torch.Tensor) -> torch.Tensor:
    return patch_map.flatten(2).transpose(1, 2).contiguous()
