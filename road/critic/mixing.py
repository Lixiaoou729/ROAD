"""Frozen CCM/CSM: average 1/3/5 convolutions and independent CLS linears."""

import torch
from torch import nn

from .token_spatial import patch_map_to_tokens, patch_tokens_to_map


class TokenMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.summary_proj = nn.Linear(dim, dim)
        self.branches = nn.ModuleList(
            [nn.Conv2d(dim, dim, k, padding=k // 2) for k in (1, 3, 5)]
        )

    def forward(self, tokens):
        cls, patches = tokens[:, :1], tokens[:, 1:]
        spatial = patch_tokens_to_map(patches)
        patches = patch_map_to_tokens(
            torch.stack([m(spatial) for m in self.branches]).mean(0)
        )
        return torch.cat([self.summary_proj(cls), patches], dim=1)


class RandomMixing(nn.Module):
    def __init__(self, dim=768, levels=4, dummy_registers=4):
        super().__init__()
        self.ccm = nn.ModuleList([TokenMixer(dim) for _ in range(levels)])
        self.csm = nn.ModuleList([TokenMixer(dim) for _ in range(levels)])
        self.dummy_registers = dummy_registers
        self.register_buffer("last_refresh_step", torch.tensor(0, dtype=torch.int64))
        self.requires_grad_(False)

    def forward(self, features):
        if len(features) != len(self.ccm):
            raise ValueError("Incorrect number of DINO feature levels")
        # Historical TokenCSMProjector and paper Eq. (mixing) both apply
        # the affine CSM to the SUM. Moving the skip outside would change the
        # diagonal block, branch biases, and same-level spatial filtering.
        # See docs/behavior_changes.md for the exact source references.
        mixed = [m(x) for m, x in zip(self.ccm, features)]
        outputs = [None] * len(mixed)
        outputs[-1] = self.csm[-1](mixed[-1])
        for i in range(len(mixed) - 2, -1, -1):
            outputs[i] = self.csm[i](outputs[i + 1] + mixed[i])
        # Head convention: patches, CLS, four zero dummy registers.
        return [
            torch.cat(
                [
                    x[:, 1:],
                    x[:, :1],
                    x.new_zeros(x.shape[0], self.dummy_registers, x.shape[2]),
                ],
                1,
            )
            for x in outputs
        ]

    @torch.no_grad()
    def refresh(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer.reset_parameters()
        self.requires_grad_(False)


def refresh_if_due(mixing, completed_steps, interval):
    """Called AFTER an optimizer step, before checkpointing/next step.

    A saved completed_steps includes its refresh; resume never redraws it.
    Every rank advances its RNG; rank-zero parameters are authoritative.
    """
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("Refresh interval must be positive or null")
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be nonnegative")
    if interval is None or completed_steps == 0 or completed_steps % interval:
        return False
    if completed_steps <= int(mixing.last_refresh_step):
        return False
    mixing.refresh()
    mixing.last_refresh_step.fill_(completed_steps)
    if torch.distributed.is_initialized():
        for tensor in list(mixing.parameters()) + list(mixing.buffers()):
            torch.distributed.broadcast(tensor, src=0)
    return True
