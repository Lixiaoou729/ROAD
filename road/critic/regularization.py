"""The only two ROAD gradient penalties; all logits include patch AND CLS."""

import math

import torch


def penalty(logits, images, kind="jr1", generator=None, probe=None):
    if logits.ndim != 2 or logits.shape[0] != images.shape[0] or logits.shape[1] < 1:
        raise ValueError("Expected a nonempty [batch, full_logits] tensor")
    if kind == "pooled_r1":
        score = logits.float().mean(1)
    elif kind == "jr1":
        if probe is None:
            probe = (
                torch.randint(
                    0, 2, logits.shape, device=logits.device, generator=generator
                )
                .mul(2)
                .sub(1)
            )
        if probe.shape != logits.shape:
            raise ValueError("Probe must cover all patch and CLS logits")
        score = (logits.float() * probe / math.sqrt(logits.shape[1])).sum(1)
    else:
        raise ValueError("regularization.type must be jr1 or pooled_r1")
    grad = torch.autograd.grad(
        score.sum(), images, create_graph=True, retain_graph=True
    )[0]
    return grad.float().flatten(1).square().sum(1).mean()


def lazy_scale(gamma, interval, completed_steps):
    # Events after each interval completed updates: interval-1, 2*interval-1...
    return (
        gamma * interval / 2 if gamma and (completed_steps + 1) % interval == 0 else 0.0
    )
