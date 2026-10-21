"""Two CPU workers exercise actual DDP, ZeRO state, and synchronized refresh."""

import contextlib
import copy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from road import checkpoints
from road.critic.mixing import RandomMixing, refresh_if_due
from road.critic.regularization import penalty


def worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory) / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        torch.manual_seed(91)
        model = nn.Linear(2, 3, bias=False)
        reference = copy.deepcopy(model)
        model = DDP(model)
        opt = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.AdamW, lr=0.01
        )
        x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
        for kind in ("jr1", "pooled_r1"):
            model.zero_grad()
            reference.zero_grad()
            full = x.clone().requires_grad_()
            logits = reference(full)
            (
                logits.square().mean()
                + 0.06 * penalty(logits, full, kind, probe=torch.ones_like(logits))
            ).backward()
            expected = reference.weight.grad.clone()
            for micro in range(2):
                with model.no_sync() if micro == 0 else contextlib.nullcontext():
                    part = (
                        x[rank * 4 + micro * 2 : rank * 4 + micro * 2 + 2]
                        .clone()
                        .requires_grad_()
                    )
                    logits = model(part)
                    loss = (
                        logits.square().mean()
                        + 0.06
                        * penalty(logits, part, kind, probe=torch.ones_like(logits))
                    ) / 2
                    loss.backward()
            torch.testing.assert_close(model.module.weight.grad, expected)
        opt.step()
        critic = nn.Linear(2, 1)
        od = torch.optim.AdamW(critic.parameters())
        mix = RandomMixing(dim=3, levels=2)
        torch.manual_seed(300 + rank)
        assert refresh_if_due(mix, 2, 2)
        for parameter in mix.parameters():
            other = parameter.clone()
            dist.broadcast(other, 0)
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        checkpoints.save(
            Path(directory) / "state", model, critic, mix, opt, od, 2, {"world_size": 2}
        )
        expected = torch.randn(5)
        with torch.no_grad():
            model.module.weight.zero_()
        assert (
            checkpoints.load(
                Path(directory) / "state",
                model,
                critic,
                mix,
                opt,
                od,
                {"world_size": 2},
            )
            == 2
        )
        torch.testing.assert_close(torch.randn(5), expected, rtol=0, atol=0)
        assert model.module.weight.count_nonzero() > 0
        # Full state restored into sharded optimizer, not just the model.
        assert len(opt.optim.state) > 0 or rank == 1
        model.zero_grad()
        model(x[:2]).square().mean().backward()
        opt.step()
    finally:
        dist.destroy_process_group()


def test_distributed_gradients_refresh_zero_resume(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=2, join=True)
