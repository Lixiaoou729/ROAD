import itertools
from pathlib import Path

import pytest
import torch
from torch import nn

from road import checkpoints
from road.config import load
from road.critic.mixing import RandomMixing, refresh_if_due
from road.critic.regularization import lazy_scale, penalty
from road.critic.transformer import TransformerCritic, update_spectral_norm
from road.data import StepBatches
from road.vjp import staged_backward

torch.set_num_threads(1)


def test_pooled_and_jr1_distinct_and_full_cls():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    # Two patch logits and CLS. Mean derivative cancels but Jacobian does not.
    w = torch.tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]], requires_grad=True)
    logits = x @ w.T
    assert penalty(logits, x, "pooled_r1").item() == 0
    estimates = [
        penalty(logits, x, "jr1", probe=torch.tensor([p]))
        for p in itertools.product((-1.0, 1.0), repeat=3)
    ]
    expected = w.square().sum() / 3
    torch.testing.assert_close(torch.stack(estimates).mean(), expected)
    torch.stack(estimates).mean().backward()
    torch.testing.assert_close(w.grad, 2 * w / 3)


@pytest.mark.parametrize("kind", ["jr1", "pooled_r1"])
def test_accumulated_r1_matches_global_batch(kind):
    weights = nn.Parameter(torch.tensor([[0.5, -0.3], [0.7, 0.2]]))
    images = torch.randn(4, 2)
    probe = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    x = images.clone().requires_grad_()
    p = penalty(x @ weights.T, x, kind, probe=probe)
    (p * 0.0075 * 16 / 2).backward()
    expected = weights.grad.clone()
    weights.grad = None
    for i in range(4):
        x = images[i : i + 1].clone().requires_grad_()
        (
            penalty(x @ weights.T, x, kind, probe=probe[i : i + 1])
            * 0.0075
            * 16
            / 2
            / 4
        ).backward()
    torch.testing.assert_close(weights.grad, expected)


def test_lazy_schedule_and_disabled():
    assert [i + 1 for i in range(33) if lazy_scale(0.0075, 16, i)] == [16, 32]
    assert lazy_scale(0.0075, 16, 15) == 0.0075 * 8
    assert all(lazy_scale(0, 16, i) == 0 for i in range(32))


def test_critic_gate_and_cls_and_double_backward():
    torch.manual_seed(7)
    c = TransformerCritic(
        "sdxl", dim=24, heads=3, depths=(2, 1), checkpoint_blocks=True
    )
    update_spectral_norm(c)
    features = [torch.randn(2, 9, 24, requires_grad=True) for _ in range(2)]
    text = torch.randn(2, 6, 2048)
    logits = c(features, text)
    assert logits.shape == (2, 10)  # each head: four patches + CLS, no dummy logits
    logits.square().mean().backward(retain_graph=True)
    gate = c.heads[0].blocks[0].gate_ca
    assert gate.grad.abs().sum() > 0
    cross = c.heads[0].blocks[0].cross_attn
    assert torch.count_nonzero(cross.out_proj.weight_orig) > 0
    assert hasattr(cross.out_proj, "weight_u")
    with torch.no_grad():
        gate.add_(gate.grad * 0.01)
    c.zero_grad()
    logits = c(features, text)
    gradients = torch.autograd.grad(logits.sum(), features, create_graph=True)
    sum(g.square().sum() for g in gradients).backward()
    assert cross.out_proj.weight_orig.grad is not None
    assert torch.isfinite(cross.out_proj.weight_orig.grad).all()


def test_sd35_active_adapter():
    c = TransformerCritic(
        "sd35m", dim=24, heads=3, depths=(1,), checkpoint_blocks=False
    )
    update_spectral_norm(c)
    c([torch.randn(1, 9, 24)], torch.randn(1, 5, 4096)).sum().backward()
    assert isinstance(c.adapter[0], nn.LayerNorm)
    assert c.heads[0].blocks[0].gate_ca.grad.abs().sum() > 0


def test_refresh_changes_only_mixing_and_schedule():
    m = RandomMixing(dim=6, levels=2)
    c = nn.Linear(6, 1)
    opt = torch.optim.AdamW(c.parameters())
    before = {k: v.clone() for k, v in m.state_dict().items()}
    head = {k: v.clone() for k, v in c.state_dict().items()}
    assert not refresh_if_due(m, 5000, None)
    assert not refresh_if_due(m, 0, 5000)
    assert not refresh_if_due(m, 4999, 5000)
    assert refresh_if_due(m, 5000, 5000)
    assert any(not torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, c.state_dict()[k]) for k, v in head.items())
    assert not opt.state
    assert not any(p.requires_grad for p in m.parameters())
    x = [torch.randn(1, 5, 6, requires_grad=True) for _ in range(2)]
    outputs = m(x)
    assert outputs[0].shape == (1, 9, 6)
    assert torch.count_nonzero(outputs[0][:, -4:]) == 0
    sum(y.sum() for y in outputs).backward()
    assert x[0].grad.abs().sum() > 0


def test_staged_vjp_matches_direct_and_rng():
    torch.manual_seed(6)
    g = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.2))
    decoder = nn.Sequential(nn.Linear(4, 5), nn.Dropout(0.1))
    decoder.requires_grad_(False)
    noise = torch.randn(2, 3)
    rng = torch.get_rng_state()
    loss = decoder(g(noise)).square().mean() / 3
    loss.backward()
    grads = [p.grad.clone() for p in g.parameters()]
    end = torch.get_rng_state()
    g.zero_grad()
    torch.set_rng_state(rng)
    staged_backward(g, decoder, lambda x: x.square().mean(), noise, 1 / 3)
    for p, v in zip(g.parameters(), grads):
        torch.testing.assert_close(p.grad, v)
    assert torch.equal(torch.get_rng_state(), end)


def test_data_order_distributed_resume():
    s = StepBatches(103, 16, 10)
    for step in [0, 5, 6, 100]:
        batches = [s.indices(step, r, 2, 2, 4) for r in range(4)]
        assert len(set(sum(batches, []))) == 16
        fresh = StepBatches(103, 16, 10)
        assert fresh.indices(step, 2, 2, 2, 4) == batches[2]


def toy_step(g, c, m, og, od, step):
    x = torch.randn(2, 3)
    og.zero_grad()
    od.zero_grad()
    loss = c(g(x)).square().mean()
    loss.backward()
    og.step()
    od.step()
    refresh_if_due(m, step + 1, 2)


def test_checkpoint_resume_update_rng_and_refresh(tmp_path):
    torch.manual_seed(10)
    g = nn.Linear(3, 3)
    c = nn.Linear(3, 1)
    m = RandomMixing(3, 2)
    og = torch.optim.AdamW(g.parameters())
    od = torch.optim.AdamW(c.parameters())
    for step in range(2):
        toy_step(g, c, m, og, od, step)
    checkpoints.save(tmp_path / "state", g, c, m, og, od, 2, {"config": "test"})
    toy_step(g, c, m, og, od, 2)
    expected = {k: v.clone() for k, v in g.state_dict().items()}
    mix = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state()
    step = checkpoints.load(tmp_path / "state", g, c, m, og, od, {"config": "test"})
    assert step == 2
    toy_step(g, c, m, og, od, step)
    for k, v in expected.items():
        torch.testing.assert_close(g.state_dict()[k], v, rtol=0, atol=0)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in mix.items())
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError):
        checkpoints.load(tmp_path / "state", g, c, m, og, od, {"config": "wrong"})


@pytest.mark.parametrize(
    "family,batch,gamma", [("sdxl", 64, 0.0075), ("sd35m", 32, 0.00075)]
)
def test_final_configs(family, batch, gamma):
    c = load(Path(__file__).parents[1] / "configs" / f"{family}.yaml")
    assert c.training.global_batch == batch
    assert c.regularization.gamma == gamma


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: setattr(c.mixing, "refresh_interval", 0),
        lambda c: setattr(c.training, "accumulation", 1),
        lambda c: setattr(c.regularization, "type", "hybrid"),
        lambda c: setattr(c.regularization, "gamma", float("nan")),
    ],
)
def test_config_rejects_wrong_semantics(mutation):
    c = load(Path(__file__).parents[1] / "configs" / "sdxl.yaml")
    mutation(c)
    with pytest.raises(ValueError):
        c.validate()
