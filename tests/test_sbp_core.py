"""Unit tests for ScoreNet, interpolation, and Euler–Maruyama sampling."""

from __future__ import annotations

import numpy as np
import torch

from core.SBP import ScoreNet, euler_maruyama_sample, interpolate_samples, score_matching_loss


def test_score_net_output_shape():
    dim = 8
    net = ScoreNet(dim)
    x = torch.randn(16, dim)
    t = torch.rand(16, 1)
    out = net(x, t)
    assert out.shape == (16, dim)


def test_score_net_accepts_1d_time():
    dim = 8
    net = ScoreNet(dim)
    x = torch.randn(4, dim)
    t = torch.rand(4)  # 1-d time vector
    out = net(x, t)
    assert out.shape == (4, dim)


def test_interpolate_samples_shape_and_bounds():
    x0 = torch.zeros(10, 8)
    x1 = torch.ones(10, 8)
    t = torch.full((10, 1), 0.5)
    xt = interpolate_samples(x0, x1, t)
    assert xt.shape == (10, 8)
    # At t=0.5 mean should be near 0.5 (noise has mean 0)
    assert abs(xt.mean().item() - 0.5) < 0.3


def test_score_matching_loss_is_scalar():
    dim = 4
    net = ScoreNet(dim)
    x = torch.randn(8, dim, requires_grad=True)
    t = torch.rand(8, 1).clamp(0.05, 0.95)
    loss = score_matching_loss(net, x, t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_euler_maruyama_trajectory_shape():
    from core.SBP import device

    dim = 4
    net = ScoreNet(dim).to(device)
    x0 = torch.randn(12, dim, device=device)
    traj = euler_maruyama_sample(x0, net, T=1.0, steps=5, sigma=1.0)
    # traj: [steps+1, batch, dim]
    assert traj.shape == (6, 12, dim)
    assert np.isfinite(traj).all()
