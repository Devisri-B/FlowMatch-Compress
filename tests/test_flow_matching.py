"""Unit tests for Continuous Flow Matching (OT-CFM) Engine."""

import pytest
import torch
from models.dit import DiT
from core.flow_matching import FlowMatching


def test_flow_matching_interpolation():
    fm = FlowMatching()
    x_0 = torch.zeros(2, 3, 16, 16)
    x_1 = torch.ones(2, 3, 16, 16)
    t = torch.tensor([0.25, 0.75])

    x_t, t_out, u_t = fm.sample_location_and_target(x_0, x_1, t=t)

    # Analytical checks
    assert torch.allclose(u_t, torch.ones_like(u_t))  # x_1 - x_0 = 1
    assert torch.allclose(x_t[0], torch.full_like(x_t[0], 0.25))
    assert torch.allclose(x_t[1], torch.full_like(x_t[1], 0.75))


def test_flow_matching_loss_computation():
    fm = FlowMatching()
    model = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    x_0 = torch.randn(2, 3, 16, 16)

    loss_dict = fm.compute_loss(model, x_0)
    assert "loss" in loss_dict
    assert loss_dict["loss"].ndim == 0
    assert loss_dict["loss"].item() > 0


def test_euler_and_heun_sampling():
    fm = FlowMatching()
    model = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)

    sample_euler = fm.sample_euler(model, shape=(1, 3, 16, 16), steps=4)
    assert sample_euler.shape == (1, 3, 16, 16)

    sample_heun = fm.sample_heun(model, shape=(1, 3, 16, 16), steps=4)
    assert sample_heun.shape == (1, 3, 16, 16)
