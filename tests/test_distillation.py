"""Unit tests for CFG and Step Distillation."""

import pytest
import torch
from models.dit import DiT
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import ProgressiveStepDistiller, FourStepSampler


def test_cfg_distillation_loss():
    teacher = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    student = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    distiller = CFGDistiller(teacher, student)

    x_t = torch.randn(2, 3, 16, 16)
    t = torch.tensor([0.3, 0.7])
    y = torch.tensor([0, 1])

    loss_dict = distiller.compute_distillation_loss(x_t, t, y)
    assert "loss" in loss_dict
    assert "cosine_sim" in loss_dict
    assert loss_dict["loss"].ndim == 0
    assert not torch.isnan(loss_dict["loss"])


def test_progressive_step_distillation():
    teacher = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    student = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    distiller = ProgressiveStepDistiller(teacher, student)

    x_t = torch.randn(2, 3, 16, 16)
    t = torch.tensor([0.8, 0.4])
    dt = torch.tensor([0.2, 0.2])

    loss_dict = distiller.compute_step_distillation_loss(x_t, t, dt)
    assert "loss" in loss_dict
    assert "endpoint_loss" in loss_dict


def test_four_step_sampler():
    student = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    sample, traj = FourStepSampler.sample(student, shape=(1, 3, 16, 16), num_steps=4, return_trajectory=True)
    assert sample.shape == (1, 3, 16, 16)
    assert len(traj) == 5  # Initial noise + 4 steps
