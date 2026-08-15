"""Unit tests for Diffusion Transformer (DiT)."""

import pytest
import torch
from models.dit import DiT, PatchEmbed


def test_patch_embed():
    patch_embed = PatchEmbed(img_size=32, patch_size=4, in_chans=3, embed_dim=64)
    x = torch.randn(2, 3, 32, 32)
    tokens = patch_embed(x)
    # (32/4)^2 = 64 patches
    assert tokens.shape == (2, 64, 64)


def test_dit_forward_pass():
    model = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=64, depth=2, num_heads=4, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.2, 0.8])
    y = torch.tensor([1, 5])
    w = torch.tensor([3.5, 7.0])

    out = model(x, t, y=y, w=w)
    assert out.shape == (2, 3, 32, 32)
    assert not torch.isnan(out).any()


def test_dit_unconditional_null_token():
    model = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=64, depth=2, num_heads=4, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.5, 0.5])
    y = torch.tensor([3, 4])
    force_null = torch.tensor([True, True])

    out = model(x, t, y=y, force_drop_ids=force_null)
    assert out.shape == (2, 3, 32, 32)
