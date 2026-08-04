"""
Diffusion Transformer (DiT) architecture with AdaLN-Zero conditioning.
Paper: "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023).
Supports:
- Patchify / Depatchify for 2D visual latents
- Adaptive Layer Normalization with Zero-initialization (AdaLN-Zero)
- Multi-Head Self-Attention with scaled dot-product attention
- Continuous timestep, class conditioning, and guidance scale inputs
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TimestepEmbedding, LabelEmbedding, GuidanceEmbedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulates normalized tensor with affine parameters (scale and shift)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention using PyTorch scaled dot-product attention."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    """
    DiT Block with AdaLN-Zero modulation.
    Produces 6 modulation parameters per block: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

        # 6 affine modulation chunks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim)
            c: (B, hidden_dim) conditioning vector
        """
        mod_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_params.chunk(6, dim=1)

        # Self-Attention with AdaLN modulation & gating
        norm_x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(norm_x1)

        # MLP with AdaLN modulation & gating
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x2)

        return x


class FinalLayer(nn.Module):
    """Final AdaLN layer followed by linear projection to patch dimensions."""
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """
    Diffusion Transformer Backbone for Continuous Flow Matching.
    Predicts velocity field v_theta(x_t, t, c).
    """
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
        enable_guidance_cond: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_dim = hidden_dim

        # Input patch projection
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)

        # Conditioning embedders
        self.t_embedder = TimestepEmbedding(hidden_dim)
        self.y_embedder = LabelEmbedding(num_classes, hidden_dim, dropout_prob=class_dropout_prob)
        self.enable_guidance_cond = enable_guidance_cond
        if enable_guidance_cond:
            self.w_embedder = GuidanceEmbedding(hidden_dim)

        # 2D Positional Embeddings (learnable)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        # DiT Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final prediction head
        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize patch embed like a linear layer
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize pos_embed with normal
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize timestep and conditioning MLPs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Crucial: Initialize AdaLN-Zero gates to zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize final layer to zero
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_size**2 * C)
        returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) noisy input sample at time t
            t: (B,) continuous timestep in [0, 1]
            y: (B,) class labels (optional)
            w: (B,) guidance scale (optional, used in CFG distillation)
            force_drop_ids: (B,) boolean mask for null conditioning
        Returns:
            v_t: (B, C, H, W) predicted velocity field dx/dt
        """
        # Patchify + Positional Embedding
        x = self.x_embedder(x) + self.pos_embed

        # Condition Vector: c = t_emb + y_emb (+ w_emb)
        c = self.t_embedder(t)
        if y is not None:
            c = c + self.y_embedder(y, force_drop_ids=force_drop_ids)
        if self.enable_guidance_cond and w is not None:
            c = c + self.w_embedder(w)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final projection and unpatchify
        x = self.final_layer(x, c)
        v = self.unpatchify(x)
        return v
