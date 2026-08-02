"""
Embedding modules for Flow Matching Diffusion Transformers (DiT).
Includes:
- Continuous Timestep Fourier/Sinusoidal Embeddings
- Classifier/Conditioning Embeddings with CFG Dropout
- Guidance Scale Embeddings for CFG Distillation
"""

import math
import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """
    Continuous timestep embedding using sinusoidal projection followed by an MLP.
    Maps t in [0, 1] to a latent conditioning vector of dimension `hidden_dim`.
    """
    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (B,) or (B, 1) with values typically in [0, 1]
        Returns:
            Tensor of shape (B, hidden_dim)
        """
        if t.ndim == 2:
            t = t.squeeze(-1)
        
        half_dim = self.frequency_dim // 2
        # Scale to [0, 1000] for standard frequency response
        t_scaled = t * 1000.0
        
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t_scaled[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.frequency_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return self.mlp(embedding)


class LabelEmbedding(nn.Module):
    """
    Discrete class label embedding with Classifier-Free Guidance (CFG) null token support.
    """
    def __init__(self, num_classes: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        # +1 for unconditional null token
        self.num_classes = num_classes
        self.null_token_id = num_classes
        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor, force_drop_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            labels: (B,) integer class labels
            force_drop_ids: (B,) boolean tensor, True if label should be replaced with null token
        Returns:
            (B, hidden_dim) conditioning vectors
        """
        if self.training and self.dropout_prob > 0.0:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_mask, torch.full_like(labels, self.null_token_id), labels)
        elif force_drop_ids is not None:
            labels = torch.where(force_drop_ids, torch.full_like(labels, self.null_token_id), labels)
            
        return self.embedding(labels)


class GuidanceEmbedding(nn.Module):
    """
    Guidance scale embedding for CFG-distilled student models.
    Conditioning the student on w in [1.0, 15.0] allows a single model to output
    different levels of prompt adherence in a single forward pass.
    """
    def __init__(self, hidden_dim: int, frequency_dim: int = 128):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w: (B,) guidance scale (e.g. 1.0, 3.5, 7.5)
        Returns:
            (B, hidden_dim)
        """
        if w.ndim == 2:
            w = w.squeeze(-1)
        half_dim = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=w.device) / half_dim
        )
        args = w[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)
