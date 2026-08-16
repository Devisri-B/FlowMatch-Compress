import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

try:
    import huggingface_hub
    huggingface_hub.logging.set_verbosity_error()
except Exception:
    pass

"""
Latent Autoencoder Engine for High-Resolution Latent Flow Matching.
Wraps stabilityai/sd-vae-ft-mse to encode 256x256 RGB images into 4x32x32 continuous latents
and decode latents back to photorealistic RGB images.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    AutoencoderKL = None
    DIFFUSERS_AVAILABLE = False


class VAEEngine(nn.Module):
    """
    Manages VAE latent space encoding and decoding (SD-VAE standard).
    Latent scaling factor: 0.18215 (Standard Latent Diffusion scaling).
    """
    SCALE_FACTOR = 0.18215

    def __init__(self, pretrained_model: str = "stabilityai/sd-vae-ft-mse", device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "The 'diffusers' library is required for high-resolution VAE decoding. "
                "Please run: pip install diffusers"
            )
        self.vae = AutoencoderKL.from_pretrained(pretrained_model, torch_dtype=torch.float32).to(self.device).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: (B, 3, H, W) normalized to [-1, 1]
        Returns:
            latents: (B, 4, H/8, W/8)
        """
        img_tensor = img_tensor.to(self.device)
        posterior = self.vae.encode(img_tensor).latent_dist
        latents = posterior.sample() * self.SCALE_FACTOR
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, 4, H/8, W/8)
        Returns:
            img_tensor: (B, 3, H, W) clamped to [-1, 1]
        """
        latents = latents.to(self.device) / self.SCALE_FACTOR
        imgs = self.vae.decode(latents).sample
        return torch.clamp(imgs, -1.0, 1.0)

    @staticmethod
    def latents_to_rgb(imgs_tensor: torch.Tensor) -> torch.Tensor:
        """Converts [-1, 1] tensor to [0, 1] display range."""
        return torch.clamp((imgs_tensor + 1.0) / 2.0, 0.0, 1.0)
