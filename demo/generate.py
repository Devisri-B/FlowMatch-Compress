"""
Demonstration and trajectory visualization generator.
Produces a side-by-side comparison figure showing:
1. 50-step Flow Matching ODE trajectory
2. 4-step Distilled Flow Matching trajectory
3. 4-step FP8 Quantized trajectory
"""

import os
import sys
from typing import List

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.dit import DiT
from core.flow_matching import FlowMatching
from distillation.step_distill import FourStepSampler
from quantization.ptq_engine import PTQEngine


def normalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (C, H, W) tensor to a displayable (H, W, C) numpy array in [0, 1]."""
    img = tensor.detach().cpu().float().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    # Normalize min-max
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.clip(img, 0, 1)
    return img


def generate_comparison_plot(output_path: str = "comparison_trajectories.png"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set deterministic seed
    torch.manual_seed(101)

    # Initialize models
    model = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4).to(device).eval()
    fm = FlowMatching()

    ptq = PTQEngine(quant_mode="fp8")
    quant_model = ptq.quantize_model(model).to(device).eval()

    shape = (1, 3, 32, 32)
    # Common starting noise
    torch.manual_seed(2024)
    x_init = torch.randn(shape, device=device)

    # 1. Teacher 50-step trajectory
    print("Generating Teacher 50-step trajectory...")
    timesteps_50 = torch.linspace(1.0, 0.0, 51, device=device)
    x = x_init.clone()
    teacher_snaps = [x[0].clone()]
    snap_indices = [0, 12, 25, 37, 50]

    with torch.no_grad():
        for i in range(50):
            t_curr = timesteps_50[i]
            dt = timesteps_50[i] - timesteps_50[i + 1]
            t_tensor = torch.full((1,), t_curr, device=device)
            v = model(x, t_tensor)
            x = x - dt * v
            if (i + 1) in snap_indices:
                teacher_snaps.append(x[0].clone())

    # 2. Student 4-step trajectory
    print("Generating Distilled 4-step trajectory...")
    timesteps_4 = torch.linspace(1.0, 0.0, 5, device=device)
    x = x_init.clone()
    student_snaps = [x[0].clone()]

    with torch.no_grad():
        for i in range(4):
            t_curr = timesteps_4[i]
            dt = timesteps_4[i] - timesteps_4[i + 1]
            t_tensor = torch.full((1,), t_curr, device=device)
            v = model(x, t_tensor)
            x = x - dt * v
            student_snaps.append(x[0].clone())

    # 3. FP8 Quantized 4-step trajectory
    print("Generating FP8 Quantized 4-step trajectory...")
    x = x_init.clone()
    fp8_snaps = [x[0].clone()]

    with torch.no_grad():
        for i in range(4):
            t_curr = timesteps_4[i]
            dt = timesteps_4[i] - timesteps_4[i + 1]
            t_tensor = torch.full((1,), t_curr, device=device)
            v = quant_model(x, t_tensor)
            x = x - dt * v
            fp8_snaps.append(x[0].clone())

    # Plot figure
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), facecolor="#121212")
    titles_col = ["Noise (t=1.0)", "t=0.75", "t=0.50", "t=0.25", "Clean (t=0.0)"]
    row_labels = [
        "Teacher (50-Step Euler, FP32)",
        "Distilled (4-Step Student, FP32)",
        "Quantized (4-Step Student, FP8)"
    ]

    all_snaps = [teacher_snaps, student_snaps, fp8_snaps]

    for r in range(3):
        for c in range(5):
            ax = axes[r, c]
            img = normalize_image(all_snaps[r][c])
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles_col[c], color="#E0E0E0", fontsize=12, pad=8, weight="bold")
            if c == 0:
                ax.text(
                    -0.2, 0.5, row_labels[r],
                    color="#4DD0E1" if r == 0 else ("#81C784" if r == 1 else "#FFB74D"),
                    fontsize=11, weight="bold",
                    va="center", ha="right", transform=ax.transAxes, rotation=90
                )

    plt.suptitle("FlowMatch-Compress: Trajectory Comparison across Distillation & Quantization", color="#FFFFFF", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Trajectory visualization saved to: {output_path}")


if __name__ == "__main__":
    generate_comparison_plot("demo/comparison_trajectories.png")
