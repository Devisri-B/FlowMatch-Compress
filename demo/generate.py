"""
High-Resolution Photorealistic Trajectory Comparison Plot Generator.
Generates side-by-side comparison across:
- Row 1: Teacher (50-Step Euler ODE, FP32)
- Row 2: Distilled Student (4-Step Fast Solver, FP32)
- Row 3: Quantized Student (4-Step Fast Solver, FP8)

Using Continuous-Time Latent Flow Matching + VAE decoding (256x256 photorealistic images).
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
from core.vae_engine import VAEEngine
from distillation.step_distill import FourStepSampler
from quantization.ptq_engine import PTQEngine


def generate_comparison_plot(output_path: str = "demo/comparison_trajectories.png"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAEEngine(device=device)

    # Check for trained high-res latent checkpoints
    ckpt_teacher = "checkpoints/teacher_latent_hires.pt"
    ckpt_student = "checkpoints/student_distilled_latent_hires.pt"

    teacher = DiT(img_size=32, patch_size=4, in_channels=4, hidden_dim=128, depth=4, num_classes=3).to(device)
    if os.path.exists(ckpt_teacher):
        teacher.load_state_dict(torch.load(ckpt_teacher, map_location=device))
        print("✓ Loaded trained High-Res Latent Teacher checkpoint.")

    student = DiT(img_size=32, patch_size=4, in_channels=4, hidden_dim=128, depth=4, num_classes=3).to(device)
    if os.path.exists(ckpt_student):
        student.load_state_dict(torch.load(ckpt_student, map_location=device))
        print("✓ Loaded trained High-Res Latent Distilled Student checkpoint.")
    else:
        student.load_state_dict(teacher.state_dict())

    teacher.eval()
    student.eval()

    # Quantize student to FP8
    ptq = PTQEngine(quant_mode="fp8")
    quant_student = ptq.quantize_model(student).to(device).eval()

    torch.manual_seed(42)
    shape = (1, 4, 32, 32)
    x_init = torch.randn(shape, device=device)
    target_class = torch.tensor([0], device=device)  # Sports Car

    # 1. Teacher 50-step trajectory
    print("Generating Teacher 50-step trajectory in latent space...")
    timesteps_50 = torch.linspace(1.0, 0.0, 51, device=device)
    x = x_init.clone()
    teacher_snaps = [x.clone()]
    snap_indices = [12, 25, 37, 50]

    with torch.no_grad():
        for i in range(50):
            t_curr = timesteps_50[i]
            dt = timesteps_50[i] - timesteps_50[i + 1]
            t_tensor = torch.full((1,), t_curr, device=device)
            v = teacher(x, t_tensor, y=target_class)
            x = x - dt * v
            if (i + 1) in snap_indices:
                teacher_snaps.append(x.clone())

    # 2. Student 4-step trajectory
    print("Generating Distilled 4-step trajectory...")
    _, student_snaps = FourStepSampler.sample(
        student, shape, num_steps=4, y=target_class, device=device, return_trajectory=True
    )

    # 3. FP8 Quantized 4-step trajectory
    print("Generating FP8 Quantized 4-step trajectory...")
    _, fp8_snaps = FourStepSampler.sample(
        quant_student, shape, num_steps=4, y=target_class, device=device, return_trajectory=True
    )

    # Decode latents to 256x256 RGB using VAE
    print("Decoding latents to 256x256 photorealistic images via VAE...")
    def decode_snaps(snaps_list):
        imgs = []
        for s in snaps_list:
            with torch.no_grad():
                rgb = vae.decode(s)
                rgb = VAEEngine.latents_to_rgb(rgb)[0].permute(1, 2, 0).cpu().numpy()
                imgs.append(rgb)
        return imgs

    all_snaps = [
        decode_snaps(teacher_snaps),
        decode_snaps(student_snaps),
        decode_snaps(fp8_snaps)
    ]

    fig, axes = plt.subplots(3, 5, figsize=(16, 10), facecolor="#121212")
    titles_col = ["Latent Noise (t=1.0)", "Coarse Scene (t=0.75)", "Structure (t=0.50)", "Refinement (t=0.25)", "Photorealistic (t=0.0)"]
    row_labels = [
        "Teacher (50-Step Euler, FP32)",
        "Distilled (4-Step Student, FP32)",
        "Quantized (4-Step Student, FP8)"
    ]

    for r in range(3):
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(all_snaps[r][c])
            ax.axis("off")
            if r == 0:
                ax.set_title(titles_col[c], color="#E0E0E0", fontsize=12, pad=8, weight="bold")
            if c == 0:
                ax.text(
                    -0.15, 0.5, row_labels[r],
                    color="#4DD0E1" if r == 0 else ("#81C784" if r == 1 else "#FFB74D"),
                    fontsize=11, weight="bold",
                    va="center", ha="right", transform=ax.transAxes, rotation=90
                )

    plt.suptitle("FlowMatch-Compress: Real-World Latent Flow Matching Across Distillation & Quantization", color="#FFFFFF", fontsize=15, weight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Real-world trajectory visualization saved to: {output_path}")


if __name__ == "__main__":
    generate_comparison_plot("demo/comparison_trajectories.png")
