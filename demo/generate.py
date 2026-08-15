"""
High-Resolution Comparison Plot Generator.
Generates a side-by-side comparison figure across 3 classes:
- Class 0: Glowing Cyan Ring
- Class 1: Crisp Red Cross
- Class 2: Neon Green Square

Comparing:
- Row 1: Teacher (50-Step Euler ODE, FP32)
- Row 2: Distilled Student (4-Step Fast Solver, FP32)
- Row 3: Quantized Student (4-Step Fast Solver, FP8)
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


def tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (C, H, W) tensor normalized in [-1, 1] to a displayable (H, W, C) in [0, 1]."""
    img = tensor.detach().cpu().float()
    img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).numpy()
    return img


def generate_comparison_plot(output_path: str = "demo/comparison_trajectories.png"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for trained checkpoints
    ckpt_teacher = "checkpoints/teacher.pt"
    ckpt_student = "checkpoints/student_distilled.pt"

    teacher = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=3).to(device)
    if os.path.exists(ckpt_teacher):
        teacher.load_state_dict(torch.load(ckpt_teacher, map_location=device))
        print("✓ Loaded trained Teacher checkpoint.")
    else:
        print("Notice: using untrained teacher weights. Run 'python3 train.py' to train.")

    student = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=3).to(device)
    if os.path.exists(ckpt_student):
        student.load_state_dict(torch.load(ckpt_student, map_location=device))
        print("✓ Loaded trained Distilled Student checkpoint.")
    else:
        student.load_state_dict(teacher.state_dict())

    teacher.eval()
    student.eval()

    # Quantize student to FP8
    ptq = PTQEngine(quant_mode="fp8")
    quant_student = ptq.quantize_model(student).to(device).eval()

    fm = FlowMatching()

    # Let's visualize the 5-step trajectory for Class 0 (Ring), Class 1 (Cross), and Class 2 (Square)
    # We will display: Noise (t=1.0) -> Step 1 -> Step 2 -> Step 3 -> Final Generated Shape (t=0.0)
    torch.manual_seed(42)
    shape = (1, 3, 32, 32)
    x_init = torch.randn(shape, device=device)
    target_class = torch.tensor([0], device=device)  # Ring

    # 1. Teacher 50-step trajectory snapshots
    print("Generating Teacher 50-step trajectory...")
    timesteps_50 = torch.linspace(1.0, 0.0, 51, device=device)
    x = x_init.clone()
    teacher_snaps = [x[0].clone()]
    snap_indices = [12, 25, 37, 50]

    with torch.no_grad():
        for i in range(50):
            t_curr = timesteps_50[i]
            dt = timesteps_50[i] - timesteps_50[i + 1]
            t_tensor = torch.full((1,), t_curr, device=device)
            v = teacher(x, t_tensor, y=target_class)
            x = x - dt * v
            if (i + 1) in snap_indices:
                teacher_snaps.append(x[0].clone())

    # 2. Student 4-step trajectory snapshots
    print("Generating Distilled 4-step trajectory...")
    _, student_snaps = FourStepSampler.sample(
        student, shape, num_steps=4, y=target_class, device=device, return_trajectory=True
    )
    student_snaps = [s[0] for s in student_snaps]

    # 3. FP8 Quantized 4-step trajectory snapshots
    print("Generating FP8 Quantized 4-step trajectory...")
    _, fp8_snaps = FourStepSampler.sample(
        quant_student, shape, num_steps=4, y=target_class, device=device, return_trajectory=True
    )
    fp8_snaps = [s[0] for s in fp8_snaps]

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
            img = tensor_to_img(all_snaps[r][c])
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
