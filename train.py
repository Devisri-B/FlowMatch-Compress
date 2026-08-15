"""
Fast Training and Distillation script for FlowMatch-Compress.
Trains DiT on geometric classes (Ring, Cross, Square) in ~15-20 seconds on Apple Silicon / CUDA,
then distills a 4-step student model and saves checkpoints.
"""

import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.dit import DiT
from core.flow_matching import FlowMatching
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import ProgressiveStepDistiller


def generate_shape_batch(b: int = 32, device: str = "cpu"):
    """
    Generates synthetic batches of 3 geometric classes:
    - Class 0: Glowing Cyan Ring
    - Class 1: Crisp Red Cross
    - Class 2: Neon Green Square
    Returns images normalized to [-1.0, 1.0].
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 32, device=device),
        torch.linspace(-1, 1, 32, device=device),
        indexing="ij"
    )
    labels = torch.randint(0, 3, (b,), device=device)
    imgs = torch.zeros(b, 3, 32, 32, device=device)

    for i in range(b):
        lbl = labels[i].item()
        if lbl == 0:  # Ring
            r = torch.sqrt(x**2 + y**2)
            mask = torch.exp(-((r - 0.5)**2) / 0.03)
            imgs[i, 0] = mask * 0.1
            imgs[i, 1] = mask * 0.8
            imgs[i, 2] = mask * 1.0
        elif lbl == 1:  # Cross
            mask = ((x.abs() < 0.2) & (y.abs() < 0.75)) | ((y.abs() < 0.2) & (x.abs() < 0.75))
            imgs[i, 0] = mask.float() * 1.0
            imgs[i, 1] = mask.float() * 0.2
            imgs[i, 2] = mask.float() * 0.2
        else:  # Square
            mask = (x.abs() < 0.6) & (y.abs() < 0.6) & ~((x.abs() < 0.35) & (y.abs() < 0.35))
            imgs[i, 0] = mask.float() * 0.2
            imgs[i, 1] = mask.float() * 1.0
            imgs[i, 2] = mask.float() * 0.4

    # Normalize to [-1, 1]
    return imgs * 2.0 - 1.0, labels


def train_pipeline(save_dir: str = "checkpoints"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training FlowMatch-Compress on hardware: {device.upper()}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Train Teacher Model
    teacher = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=3).to(device)
    fm = FlowMatching()
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3, weight_decay=1e-4)

    print("\n--- Phase 1: Training Flow Matching Teacher (350 steps) ---")
    start = time.time()
    teacher.train()
    for step in range(1, 351):
        x0, y = generate_shape_batch(32, device=device)
        res = fm.compute_loss(teacher, x0, y=y)
        teacher_opt.zero_grad()
        res["loss"].backward()
        teacher_opt.step()

        if step % 70 == 0:
            print(f"  Step {step:3d}/350 | Loss: {res['loss'].item():.4f}")

    teacher_ckpt = os.path.join(save_dir, "teacher.pt")
    torch.save(teacher.state_dict(), teacher_ckpt)
    print(f"✓ Teacher saved to {teacher_ckpt} ({time.time() - start:.1f}s)")

    # 2. Distill Student Model (4-step trajectory distillation)
    print("\n--- Phase 2: Distilling 4-Step Trajectory Student (200 steps) ---")
    student = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=3).to(device)
    # Initialize student weights from trained teacher for fast convergence
    student.load_state_dict(teacher.state_dict())
    step_distiller = ProgressiveStepDistiller(teacher, student)
    student_opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

    start_distill = time.time()
    student.train()
    for step in range(1, 201):
        x0, y = generate_shape_batch(32, device=device)
        # Sample intermediate noisy state x_t
        t = torch.rand(32, device=device)
        x_t, _, _ = fm.sample_location_and_target(x0, t=t)
        dt = torch.full((32,), 0.25, device=device)

        loss_dict = step_distiller.compute_step_distillation_loss(x_t, t, dt, y=y)
        student_opt.zero_grad()
        loss_dict["loss"].backward()
        student_opt.step()

        if step % 50 == 0:
            print(f"  Step {step:3d}/200 | Distill Loss: {loss_dict['loss'].item():.4f}")

    student_ckpt = os.path.join(save_dir, "student_distilled.pt")
    torch.save(student.state_dict(), student_ckpt)
    print(f"✓ Distilled Student saved to {student_ckpt} ({time.time() - start_distill:.1f}s)")
    print(f"\n🎉 Training & Distillation complete! Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    train_pipeline()
