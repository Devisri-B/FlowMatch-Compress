"""
Unified Training and Distillation pipeline for FlowMatch-Compress.
Supports:
1. Real-World CIFAR-10 Photographic Dataset (10 classes: Airplane, Automobile, Bird, Cat, etc.)
2. Geometric Shapes dataset for quick debugging
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.dit import DiT
from core.flow_matching import FlowMatching
from core.dataset import get_cifar10_loader, CIFAR10_CLASSES
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import ProgressiveStepDistiller


def train_cifar10(save_dir: str = "checkpoints", steps: int = 500, distill_steps: int = 250):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📸 Training FlowMatch-Compress on Real-World CIFAR-10 Photos [{device.upper()}]")
    os.makedirs(save_dir, exist_ok=True)

    loader = get_cifar10_loader(batch_size=64, shuffle=True)
    loader_iter = iter(loader)

    def get_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    # 1. Train Teacher Model (10 classes, 3 channels)
    teacher = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=10).to(device)
    fm = FlowMatching()
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=2e-3, weight_decay=1e-4)

    print(f"\n--- Phase 1: Training Flow Matching Teacher on Real Photos ({steps} steps) ---")
    start = time.time()
    teacher.train()
    for step in range(1, steps + 1):
        x0, y = get_batch()
        x0, y = x0.to(device), y.to(device)
        res = fm.compute_loss(teacher, x0, y=y)
        teacher_opt.zero_grad()
        res["loss"].backward()
        teacher_opt.step()

        if step % 100 == 0 or step == steps:
            print(f"  Step {step:4d}/{steps} | Loss: {res['loss'].item():.4f}")

    teacher_ckpt = os.path.join(save_dir, "teacher_cifar10.pt")
    torch.save(teacher.state_dict(), teacher_ckpt)
    print(f"✓ CIFAR-10 Teacher saved to {teacher_ckpt} ({time.time() - start:.1f}s)")

    # 2. Distill Student Model (4-Step Trajectory Distillation)
    print(f"\n--- Phase 2: Distilling 4-Step Trajectory Student on Real Photos ({distill_steps} steps) ---")
    student = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=10).to(device)
    student.load_state_dict(teacher.state_dict())
    step_distiller = ProgressiveStepDistiller(teacher, student)
    student_opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

    start_distill = time.time()
    student.train()
    for step in range(1, distill_steps + 1):
        x0, y = get_batch()
        x0, y = x0.to(device), y.to(device)
        t = torch.rand(x0.shape[0], device=device)
        x_t, _, _ = fm.sample_location_and_target(x0, t=t)
        dt = torch.full((x0.shape[0],), 0.25, device=device)

        loss_dict = step_distiller.compute_step_distillation_loss(x_t, t, dt, y=y)
        student_opt.zero_grad()
        loss_dict["loss"].backward()
        student_opt.step()

        if step % 50 == 0 or step == distill_steps:
            print(f"  Step {step:4d}/{distill_steps} | Distill Loss: {loss_dict['loss'].item():.4f}")

    student_ckpt = os.path.join(save_dir, "student_distilled_cifar10.pt")
    torch.save(student.state_dict(), student_ckpt)
    print(f"✓ Distilled 4-Step Student saved to {student_ckpt} ({time.time() - start_distill:.1f}s)")
    print(f"\n🎉 CIFAR-10 Training & Distillation complete! Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "shapes"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--distill_steps", type=int, default=250)
    args = parser.parse_args()

    if args.dataset == "cifar10":
        train_cifar10(steps=args.steps, distill_steps=args.distill_steps)
