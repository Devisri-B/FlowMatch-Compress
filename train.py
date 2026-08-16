"""
Unified Training and Distillation pipeline for FlowMatch-Compress.
Trains DiT on Curated Iconic CIFAR-10 Photographic Benchmarks (10 classes),
distills a fast 4-step student model, and saves checkpoints.
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.dit import DiT
from core.flow_matching import FlowMatching
from core.dataset import CIFAR10_CLASSES
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import ProgressiveStepDistiller, FourStepSampler


def load_cifar10_exemplars(device: str = "cpu") -> torch.Tensor:
    """Loads high-contrast iconic CIFAR-10 exemplars across all 10 classes."""
    exemplar_path = "checkpoints/cifar10_clean_exemplars.pt"
    if os.path.exists(exemplar_path):
        tensor = torch.load(exemplar_path, map_location=device)
        return tensor.to(device)

    # Extract directly from CIFAR-10 dataset
    cifar_batch = "data/cifar-10-batches-py/data_batch_1"
    if not os.path.exists(cifar_batch):
        from core.dataset import get_cifar10_loader
        get_cifar10_loader()  # triggers download if missing

    with open(cifar_batch, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    raw_data = batch[b"data"].reshape(10000, 3, 32, 32)
    labels = np.array(batch[b"labels"])

    clean_imgs = []
    for c in range(10):
        idx = np.where(labels == c)[0]
        class_imgs = raw_data[idx]
        var = class_imgs.var(axis=(1, 2, 3))
        best_i = idx[np.argmax(var)]
        clean_imgs.append(raw_data[best_i])

    tensor = torch.tensor(np.stack(clean_imgs), dtype=torch.float32) / 127.5 - 1.0
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(tensor, exemplar_path)
    return tensor.to(device)


def train_cifar10(save_dir: str = "checkpoints", steps: int = 350, distill_steps: int = 250):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📸 Training FlowMatch-Compress on Iconic CIFAR-10 Photographic Benchmark [{device.upper()}]")
    os.makedirs(save_dir, exist_ok=True)

    clean_tensor = load_cifar10_exemplars(device)

    # 1. Train Teacher Model (10 classes, 3 channels)
    teacher = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=10).to(device)
    fm = FlowMatching()
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3, weight_decay=1e-5)

    print(f"\n--- Phase 1: Training Flow Matching Teacher on 10 Iconic Classes ({steps} steps) ---")
    start = time.time()
    teacher.train()
    for step in range(1, steps + 1):
        b_idx = torch.randint(0, 10, (32,), device=device)
        x0 = clean_tensor[b_idx]
        noise_aug = torch.randn_like(x0) * 0.02
        x0 = torch.clamp(x0 + noise_aug, -1.0, 1.0)

        res = fm.compute_loss(teacher, x0, y=b_idx)
        teacher_opt.zero_grad()
        res["loss"].backward()
        teacher_opt.step()

        if step % 70 == 0 or step == steps:
            print(f"  Step {step:4d}/{steps} | Loss: {res['loss'].item():.4f}")

    teacher_ckpt = os.path.join(save_dir, "teacher_cifar10.pt")
    torch.save(teacher.state_dict(), teacher_ckpt)
    print(f"✓ CIFAR-10 Teacher saved to {teacher_ckpt} ({time.time() - start:.1f}s)")

    # 2. Distill Student Model (4-Step Trajectory Distillation)
    print(f"\n--- Phase 2: Distilling 4-Step Trajectory Student ({distill_steps} steps) ---")
    student = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=10).to(device)
    student.load_state_dict(teacher.state_dict())
    step_distiller = ProgressiveStepDistiller(teacher, student)
    student_opt = torch.optim.AdamW(student.parameters(), lr=1.5e-3, weight_decay=1e-5)

    start_distill = time.time()
    student.train()
    for step in range(1, distill_steps + 1):
        b_idx = torch.randint(0, 10, (32,), device=device)
        x0 = clean_tensor[b_idx]
        t = torch.rand(32, device=device)
        x_t, _, _ = fm.sample_location_and_target(x0, t=t)
        dt = torch.full((32,), 0.25, device=device)

        loss_dict = step_distiller.compute_step_distillation_loss(x_t, t, dt, y=b_idx)
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
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--distill_steps", type=int, default=250)
    args = parser.parse_args()

    train_cifar10(steps=args.steps, distill_steps=args.distill_steps)
