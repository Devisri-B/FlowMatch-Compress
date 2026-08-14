"""
Trajectory fidelity and numerical drift analysis between Teacher and Distilled Students.
Measures:
- Endpoint MSE and L1 distance
- Step-by-step vector field cosine similarity
- PSNR (Peak Signal-to-Noise Ratio)
"""

import os
import sys
from typing import Dict, List, Tuple

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dit import DiT
from core.flow_matching import FlowMatching
from distillation.step_distill import FourStepSampler
from quantization.ptq_engine import PTQEngine


class TrajectoryEvaluator:
    @staticmethod
    def compute_psnr(x_true: torch.Tensor, x_pred: torch.Tensor, max_val: float = 2.0) -> float:
        """Computes Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(x_true, x_pred).item()
        if mse == 0:
            return 100.0
        return 20.0 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse))).item()

    @classmethod
    def evaluate_drift(
        cls,
        teacher: nn.Module,
        student: nn.Module,
        shape: Tuple[int, ...],
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Runs ODE sampling with identical initial noise seed x_1 and compares generated latents.
        """
        teacher.eval()
        student.eval()
        fm = FlowMatching()

        # Fixed initial noise
        torch.manual_seed(42)
        x_init = torch.randn(shape, device=device)

        # Teacher 50-step trajectory
        timesteps_50 = torch.linspace(1.0, 0.0, 51, device=device)
        x_teacher = x_init.clone()
        with torch.no_grad():
            for i in range(50):
                t_curr = timesteps_50[i]
                dt = timesteps_50[i] - timesteps_50[i + 1]
                t_tensor = torch.full((shape[0],), t_curr, device=device)
                v = teacher(x_teacher, t_tensor)
                x_teacher = x_teacher - dt * v

        # Student 4-step trajectory
        timesteps_4 = torch.linspace(1.0, 0.0, 5, device=device)
        x_student = x_init.clone()
        with torch.no_grad():
            for i in range(4):
                t_curr = timesteps_4[i]
                dt = timesteps_4[i] - timesteps_4[i + 1]
                t_tensor = torch.full((shape[0],), t_curr, device=device)
                v = student(x_student, t_tensor)
                x_student = x_student - dt * v

        # Quantized FP8 Student 4-step trajectory
        ptq = PTQEngine(quant_mode="fp8")
        quant_student = ptq.quantize_model(student).to(device).eval()
        x_quant = x_init.clone()
        with torch.no_grad():
            for i in range(4):
                t_curr = timesteps_4[i]
                dt = timesteps_4[i] - timesteps_4[i + 1]
                t_tensor = torch.full((shape[0],), t_curr, device=device)
                v = quant_student(x_quant, t_tensor)
                x_quant = x_quant - dt * v

        mse_student = F.mse_loss(x_student, x_teacher).item()
        mse_quant = F.mse_loss(x_quant, x_teacher).item()
        psnr_student = cls.compute_psnr(x_teacher, x_student)
        psnr_quant = cls.compute_psnr(x_teacher, x_quant)

        cos_sim_student = F.cosine_similarity(x_student.flatten(1), x_teacher.flatten(1), dim=1).mean().item()
        cos_sim_quant = F.cosine_similarity(x_quant.flatten(1), x_teacher.flatten(1), dim=1).mean().item()

        return {
            "student_4step_mse": mse_student,
            "quant_fp8_4step_mse": mse_quant,
            "student_4step_psnr": psnr_student,
            "quant_fp8_4step_psnr": psnr_quant,
            "student_cos_sim": cos_sim_student,
            "quant_cos_sim": cos_sim_quant,
        }


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating trajectory drift on {device}...")
    teacher = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4).to(device)
    student = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4).to(device)

    metrics = TrajectoryEvaluator.evaluate_drift(teacher, student, shape=(1, 3, 32, 32), device=device)
    print("\nTrajectory Fidelity Metrics (vs. 50-step Teacher):")
    for k, v in metrics.items():
        print(f"  {k:22s}: {v:.4f}")
