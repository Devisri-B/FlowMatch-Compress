"""
Step Distillation and Trajectory Compression Engine for Flow Matching.
Supports:
1. Progressive Distillation (halving ODE steps: 32 -> 16 -> 8 -> 4 -> 2)
2. Direct 4-step Consistency Trajectory Distillation
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveStepDistiller:
    """
    Halves the required ODE sampling steps by training the student to take 1 macro-step
    matching 2 mini-steps of the teacher.
    """
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, target_steps: int = 4):
        self.teacher = teacher_model
        self.student = student_model
        self.target_steps = target_steps

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def compute_step_distillation_loss(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes 2-to-1 step distillation loss:
        1. Teacher executes 2 steps of size dt / 2.
        2. Student executes 1 step of size dt.
        3. Match student's trajectory to teacher's endpoint.
        """
        B = x_t.shape[0]
        dt_b = dt.view(B, *([1] * (x_t.ndim - 1)))
        half_dt = dt_b * 0.5

        # --- Teacher 2-step simulation ---
        with torch.no_grad():
            # Step 1
            v1 = self.teacher(x_t, t, y=y, w=w)
            x_mid = x_t - half_dt * v1
            t_mid = torch.clamp(t - 0.5 * dt, min=0.0, max=1.0)

            # Step 2
            v2 = self.teacher(x_mid, t_mid, y=y, w=w)
            x_target_endpoint = x_mid - half_dt * v2

            # Effective velocity for the combined macro-step
            # x_t - dt * v_eff = x_target_endpoint  =>  v_eff = (x_t - x_target_endpoint) / dt
            v_target = (x_t - x_target_endpoint) / (dt_b + 1e-7)

        # --- Student 1-step prediction ---
        v_student = self.student(x_t, t, y=y, w=w)

        loss = F.mse_loss(v_student, v_target)
        endpoint_loss = F.mse_loss(x_t - dt_b * v_student, x_target_endpoint)

        return {
            "loss": loss + 0.5 * endpoint_loss,
            "v_loss": loss,
            "endpoint_loss": endpoint_loss,
            "v_student": v_student,
            "v_target": v_target
        }


class FourStepSampler:
    """
    Dedicated 4-step solver for distilled student models.
    Reduces total model evaluations from 50 to 4.
    """
    @staticmethod
    @torch.no_grad()
    def sample(
        student_model: nn.Module,
        shape: Tuple[int, ...],
        num_steps: int = 4,
        y: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        device: str = "cpu",
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Executes fast 4-step generation: t = [1.0, 0.75, 0.5, 0.25, 0.0]
        """
        student_model.eval()
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        trajectory = [x.clone()] if return_trajectory else None

        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_curr - t_next

            t_tensor = torch.full((shape[0],), t_curr, device=device)
            v = student_model(x, t_tensor, y=y, w=w)
            x = x - dt * v

            if return_trajectory:
                trajectory.append(x.clone())

        return (x, trajectory) if return_trajectory else (x, None)
