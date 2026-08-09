"""
Classifier-Free Guidance (CFG) Distillation.
Paper: "On Distillation of Guided Diffusion Models" (Salimans & Ho, 2022).

Standard CFG requires 2 model evaluations per ODE step:
    v_cfg = v_uncond + w * (v_cond - v_uncond)
CFG Distillation trains a student model to predict v_cfg directly in a single forward pass,
conditioned on the guidance scale w.
Result: Immediate 2x inference speedup without compromising prompt alignment.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFGDistiller:
    """
    Distills dual-pass CFG teacher into single-pass student.
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        min_guidance: float = 1.0,
        max_guidance: float = 8.0
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.min_guidance = min_guidance
        self.max_guidance = max_guidance

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def sample_guidance_scales(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Samples guidance scale w uniformly from [min_guidance, max_guidance]."""
        return torch.empty(batch_size, device=device).uniform_(self.min_guidance, self.max_guidance)

    def compute_distillation_loss(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        w: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the CFG distillation loss:
            1. Teacher computes v_cond and v_uncond.
            2. Compute target: v_target = v_uncond + w * (v_cond - v_uncond).
            3. Student predicts v_student(x_t, t, y, w) in ONE pass.
            4. Minimize || v_student - v_target ||^2.
        """
        B = x_t.shape[0]
        device = x_t.device

        if w is None:
            w = self.sample_guidance_scales(B, device)

        # 1. Dual-pass Teacher evaluation
        with torch.no_grad():
            v_cond = self.teacher(x_t, t, y=y)
            force_null = torch.ones(B, dtype=torch.bool, device=device)
            v_uncond = self.teacher(x_t, t, y=y, force_drop_ids=force_null)

            # Combine using CFG formula
            w_broadcast = w.view(B, *([1] * (x_t.ndim - 1)))
            v_target = v_uncond + w_broadcast * (v_cond - v_uncond)

        # 2. Single-pass Student evaluation conditioned on w
        v_student = self.student(x_t, t, y=y, w=w)

        # 3. Mean Squared Error Distillation Loss
        loss = F.mse_loss(v_student, v_target)

        # Also compute cosine similarity metric for quality monitoring
        flat_pred = v_student.flatten(1)
        flat_target = v_target.flatten(1)
        cos_sim = F.cosine_similarity(flat_pred, flat_target, dim=1).mean()

        return {
            "loss": loss,
            "cosine_sim": cos_sim,
            "v_student": v_student,
            "v_target": v_target
        }

    @torch.no_grad()
    def sample_student_euler(
        self,
        shape: Tuple[int, ...],
        steps: int = 50,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.0,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Runs Euler sampling with the student model in a SINGLE pass per step!
        """
        self.student.eval()
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
        w_tensor = torch.full((shape[0],), cfg_scale, device=device)

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_curr - t_next

            t_tensor = torch.full((shape[0],), t_curr, device=device)

            # Single forward pass!
            v = self.student(x, t_tensor, y=y, w=w_tensor)
            x = x - dt * v

        return x
