"""
Continuous-Time Flow Matching (OT-CFM / Rectified Flow) Engine.
References:
- "Flow Matching for Generative Modeling" (Lipman et al., ICLR 2023)
- "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (Liu et al., ICLR 2023)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatching:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).
    Interpolation: x_t = (1 - t) * x_0 + t * x_1
    where x_0 is clean data, x_1 ~ N(0, I) is standard normal prior noise.
    Target vector field: u_t(x_t | x_0, x_1) = x_1 - x_0.
    Reverse generation: integrate ODE from t = 1 (noise) down to t = 0 (clean data).
    """
    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def sample_location_and_target(
        self,
        x_0: torch.Tensor,
        x_1: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples t ~ Uniform(0, 1) and computes x_t and conditional velocity target u_t.
        Args:
            x_0: (B, C, H, W) clean data batch
            x_1: (B, C, H, W) noise sample (sampled from N(0, I) if None)
            t: (B,) optional specified timesteps
        Returns:
            x_t: (B, C, H, W) intermediate noisy sample
            t: (B,) timesteps in [0, 1]
            u_t: (B, C, H, W) target velocity field (x_1 - x_0)
        """
        B = x_0.shape[0]
        device = x_0.device

        if x_1 is None:
            x_1 = torch.randn_like(x_0)

        if t is None:
            # Low-discrepancy uniform sampling
            t = torch.rand(B, device=device)

        # Broadcast t to (B, 1, 1, 1)
        t_b = t.view(B, *([1] * (x_0.ndim - 1)))

        # Linear OT Path
        x_t = (1.0 - t_b) * x_0 + t_b * x_1

        # Analytical time-derivative: d/dt [(1-t)*x0 + t*x1] = x1 - x0
        u_t = x_1 - x_0

        return x_t, t, u_t

    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        x_1: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the Flow Matching regression loss: E || v_theta(x_t, t) - (x_1 - x_0) ||^2.
        """
        x_t, t, u_t = self.sample_location_and_target(x_0, x_1)
        v_pred = model(x_t, t, y=y)
        loss = F.mse_loss(v_pred, u_t)
        return {"loss": loss, "v_pred": v_pred, "u_target": u_t}

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        steps: int = 50,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Euler ODE integration from t = 1.0 down to t = 0.0.
        Formula: x_{t - dt} = x_t - dt * v_theta(x_t, t)
        """
        model.eval()
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
        trajectory = [x.clone()] if return_trajectory else []

        for i in range(steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_curr - t_next  # dt > 0

            t_tensor = torch.full((shape[0],), t_curr, device=device)

            if cfg_scale > 1.0 and y is not None:
                # Dual-pass Classifier-Free Guidance
                v_cond = model(x, t_tensor, y=y)
                force_null = torch.ones(shape[0], dtype=torch.bool, device=device)
                v_uncond = model(x, t_tensor, y=y, force_drop_ids=force_null)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t_tensor, y=y)

            # Step backward from t=1 to t=0
            x = x - dt * v

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_heun(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        steps: int = 25,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        device: Union[str, torch.device] = "cpu"
    ) -> torch.Tensor:
        """
        2nd-order Runge-Kutta / Heun's predictor-corrector ODE solver.
        Provides higher accuracy per step than Euler.
        """
        model.eval()
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        def eval_v(x_in: torch.Tensor, t_val: float) -> torch.Tensor:
            t_tensor = torch.full((shape[0],), t_val, device=device)
            if cfg_scale > 1.0 and y is not None:
                v_cond = model(x_in, t_tensor, y=y)
                force_null = torch.ones(shape[0], dtype=torch.bool, device=device)
                v_uncond = model(x_in, t_tensor, y=y, force_drop_ids=force_null)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return model(x_in, t_tensor, y=y)

        for i in range(steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_curr - t_next

            # Predictor step (Euler)
            d1 = eval_v(x, t_curr)
            x_pred = x - dt * d1

            # Corrector step
            if i < steps - 1:
                d2 = eval_v(x_pred, t_next)
                x = x - dt * 0.5 * (d1 + d2)
            else:
                x = x_pred

        return x
