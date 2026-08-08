"""
Timestep schedulers for Flow Matching integration.
Includes:
- Linear schedule
- Cosine schedule
- Resolution-dependent time shift schedule (Flux / SD3 style)
"""

import math
import torch


class TimeScheduler:
    """Provides discretized integration timesteps for Flow Matching ODE solvers."""
    @staticmethod
    def linear_schedule(num_steps: int, device: str = "cpu") -> torch.Tensor:
        """Standard uniform linear spacing from 1.0 to 0.0."""
        return torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    @staticmethod
    def shifted_schedule(num_steps: int, shift: float = 3.0, device: str = "cpu") -> torch.Tensor:
        """
        Non-linear time shifting used in SD3 and Flux.
        Concentrates integration steps where the vector field curvature is highest.
        t_shifted = shift * t / (1 + (shift - 1) * t)
        """
        linear = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        shifted = (shift * linear) / (1.0 + (shift - 1.0) * linear)
        return shifted

    @staticmethod
    def cosine_schedule(num_steps: int, device: str = "cpu") -> torch.Tensor:
        """Cosine schedule concentrating more steps near t=0 (fine detail generation)."""
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        cosine = torch.cos(steps * math.pi * 0.5)
        return cosine
