"""
Post-Training Quantization (PTQ) Engine.
Manages:
- Selective layer quantization (INT8, INT4, FP8)
- Calibration with activation monitoring
- Parameter footprint and memory compression calculation
"""

import copy
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .quant_layers import QuantizedLinear


class PTQEngine:
    """
    Orchestrates Post-Training Quantization for Diffusion Transformers.
    """
    def __init__(
        self,
        quant_mode: str = "fp8",
        quantize_activations: bool = False,
        preserve_adaln: bool = True
    ):
        self.quant_mode = quant_mode.lower()
        self.quantize_activations = quantize_activations
        self.preserve_adaln = preserve_adaln

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Clones the model and selectively replaces linear layers with QuantizedLinear.
        By default, preserves AdaLN modulation and final projection in FP32/FP16 for stability.
        """
        quant_model = copy.deepcopy(model)
        self._replace_layers(quant_model)
        return quant_model

    def _replace_layers(self, module: nn.Module, parent_name: str = ""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Check if this layer should be preserved in full precision
            is_sensitive = False
            if self.preserve_adaln:
                if "adaLN" in full_name or "final_layer" in full_name or "t_embedder" in full_name:
                    is_sensitive = True

            if isinstance(child, nn.Linear) and not is_sensitive:
                q_layer = QuantizedLinear.from_float(
                    child,
                    quant_mode=self.quant_mode,
                    quantize_activations=self.quantize_activations
                )
                setattr(module, name, q_layer)
            else:
                self._replace_layers(child, full_name)

    def calibrate(self, quant_model: nn.Module, calibration_batches: List[torch.Tensor]):
        """
        Runs forward passes through calibration data to determine activation dynamic ranges.
        """
        quant_model.eval()
        activation_stats: Dict[str, List[float]] = {}

        hooks = []
        for name, module in quant_model.named_modules():
            if isinstance(module, QuantizedLinear) and self.quantize_activations:
                def make_hook(mod_name, mod):
                    def hook_fn(m, inp, out):
                        act = inp[0].detach()
                        max_act = act.abs().max().item()
                        if mod_name not in activation_stats:
                            activation_stats[mod_name] = []
                        activation_stats[mod_name].append(max_act)
                    return hook_fn
                hooks.append(module.register_forward_hook(make_hook(name, module)))

        with torch.no_grad():
            for batch in calibration_batches:
                # Mock forward pass
                t = torch.rand(batch.shape[0], device=batch.device)
                quant_model(batch, t)

        for h in hooks:
            h.remove()

        # Update activation scales
        for name, module in quant_model.named_modules():
            if name in activation_stats:
                peak = max(activation_stats[name])
                module.act_scale.fill_(max(peak, 1e-4))

    @staticmethod
    def compute_model_size_mb(model: nn.Module) -> Dict[str, float]:
        """
        Calculates total model memory size in Megabytes (MB).
        """
        total_bytes = 0
        total_params = 0

        for m in model.modules():
            if isinstance(m, QuantizedLinear):
                total_bytes += m.memory_footprint_bytes()
                total_params += m.in_features * m.out_features
            elif len(list(m.children())) == 0:  # Leaf module
                for p in m.parameters(recurse=False):
                    total_bytes += p.numel() * p.element_size()
                    total_params += p.numel()

        return {
            "total_params": total_params,
            "size_mb": total_bytes / (1024 * 1024),
            "total_bytes": total_bytes
        }
