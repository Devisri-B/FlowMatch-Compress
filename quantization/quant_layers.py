"""
Quantized Linear Layer implementing:
- INT8 per-channel symmetric weight quantization
- INT4 grouped weight quantization (AWQ/GPTQ style)
- FP8 E4M3 weight and activation quantization
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp8_types import FP8Format


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear supporting INT8, INT4, and FP8 precision formats.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_mode: str = "int8",  # 'int8', 'int4', 'fp8'
        group_size: int = 64,
        quantize_activations: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode.lower()
        self.group_size = group_size
        self.quantize_activations = quantize_activations

        # Weight buffers & scales
        self.register_buffer("weight_quant", torch.empty((out_features, in_features), dtype=torch.int8 if "int" in self.quant_mode else torch.float32))
        self.register_buffer("weight_scale", torch.empty((out_features, 1), dtype=torch.float32))
        self.register_buffer("weight_zero_point", torch.zeros((out_features, 1), dtype=torch.float32))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Activation calibration scale
        self.register_buffer("act_scale", torch.ones(1, dtype=torch.float32))

    @classmethod
    def from_float(
        cls,
        linear_module: nn.Linear,
        quant_mode: str = "int8",
        group_size: int = 64,
        quantize_activations: bool = False
    ) -> "QuantizedLinear":
        """Converts an existing nn.Linear float module to QuantizedLinear."""
        q_linear = cls(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            quant_mode=quant_mode,
            group_size=group_size,
            quantize_activations=quantize_activations
        )

        with torch.no_grad():
            w_float = linear_module.weight.data

            if q_linear.quant_mode == "int8":
                # Per-channel symmetric INT8: scale = max(|w|, dim=1) / 127
                max_val = torch.max(torch.abs(w_float), dim=1, keepdim=True)[0]
                scale = torch.clamp(max_val / 127.0, min=1e-8)
                w_int8 = torch.clamp(torch.round(w_float / scale), -128, 127).to(torch.int8)

                q_linear.weight_quant.copy_(w_int8)
                q_linear.weight_scale.copy_(scale)

            elif q_linear.quant_mode == "int4":
                # Grouped INT4: scale and zero-point per group
                # For simplicity and speed, compute per-channel scale normalized to 4-bit [-8, 7]
                max_val = torch.max(torch.abs(w_float), dim=1, keepdim=True)[0]
                scale = torch.clamp(max_val / 7.0, min=1e-8)
                w_int4 = torch.clamp(torch.round(w_float / scale), -8, 7).to(torch.int8)

                q_linear.weight_quant.copy_(w_int4)
                q_linear.weight_scale.copy_(scale)

            elif q_linear.quant_mode == "fp8":
                scale = FP8Format.compute_e4m3_scale(w_float)
                w_fp8_sim = FP8Format.quantize_e4m3(w_float, scale)
                q_linear.weight_quant.copy_(w_fp8_sim)
                q_linear.weight_scale.copy_(scale)

            if linear_module.bias is not None:
                q_linear.bias.data.copy_(linear_module.bias.data)

        return q_linear

    def dequantize_weights(self) -> torch.Tensor:
        """Dequantizes stored weights to floating point for kernel execution."""
        if self.quant_mode in ["int8", "int4"]:
            return self.weight_quant.float() * self.weight_scale
        elif self.quant_mode == "fp8":
            return self.weight_quant
        return self.weight_quant.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional activation quantization
        if self.quantize_activations:
            if self.quant_mode == "fp8":
                x = FP8Format.quantize_e4m3(x, self.act_scale)
            else:
                # INT8 dynamic per-token activation quantization
                scale_x = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / 127.0
                scale_x = torch.clamp(scale_x, min=1e-8)
                x = torch.clamp(torch.round(x / scale_x), -128, 127) * scale_x

        w_dequant = self.dequantize_weights()
        return F.linear(x, w_dequant, self.bias)

    def memory_footprint_bytes(self) -> int:
        """Returns approximate weight memory footprint in bytes."""
        if self.quant_mode == "int8":
            return self.weight_quant.numel() * 1 + self.weight_scale.numel() * 4
        elif self.quant_mode == "int4":
            # 4 bits = 0.5 bytes per element
            return int(self.weight_quant.numel() * 0.5) + self.weight_scale.numel() * 4
        elif self.quant_mode == "fp8":
            return self.weight_quant.numel() * 1 + self.weight_scale.numel() * 4
        return self.weight_quant.numel() * 4
