"""
FP8 (8-bit Floating Point) Emulation and Numerical Specifications.
Implements:
- E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits, bias=7) - High precision forward format
- E5M2 (1 sign bit, 5 exponent bits, 2 mantissa bits, bias=15) - High dynamic range format
"""

import math
import torch


class FP8Format:
    """Mathematical properties of FP8 formats."""
    # E4M3: max norm = 448.0, min positive normal = 2^-6 = 0.015625
    E4M3_MAX = 448.0
    E4M3_MIN = 2 ** -6
    E4M3_EXP_BITS = 4
    E4M3_MANT_BITS = 3
    E4M3_BIAS = 7

    # E5M2: max norm = 57344.0, min positive normal = 2^-14
    E5M2_MAX = 57344.0
    E5M2_MIN = 2 ** -14
    E5M2_EXP_BITS = 5
    E5M2_MANT_BITS = 2
    E5M2_BIAS = 15

    @classmethod
    def quantize_e4m3(cls, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Simulates FP8-E4M3 quantization via affine scaling:
        x_scaled = clamp(x / scale, -448, 448)
        Quantizes mantissa to 3 bits (8 uniform subdivisions per power-of-two bin).
        """
        scaled = tensor / (scale + 1e-8)
        clamped = torch.clamp(scaled, -cls.E4M3_MAX, cls.E4M3_MAX)

        # Emulate discrete floating point mantissa resolution
        abs_x = clamped.abs()
        exp = torch.floor(torch.log2(torch.clamp(abs_x, min=cls.E4M3_MIN)))
        exp = torch.clamp(exp, min=-6.0, max=8.0)
        bin_width = 2.0 ** (exp - cls.E4M3_MANT_BITS)

        # Round to nearest mantissa grid
        quantized = torch.round(clamped / bin_width) * bin_width
        return quantized * scale

    @classmethod
    def compute_e4m3_scale(cls, tensor: torch.Tensor, percentile: float = 99.99) -> torch.Tensor:
        """Computes optimal scaling factor for E4M3 dynamic range."""
        abs_max = torch.quantile(tensor.abs().float(), percentile / 100.0)
        scale = abs_max / cls.E4M3_MAX
        return torch.clamp(scale, min=1e-8)
