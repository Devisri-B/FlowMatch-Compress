"""Unit tests for INT8, INT4, and FP8 Quantization Engine."""

import pytest
import torch
import torch.nn as nn
from models.dit import DiT
from quantization.quant_layers import QuantizedLinear
from quantization.ptq_engine import PTQEngine
from quantization.fp8_types import FP8Format


def test_quantized_linear_forward():
    linear = nn.Linear(32, 64)
    x = torch.randn(4, 32)
    y_float = linear(x)

    for mode in ["int8", "int4", "fp8"]:
        q_layer = QuantizedLinear.from_float(linear, quant_mode=mode)
        y_q = q_layer(x)
        assert y_q.shape == (4, 64)
        cos_sim = torch.nn.functional.cosine_similarity(y_float, y_q, dim=-1).mean()
        assert cos_sim > 0.90, f"Precision drop too high in mode {mode}"


def test_ptq_engine_model_replacement():
    model = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=32, depth=2, num_heads=2)
    ptq = PTQEngine(quant_mode="fp8")
    quant_model = ptq.quantize_model(model)

    # Verify linear modules were replaced
    has_quant_layer = any(isinstance(m, QuantizedLinear) for m in quant_model.modules())
    assert has_quant_layer

    # Check forward pass
    x = torch.randn(1, 3, 16, 16)
    t = torch.tensor([0.5])
    out = quant_model(x, t)
    assert out.shape == (1, 3, 16, 16)


def test_model_size_reduction():
    model = DiT(img_size=16, patch_size=4, in_channels=3, hidden_dim=64, depth=4, num_heads=4)
    fp32_size = PTQEngine.compute_model_size_mb(model)

    ptq_int4 = PTQEngine(quant_mode="int4")
    quant_int4 = ptq_int4.quantize_model(model)
    int4_size = PTQEngine.compute_model_size_mb(quant_int4)

    assert int4_size["size_mb"] < fp32_size["size_mb"]
