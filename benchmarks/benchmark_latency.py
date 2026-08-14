"""
Comprehensive Inference Latency, Throughput, and Memory Benchmark Suite.
Compares:
1. 50-Step Teacher (Dual-pass CFG, FP32)
2. 50-Step CFG-Distilled Student (Single-pass CFG, FP32)
3. 4-Step Distilled Student (FP32)
4. 4-Step Quantized Student (FP8 / INT8)
"""

import os
import sys
import time
from typing import Dict, Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from models.dit import DiT
from core.flow_matching import FlowMatching
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import FourStepSampler
from quantization.ptq_engine import PTQEngine


class LatencyBenchmark:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.console = Console()

    def time_execution(self, fn, warmups: int = 2, runs: int = 5) -> float:
        """Measures average execution time in milliseconds."""
        for _ in range(warmups):
            fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            fn()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
        end = time.perf_counter()

        return ((end - start) / runs) * 1000.0  # in ms

    def run_full_suite(
        self,
        batch_size: int = 1,
        img_size: int = 32,
        channels: int = 3,
        hidden_dim: int = 256,
        depth: int = 6
    ) -> Dict[str, Any]:
        """Runs benchmarks across all model compression stages."""
        self.console.print("\n[bold cyan]⚡ Running FlowMatch-Compress Benchmark Suite...[/bold cyan]\n")
        shape = (batch_size, channels, img_size, img_size)

        # 1. Instantiate Teacher
        teacher = DiT(img_size=img_size, patch_size=4, in_channels=channels, hidden_dim=hidden_dim, depth=depth)
        teacher = teacher.to(self.device).eval()
        fm = FlowMatching()

        # 2. Instantiate Student
        student = DiT(img_size=img_size, patch_size=4, in_channels=channels, hidden_dim=hidden_dim, depth=depth)
        student = student.to(self.device).eval()

        # 3. Quantize to FP8 and INT4
        ptq_fp8 = PTQEngine(quant_mode="fp8")
        quant_fp8 = ptq_fp8.quantize_model(student).to(self.device).eval()

        ptq_int4 = PTQEngine(quant_mode="int4")
        quant_int4 = ptq_int4.quantize_model(student).to(self.device).eval()

        # Calculate model sizes
        teacher_size = PTQEngine.compute_model_size_mb(teacher)
        fp8_size = PTQEngine.compute_model_size_mb(quant_fp8)
        int4_size = PTQEngine.compute_model_size_mb(quant_int4)

        dummy_y = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        dummy_w = torch.full((batch_size,), 4.0, device=self.device)

        # Benchmark 1: Teacher (50 steps, CFG scale 4.0 -> 100 evaluations)
        t_50_cfg = self.time_execution(
            lambda: fm.sample_euler(teacher, shape, steps=50, y=dummy_y, cfg_scale=4.0, device=self.device)
        )

        # Benchmark 2: CFG-Distilled Student (50 steps, single pass -> 50 evaluations)
        cfg_distiller = CFGDistiller(teacher, student)
        t_50_distill = self.time_execution(
            lambda: cfg_distiller.sample_student_euler(shape, steps=50, y=dummy_y, cfg_scale=4.0, device=self.device)
        )

        # Benchmark 3: 4-Step Distilled Student (4 evaluations)
        t_4_distill = self.time_execution(
            lambda: FourStepSampler.sample(student, shape, num_steps=4, y=dummy_y, w=dummy_w, device=self.device)
        )

        # Benchmark 4: 4-Step FP8 Quantized Student (4 evaluations)
        t_4_fp8 = self.time_execution(
            lambda: FourStepSampler.sample(quant_fp8, shape, num_steps=4, y=dummy_y, w=dummy_w, device=self.device)
        )

        results = {
            "teacher_50step_cfg_ms": t_50_cfg,
            "student_50step_cfg_distill_ms": t_50_distill,
            "student_4step_ms": t_4_distill,
            "quant_fp8_4step_ms": t_4_fp8,
            "teacher_size_mb": teacher_size["size_mb"],
            "fp8_size_mb": fp8_size["size_mb"],
            "int4_size_mb": int4_size["size_mb"],
            "speedup_4step_vs_teacher": t_50_cfg / (t_4_distill + 1e-6),
            "memory_reduction_fp8_pct": (1.0 - fp8_size["size_mb"] / teacher_size["size_mb"]) * 100.0,
            "memory_reduction_int4_pct": (1.0 - int4_size["size_mb"] / teacher_size["size_mb"]) * 100.0,
        }

        # Print Rich Table
        table = Table(title="FlowMatch-Compress Performance & Compression Benchmark")
        table.add_column("Pipeline Stage", style="cyan", no_wrap=True)
        table.add_column("ODE Steps", justify="center", style="magenta")
        table.add_column("Model Evals", justify="center", style="magenta")
        table.add_column("Model Size (MB)", justify="right", style="green")
        table.add_column("Latency (ms)", justify="right", style="bold yellow")
        table.add_column("Speedup", justify="right", style="bold green")

        table.add_row(
            "1. Baseline Teacher (CFG)", "50", "100 (2x pass)", f"{teacher_size['size_mb']:.2f}", f"{t_50_cfg:.1f}", "1.00x (Ref)"
        )
        table.add_row(
            "2. CFG-Distilled Student", "50", "50 (1x pass)", f"{teacher_size['size_mb']:.2f}", f"{t_50_distill:.1f}", f"{t_50_cfg / t_50_distill:.2f}x"
        )
        table.add_row(
            "3. 4-Step Distilled Student", "4", "4", f"{teacher_size['size_mb']:.2f}", f"{t_4_distill:.1f}", f"{results['speedup_4step_vs_teacher']:.2f}x"
        )
        table.add_row(
            "4. 4-Step Quantized (FP8/INT8)", "4", "4", f"{fp8_size['size_mb']:.2f}", f"{t_4_fp8:.1f}", f"{t_50_cfg / t_4_fp8:.2f}x"
        )

        self.console.print(table)
        self.console.print(f"\n[bold green]✓ Memory Reduction:[/bold green] FP8: -{results['memory_reduction_fp8_pct']:.1f}% | INT4: -{results['memory_reduction_int4_pct']:.1f}%")
        self.console.print(f"[bold green]✓ Total Latency Speedup:[/bold green] [bold yellow]{results['speedup_4step_vs_teacher']:.1f}x[/bold yellow] faster than standard 50-step CFG baseline!\n")

        return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    bench = LatencyBenchmark(device=device)
    bench.run_full_suite()
