"""
Interactive Streamlit Application for FlowMatch-Compress.
Visualizes Real-World Latent Flow Matching and CIFAR-10 photographic generation.
"""

import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import numpy as np

from models.dit import DiT
from core.flow_matching import FlowMatching
from core.vae_engine import VAEEngine
from core.dataset import CIFAR10_CLASSES
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import FourStepSampler
from quantization.ptq_engine import PTQEngine


st.set_page_config(page_title="FlowMatch-Compress Engine", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    /* Ensure all column containers respect boundary widths */
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        min-width: 0 !important;
    }
    
    /* Ensure all images stay strictly within their columns without overflow */
    div[data-testid="stImage"] {
        max-width: 100% !important;
    }
    div[data-testid="stImage"] img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
    }
    
    /* Uniform Metric Card Sizing */
    div[data-testid="stMetric"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        padding: 16px 18px !important;
        border-radius: 12px !important;
        border-left: 5px solid #0284c7 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        min-height: 125px !important;
        height: 125px !important;
        max-height: 125px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }
    
    /* Uniform Metric Label */
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        height: 24px !important;
        overflow: hidden !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        line-height: 1.2 !important;
    }
    
    /* Uniform Metric Value */
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #0f172a !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        line-height: 1.1 !important;
    }
    
    /* Uniform Metric Delta */
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        height: 22px !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        line-height: 1.2 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ FlowMatch-Compress")
st.subheader("Real-World Latent Flow Matching DiT with Step/CFG Distillation & FP8/INT4 Quantization")

# Sidebar Controls
st.sidebar.header("Generation Settings")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Hardware Acceleration: **{device.upper()}**")

dataset_mode = st.sidebar.radio(
    "Generative Paradigm",
    ["High-Resolution Latent Flow Matching (256x256)", "CIFAR-10 Photographic Benchmark (32x32)"]
)

if "High-Resolution" in dataset_mode:
    class_choice = st.sidebar.selectbox(
        "Photorealistic Subject",
        ["Sports Car on Highway 🏎️ (Class 0)", "Mountain Valley Lake 🏔️ (Class 1)", "Golden Retriever Dog 🐕 (Class 2)"]
    )
    class_id = int(class_choice.split("Class ")[1].replace(")", ""))
    in_channels = 4
    ckpt_teacher = "checkpoints/teacher_latent_hires.pt"
    ckpt_student = "checkpoints/student_distilled_latent_hires.pt"
    num_classes = 3
else:
    class_choice = st.sidebar.selectbox("CIFAR-10 Class", [f"{c} (Class {i})" for i, c in enumerate(CIFAR10_CLASSES)])
    class_id = int(class_choice.split("Class ")[1].replace(")", ""))
    in_channels = 3
    ckpt_teacher = "checkpoints/teacher_cifar10.pt"
    ckpt_student = "checkpoints/student_distilled_cifar10.pt"
    num_classes = 10

pipeline_mode = st.sidebar.selectbox(
    "Pipeline Stage",
    [
        "1. Teacher (50-Step Euler, Standard CFG)",
        "2. CFG-Distilled Student (50-Step Single Pass)",
        "3. Step-Distilled Student (4-Step Fast Solver)",
        "4. Quantized Engine (4-Step FP8 / INT4)"
    ]
)

quant_mode = "fp8"
if "Quantized" in pipeline_mode:
    quant_mode = st.sidebar.radio("Quantization Precision", ["FP8 (E4M3)", "INT4 (Grouped)", "INT8 (Per-Channel)"]).split()[0].lower()

cfg_scale = st.sidebar.slider("Classifier-Free Guidance (CFG) Scale", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
seed = st.sidebar.number_input("Random Seed", value=42)

t_mtime = os.path.getmtime(ckpt_teacher) if os.path.exists(ckpt_teacher) else 0.0
s_mtime = os.path.getmtime(ckpt_student) if os.path.exists(ckpt_student) else 0.0

@st.cache_resource
def load_fresh_models_and_vae(channels: int, classes: int, t_path: str, s_path: str, t_m: float, s_m: float):
    teacher = DiT(img_size=32, patch_size=4, in_channels=channels, hidden_dim=128, depth=4, num_classes=classes).to(device)
    student = DiT(img_size=32, patch_size=4, in_channels=channels, hidden_dim=128, depth=4, num_classes=classes).to(device)

    if os.path.exists(t_path):
        teacher.load_state_dict(torch.load(t_path, map_location=device))
        print(f"✓ [Cache-Buster] Loaded fresh Teacher from {t_path} (mtime: {t_m})")
    if os.path.exists(s_path):
        student.load_state_dict(torch.load(s_path, map_location=device))
        print(f"✓ [Cache-Buster] Loaded fresh Student from {s_path} (mtime: {s_m})")
    else:
        student.load_state_dict(teacher.state_dict())

    teacher.eval()
    student.eval()

    ptq_fp8 = PTQEngine(quant_mode="fp8")
    quant_fp8 = ptq_fp8.quantize_model(student).to(device).eval()

    ptq_int4 = PTQEngine(quant_mode="int4")
    quant_int4 = ptq_int4.quantize_model(student).to(device).eval()

    vae = VAEEngine(device=device) if channels == 4 else None

    return teacher, student, quant_fp8, quant_int4, vae

teacher, student, quant_fp8, quant_int4, vae = load_fresh_models_and_vae(in_channels, num_classes, ckpt_teacher, ckpt_student, t_mtime, s_mtime)

if st.sidebar.button("🔄 Force Reload Checkpoints from Disk"):
    st.cache_resource.clear()
    st.rerun()
fm = FlowMatching()

# Calculate memory sizes
teacher_size = PTQEngine.compute_model_size_mb(teacher)
fp8_size = PTQEngine.compute_model_size_mb(quant_fp8)
int4_size = PTQEngine.compute_model_size_mb(quant_int4)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Teacher Model Size",
        value=f"{teacher_size['size_mb']:.2f} MB",
        delta="FP32 Reference",
        delta_color="off"
    )
with col2:
    fp8_pct = (1.0 - fp8_size['size_mb'] / teacher_size['size_mb']) * 100.0
    st.metric(
        label="FP8 Model Size",
        value=f"{fp8_size['size_mb']:.2f} MB",
        delta=f"-{fp8_pct:.1f}% RAM",
        delta_color="normal"
    )
with col3:
    int4_pct = (1.0 - int4_size['size_mb'] / teacher_size['size_mb']) * 100.0
    st.metric(
        label="INT4 Model Size",
        value=f"{int4_size['size_mb']:.2f} MB",
        delta=f"-{int4_pct:.1f}% RAM",
        delta_color="normal"
    )
with col4:
    steps_count = 50 if ("50-Step" in pipeline_mode) else 4
    evals_count = 100 if "Standard CFG" in pipeline_mode else (50 if "Single Pass" in pipeline_mode else 4)
    speedup_delta = f"{100 / evals_count:.1f}x speedup" if evals_count < 100 else "1.0x Baseline"
    delta_mode = "normal" if evals_count < 100 else "off"
    st.metric(
        label="Active Model Passes",
        value=f"{evals_count} evals",
        delta=speedup_delta,
        delta_color=delta_mode
    )

st.markdown("---")

def render_tensor_to_rgb(tensor: torch.Tensor, use_vae: bool):
    if use_vae and vae is not None:
        with torch.no_grad():
            rgb = vae.decode(tensor.unsqueeze(0))
            return VAEEngine.latents_to_rgb(rgb)[0].permute(1, 2, 0).cpu().numpy()
    else:
        img = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        # High-fidelity Lanczos upsampling from 32x32 to 256x256 for clean, crisp photographic presentation
        from PIL import Image
        pil_img = Image.fromarray((img * 255.0).astype(np.uint8))
        pil_up = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
        return np.array(pil_up) / 255.0

# Execution Button
btn_label = "🚀 Run 256x256 Photorealistic Latent Flow Generation" if in_channels == 4 else "🚀 Run CIFAR-10 Photographic Flow Generation"
if st.button(btn_label, width="stretch"):
    torch.manual_seed(seed)
    shape = (1, in_channels, 32, 32)
    y_target = torch.tensor([class_id], device=device)
    start_time = time.perf_counter()

    with st.spinner("Integrating ODE vector field..."):
        if "Standard CFG" in pipeline_mode:
            out, trajectory = fm.sample_euler(teacher, shape, steps=50, y=y_target, cfg_scale=cfg_scale, device=device, return_trajectory=True)
            snaps = [trajectory[0], trajectory[12], trajectory[25], trajectory[37], trajectory[50]]
        elif "CFG-Distilled" in pipeline_mode:
            distiller = CFGDistiller(teacher, student)
            out = distiller.sample_student_euler(shape, steps=50, y=y_target, cfg_scale=cfg_scale, device=device)
            snaps = [out]
        elif "Step-Distilled" in pipeline_mode:
            dummy_w = torch.full((1,), cfg_scale, device=device)
            out, trajectory = FourStepSampler.sample(student, shape, num_steps=4, y=y_target, w=dummy_w, device=device, return_trajectory=True)
            snaps = trajectory
        else:  # Quantized
            active_model = quant_fp8 if quant_mode == "fp8" else quant_int4
            dummy_w = torch.full((1,), cfg_scale, device=device)
            out, trajectory = FourStepSampler.sample(active_model, shape, num_steps=4, y=y_target, w=dummy_w, device=device, return_trajectory=True)
            snaps = trajectory

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    st.success(f"Generated successfully in **{elapsed_ms:.1f} ms**!")

    # Display results
    st.subheader(f"Generated Visual Output: {class_choice}")
    img_display = render_tensor_to_rgb(out[0], use_vae=(in_channels == 4))

    col_res, col_chart = st.columns([1, 1], gap="large")
    with col_res:
        st.image(img_display, caption=f"Generated {class_choice.split('(')[0].strip()}", width="stretch")
    with col_chart:
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 22px 26px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <h3 style="margin-top: 0; color: #0f172a; font-size: 1.25rem; font-weight: 700; border-bottom: 2px solid #0284c7; padding-bottom: 8px;">📊 Performance Breakdown</h3>
            <ul style="list-style: none; padding-left: 0; margin-bottom: 0; line-height: 1.9; color: #334155; font-size: 0.95rem;">
                <li><strong style="color: #0f172a;">Generative Paradigm:</strong> <code style="background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px;">{dataset_mode}</code></li>
                <li><strong style="color: #0f172a;">Subject / Class:</strong> <code style="background: #f1f5f9; color: #334155; padding: 2px 6px; border-radius: 4px;">{class_choice}</code></li>
                <li><strong style="color: #0f172a;">Pipeline Mode:</strong> <code style="background: #f1f5f9; color: #334155; padding: 2px 6px; border-radius: 4px;">{pipeline_mode}</code></li>
                <li><strong style="color: #0f172a;">ODE Steps Executed:</strong> <span style="font-weight: 600; color: #0284c7;">{steps_count}</span></li>
                <li><strong style="color: #0f172a;">Forward Passes:</strong> <span style="font-weight: 600; color: #0284c7;">{evals_count}</span></li>
                <li><strong style="color: #0f172a;">Measured Latency:</strong> <span style="font-weight: 700; color: #16a34a;">{elapsed_ms:.1f} ms</span></li>
                <li style="margin-top: 8px; padding-top: 10px; border-top: 1px dashed #cbd5e1;"><strong style="color: #0f172a;">Efficiency Gain:</strong> Running at <span style="color: #0284c7; font-weight: 700;">{100 / evals_count:.1f}x fewer compute evaluations</span> than standard 50-step CFG!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Trajectory Progression Row
    if len(snaps) >= 5:
        st.subheader("Denoising Trajectory Progression (t = 1.0 → t = 0.0)")
        t_cols = st.columns(5)
        titles = ["1. Latent Noise (t=1.0)", "2. Coarse Scene (t=0.75)", "3. Structure (t=0.50)", "4. Refinement (t=0.25)", "5. Final Image (t=0.0)"]
        for idx in range(5):
            with t_cols[idx]:
                st.image(render_tensor_to_rgb(snaps[idx][0], use_vae=(in_channels == 4)), caption=titles[idx], width="stretch")
