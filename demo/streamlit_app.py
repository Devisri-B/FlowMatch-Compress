"""
Interactive Streamlit Application for FlowMatch-Compress.
Visualizes step-by-step generative denoising, CFG distillation speedups, and quantization savings.
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
from distillation.cfg_distill import CFGDistiller
from distillation.step_distill import FourStepSampler
from quantization.ptq_engine import PTQEngine


st.set_page_config(page_title="FlowMatch-Compress Engine", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stMetric { background-color: #1E222D; padding: 12px; border-radius: 8px; border-left: 4px solid #4DD0E1; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ FlowMatch-Compress")
st.subheader("Flow Matching DiT with Step/CFG Distillation & FP8/INT4 Quantization")

# Sidebar Controls
st.sidebar.header("Generation Settings")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Hardware Acceleration: **{device.upper()}**")

class_choice = st.sidebar.selectbox(
    "Target Shape (Class Conditioning)",
    ["Glowing Cyan Ring (Class 0)", "Crisp Red Cross (Class 1)", "Neon Green Square (Class 2)"]
)
class_id = int(class_choice.split("Class ")[1].replace(")", ""))

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

@st.cache_resource
def load_models():
    teacher = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=3).to(device)
    student = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4, num_classes=3).to(device)

    ckpt_teacher = "checkpoints/teacher.pt"
    ckpt_student = "checkpoints/student_distilled.pt"

    if os.path.exists(ckpt_teacher):
        teacher.load_state_dict(torch.load(ckpt_teacher, map_location=device))
    if os.path.exists(ckpt_student):
        student.load_state_dict(torch.load(ckpt_student, map_location=device))
    else:
        student.load_state_dict(teacher.state_dict())

    teacher.eval()
    student.eval()

    ptq_fp8 = PTQEngine(quant_mode="fp8")
    quant_fp8 = ptq_fp8.quantize_model(student).to(device).eval()

    ptq_int4 = PTQEngine(quant_mode="int4")
    quant_int4 = ptq_int4.quantize_model(student).to(device).eval()

    return teacher, student, quant_fp8, quant_int4

teacher, student, quant_fp8, quant_int4 = load_models()
fm = FlowMatching()

# Calculate memory sizes
teacher_size = PTQEngine.compute_model_size_mb(teacher)
fp8_size = PTQEngine.compute_model_size_mb(quant_fp8)
int4_size = PTQEngine.compute_model_size_mb(quant_int4)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Teacher Model Size", f"{teacher_size['size_mb']:.2f} MB")
with col2:
    st.metric("FP8 Model Size", f"{fp8_size['size_mb']:.2f} MB", delta=f"-{(1-fp8_size['size_mb']/teacher_size['size_mb'])*100:.1f}%")
with col3:
    st.metric("INT4 Model Size", f"{int4_size['size_mb']:.2f} MB", delta=f"-{(1-int4_size['size_mb']/teacher_size['size_mb'])*100:.1f}%")
with col4:
    steps_count = 50 if ("50-Step" in pipeline_mode) else 4
    evals_count = 100 if "Standard CFG" in pipeline_mode else (50 if "Single Pass" in pipeline_mode else 4)
    st.metric("Active Model Evaluations", f"{evals_count} passes", delta=f"{100/evals_count:.1f}x speedup" if evals_count < 100 else None)

st.markdown("---")

def tensor_to_display(t: torch.Tensor):
    img = t.detach().cpu().float()
    img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).numpy()
    return img

# Execution Button
if st.button("🚀 Run Flow Generation", use_container_width=True):
    torch.manual_seed(seed)
    shape = (1, 3, 32, 32)
    y_target = torch.tensor([class_id], device=device)
    start_time = time.perf_counter()

    with st.spinner("Integrating ODE vector field..."):
        if "Standard CFG" in pipeline_mode:
            out, trajectory = fm.sample_euler(teacher, shape, steps=50, y=y_target, cfg_scale=cfg_scale, device=device, return_trajectory=True)
            # Pick 5 evenly spaced snaps
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
    st.success(f"Generation completed in **{elapsed_ms:.1f} ms**!")

    # Display results
    st.subheader(f"Generated Visual Output: {class_choice}")
    img_display = tensor_to_display(out[0])

    col_res, col_chart = st.columns([1, 2])
    with col_res:
        st.image(img_display, caption=f"Generated {class_choice.split('(')[0].strip()}", width=260)
    with col_chart:
        st.markdown(f"""
        ### Performance Breakdown
        - **Target Object**: `{class_choice}`
        - **Pipeline Mode**: `{pipeline_mode}`
        - **ODE Steps Executed**: `{steps_count}`
        - **Forward Passes**: `{evals_count}`
        - **Measured Latency**: `{elapsed_ms:.1f} ms`
        - **Efficiency**: Running at **{100 / evals_count:.1f}x fewer compute evaluations** than standard 50-step CFG!
        """)

    # Trajectory Progression Row
    if len(snaps) >= 5:
        st.subheader("Denoising Trajectory Progression (t = 1.0 → t = 0.0)")
        t_cols = st.columns(5)
        titles = ["1. Noise (t=1.0)", "2. Coarse Flow (t=0.75)", "3. Intermediate (t=0.50)", "4. Refinement (t=0.25)", "5. Final Object (t=0.0)"]
        for idx in range(5):
            with t_cols[idx]:
                st.image(tensor_to_display(snaps[idx][0]), caption=titles[idx], use_container_width=True)
