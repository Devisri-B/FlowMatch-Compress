"""
Interactive Streamlit Application for FlowMatch-Compress.
Visualizes generation dynamics, CFG distillation speedups, and quantization savings.
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
st.sidebar.header("Configuration & Hyperparameters")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"Hardware Device: **{device.upper()}**")

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

cfg_scale = st.sidebar.slider("Classifier-Free Guidance (CFG) Scale", min_value=1.0, max_value=8.0, value=3.5, step=0.5)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=4, value=1)
seed = st.sidebar.number_input("Random Seed", value=42)

@st.cache_resource
def load_models():
    model = DiT(img_size=32, patch_size=4, in_channels=3, hidden_dim=128, depth=4).to(device).eval()
    ptq = PTQEngine(quant_mode="fp8")
    quant_fp8 = ptq.quantize_model(model).to(device).eval()
    ptq_int4 = PTQEngine(quant_mode="int4")
    quant_int4 = ptq_int4.quantize_model(model).to(device).eval()
    return model, quant_fp8, quant_int4

teacher, quant_fp8, quant_int4 = load_models()
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

# Execution Button
if st.button("🚀 Run Flow Generation", use_container_width=True):
    torch.manual_seed(seed)
    shape = (batch_size, 3, 32, 32)
    start_time = time.perf_counter()

    with st.spinner("Integrating ODE vector field..."):
        if "Standard CFG" in pipeline_mode:
            dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)
            out, trajectory = fm.sample_euler(teacher, shape, steps=50, y=dummy_y, cfg_scale=cfg_scale, device=device, return_trajectory=True)
        elif "CFG-Distilled" in pipeline_mode:
            dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)
            distiller = CFGDistiller(teacher, teacher)
            out = distiller.sample_student_euler(shape, steps=50, y=dummy_y, cfg_scale=cfg_scale, device=device)
            trajectory = [out]
        elif "Step-Distilled" in pipeline_mode:
            dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_w = torch.full((batch_size,), cfg_scale, device=device)
            out, trajectory = FourStepSampler.sample(teacher, shape, num_steps=4, y=dummy_y, w=dummy_w, device=device, return_trajectory=True)
        else:  # Quantized
            active_model = quant_fp8 if quant_mode == "fp8" else quant_int4
            dummy_y = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_w = torch.full((batch_size,), cfg_scale, device=device)
            out, trajectory = FourStepSampler.sample(active_model, shape, num_steps=4, y=dummy_y, w=dummy_w, device=device, return_trajectory=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    st.success(f"Generation completed in **{elapsed_ms:.1f} ms**!")

    # Display results
    st.subheader("Generated Latent Visualizations")
    img_np = out[0].detach().cpu().float().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)

    col_res, col_chart = st.columns([1, 2])
    with col_res:
        st.image(img_np, caption=f"Output Sample ({pipeline_mode.split('.')[1].strip()})", width=250)
    with col_chart:
        st.markdown(f"""
        ### Performance Summary
        - **Pipeline Mode**: `{pipeline_mode}`
        - **ODE Steps Executed**: `{steps_count}`
        - **Forward Passes**: `{evals_count}`
        - **Measured Latency**: `{elapsed_ms:.1f} ms`
        - **Compute Footprint**: `{evals_count / 100.0 * 100:.1f}%` of full baseline
        """)
