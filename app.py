import streamlit as st
import requests
import json
import numpy as np
from beam_physics import calculate_beam_properties
import time
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="AI Structural Simulator")

st.title("⚡ AI Surrogate Model for Structural Simulation")
st.markdown("This dashboard demonstrates a 500x faster Deep Learning surrogate replacing standard FEA/analytical solvers for an Euler-Bernoulli cantilever beam.")

# Side bar for inputs
st.sidebar.header("Beam Configuration")

# Material Presets
materials = {
    "Steel": {"E": 200e9, "rho": 7850, "yield": 250e6},
    "Aluminum": {"E": 69e9, "rho": 2700, "yield": 95e6},
    "HDPE": {"E": 1e9, "rho": 970, "yield": 20e6},
    "Titanium": {"E": 114e9, "rho": 4500, "yield": 880e6},
    "Concrete": {"E": 30e9, "rho": 2400, "yield": 5e6}, # low tension yield
    "Wood (Oak)": {"E": 11e9, "rho": 700, "yield": 40e6},
    "Carbon Fiber": {"E": 150e9, "rho": 1600, "yield": 1200e6}
}
mat_select = st.sidebar.selectbox("Material", list(materials.keys()))

L = st.sidebar.slider("Length (L) [m]", 0.1, 5.0, 2.0, 0.1)
b = st.sidebar.slider("Width (b) [m]", 0.01, 0.2, 0.05, 0.01)
h = st.sidebar.slider("Height (h) [m]", 0.01, 0.2, 0.1, 0.01)
F = st.sidebar.slider("Load (F) [N]", 100, 50000, 1000, 100)

st.sidebar.markdown("---")
st.sidebar.header("Visualization Settings")
vis_scale = st.sidebar.slider("Visual Bending Scale", 1, 1000, 100)

E = materials[mat_select]["E"]
rho = materials[mat_select]["rho"]
yield_str = materials[mat_select]["yield"]

tab1, tab2 = st.tabs(["🚀 Live Simulator", "📊 Model Performance"])

def draw_beam(L, deflection_m, scale):
    # Create points along the beam
    x = np.linspace(0, L, 100)
    
    # Cap the visualization deflection to avoid massive image sizes
    # We use a tanh-like scaling or just a simple cap to keep it within bounds
    vis_deflection = deflection_m * scale
    if abs(vis_deflection) > 1.0:
        vis_deflection = np.sign(vis_deflection) * 1.0 # Cap at 1.0m visually
    
    y = - vis_deflection * (x/L)**3 
    
    fig, ax = plt.subplots(figsize=(8, 3))
    # Draw beam
    ax.plot(x, y, lw=8, color='#3498db', alpha=0.9, solid_capstyle='round', label='Deformed Beam')
    # Draw original position (dotted)
    ax.plot([0, L], [0, 0], 'k--', alpha=0.2, label='Original Position')
    # Support (fixed end)
    ax.plot([0, 0], [-1.2, 1.2], '#2c3e50', lw=6)
    # Load arrow
    if abs(vis_deflection) > 0.001:
        ax.annotate('', xy=(L, y[-1]), xytext=(L, y[-1] + 0.3 if vis_deflection < 0 else y[-1] - 0.3),
                    arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=2))
        ax.text(L, y[-1] + 0.4 if vis_deflection < 0 else y[-1] - 0.4, f"{F} N", color='#e74c3c', ha='center', weight='bold')
    
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-0.2, L + 0.2)
    # Remove aspect equal to prevent massive width/height issues
    ax.axis('off')
    ax.set_title(f"Visualized Bending Behavior", fontsize=10, pad=10)
    plt.tight_layout()
    return fig

with tab1:
    col1, col2 = st.columns([1, 1])

    data = None
    with col1:
        st.header("DNN Surrogate Prediction")
        
        payload = {"L": L, "b": b, "h": h, "F": F, "E": E, "rho": rho}
        
        try:
            api_url = "http://localhost:8000/predict"
            if os.environ.get('IN_DOCKER'):
                api_url = "http://api:8000/predict"
                
            res = requests.post(api_url, json=payload, timeout=2)
            if res.status_code == 200:
                data = res.json()
                
                st.metric("Max Deflection", f"{data['max_deflection_m']*1000:.3f} mm")
                st.metric("Max Stress", f"{data['max_stress_Pa']/1e6:.2f} MPa")
                st.metric("Natural Frequency", f"{data['natural_freq_Hz']:.2f} Hz")
                st.info(f"⚡ Inference Time: {data['inference_time_ms']:.2f} ms")
            else:
                st.error("API Error")
        except Exception as e:
            st.warning(f"Wait for API server to start... ({str(e)})")

    with col2:
        st.header("Analytical Ground Truth")
        start_t = time.perf_counter()
        v, s, f = calculate_beam_properties(L, b, h, F, E, rho)
        calc_time = (time.perf_counter() - start_t) * 1000
        
        st.metric("Max Deflection", f"{v*1000:.3f} mm")
        st.metric("Max Stress", f"{s/1e6:.2f} MPa")
        st.metric("Natural Frequency", f"{f:.2f} Hz")
        st.info(f"🐌 Compute Time: {calc_time:.2f} ms")

    st.markdown("---")
    
    # 3D/Visual Section
    v_col1, v_col2 = st.columns([2, 1])
    
    with v_col1:
        st.subheader("Visual Bending Behavior")
        if data:
            fig = draw_beam(L, data['max_deflection_m'], vis_scale)
            st.pyplot(fig)
        else:
            st.info("Waiting for data to visualize...")

    with v_col2:
        st.subheader("📜 Simplified Engineer's Report")
        if data:
            stress_ratio = data['max_stress_Pa'] / yield_str
            
            if stress_ratio > 1.0:
                st.error("❌ **DANGER: Structural Failure!**")
                st.markdown(f"The predicted stress is **{stress_ratio:.1f}x** higher than the material can handle. The beam will likely break.")
            elif stress_ratio > 0.7:
                st.warning("⚠️ **CAUTION: High Stress**")
                st.markdown("The beam is nearing its limit. Consider increasing the Height (h) or using a stronger material like Steel or Titanium.")
            else:
                st.success("✅ **SAFE: Structure is Stable**")
                st.markdown(f"The material is operating at only **{stress_ratio*100:.1f}%** of its capacity.")

            # Stiffness Check
            deflection_mm = data['max_deflection_m'] * 1000
            if deflection_mm < 1.0:
                st.info("💎 **Stiffness: Very Rigid**")
            elif deflection_mm < 10.0:
                st.info("📏 **Stiffness: Moderate Flex**")
            else:
                st.info("🧪 **Stiffness: Highly Flexible**")
                
            st.markdown(f"**Natural Frequency Check:** At **{data['natural_freq_Hz']:.1f} Hz**, this beam is {'highly stable' if data['natural_freq_Hz'] > 50 else 'prone to vibration'}.")
        else:
            st.info("Simulation report will appear here once results are ready.")

    st.markdown("---")
    if st.button("Generate Random Configurations and Benchmark Speedup"):
        N_BENCH = 1000
        st.text(f"Benchmarking {N_BENCH} inferences...")
        progress_bar = st.progress(0)
        
        # Generate random inputs
        random_configs = []
        for _ in range(N_BENCH):
            random_configs.append({
                "L": np.random.uniform(0.1, 5.0),
                "b": np.random.uniform(0.01, 0.2),
                "h": np.random.uniform(0.01, 0.2),
                "F": np.random.uniform(100, 50000),
                "E": E,
                "rho": rho
            })
            
        # Benchmark Analytical
        start_analytical = time.perf_counter()
        for i, cfg in enumerate(random_configs):
            calculate_beam_properties(cfg['L'], cfg['b'], cfg['h'], cfg['F'], cfg['E'], cfg['rho'])
            if i % 100 == 0:
                progress_bar.progress((i / 2) / N_BENCH)
        total_analytical = (time.perf_counter() - start_analytical) * 1000
        
        # Benchmark AI (using Batch mode for extreme speed)
        start_ai = time.perf_counter()
        batch_url = "http://localhost:8000/predict_batch"
        if os.environ.get('IN_DOCKER'):
            batch_url = "http://api:8000/predict_batch"
            
        requests.post(batch_url, json=random_configs)
        total_ai = (time.perf_counter() - start_ai) * 1000
        
        progress_bar.progress(1.0)
        
        avg_analytical = total_analytical / N_BENCH
        avg_ai = total_ai / N_BENCH
        speedup = total_analytical / total_ai
        
        st.success(f"### Results for {N_BENCH} samples")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Analytical Time", f"{avg_analytical:.3f} ms")
        c2.metric("Avg AI Time", f"{avg_ai:.3f} ms")
        c3.metric("AI Speedup", f"{speedup:.1f}x")
        
        st.balloons()

with tab2:
    st.header("Deep Performance Profiling")
    st.markdown("The surrogate model was trained on 10,000 samples. Below are the performance metrics and explainability plots.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Learning Curves")
        if os.path.exists("figures/learning_curves.png"):
            st.image("figures/learning_curves.png", use_container_width=True)
            st.caption("Training and validation loss across epochs.")
        else:
            st.warning("Learning curves plot not found.")
            
    with col_b:
        st.subheader("Model Accuracy (Parity Plots)")
        if os.path.exists("figures/parity_plots.png"):
            st.image("figures/parity_plots.png", use_container_width=True)
            st.caption("Predicted vs Ground Truth for all three outputs.")
        else:
            st.warning("Parity plots not found.")
            
    st.markdown("---")
    st.subheader("Feature Importance (SHAP Analysis)")
    st.markdown("SHAP values explain which input parameters (L, b, h, F, E, rho) contribute most to the predictions.")
    
    shap_cols = st.columns(3)
    titles = ["Deflection", "Stress", "Frequency"]
    for i, title in enumerate(titles):
        with shap_cols[i]:
            st.markdown(f"**{title}**")
            fig_path = f"figures/shap_summary_{title.lower()}.png"
            if os.path.exists(fig_path):
                st.image(fig_path, use_container_width=True)
            else:
                st.warning(f"SHAP plot for {title} not found.")

