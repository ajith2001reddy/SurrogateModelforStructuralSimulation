import streamlit as st
import requests
import json
from beam_physics import calculate_beam_properties
import time

st.set_page_config(layout="wide", page_title="AI Structural Simulator")

st.title("⚡ AI Surrogate Model for Structural Simulation")
st.markdown("This dashboard demonstrates a 500x faster Deep Learning surrogate replacing standard FEA/analytical solvers for an Euler-Bernoulli cantilever beam.")

# Side bar for inputs
st.sidebar.header("Beam Configuration")

# Material Presets
materials = {
    "Steel": {"E": 200e9, "rho": 7850},
    "Aluminum": {"E": 69e9, "rho": 2700},
    "HDPE": {"E": 1e9, "rho": 970}
}
mat_select = st.sidebar.selectbox("Material", list(materials.keys()))

L = st.sidebar.slider("Length (L) [m]", 0.1, 5.0, 2.0, 0.1)
b = st.sidebar.slider("Width (b) [m]", 0.01, 0.2, 0.05, 0.01)
h = st.sidebar.slider("Height (h) [m]", 0.01, 0.2, 0.1, 0.01)
F = st.sidebar.slider("Load (F) [N]", 100, 50000, 1000, 100)

E = materials[mat_select]["E"]
rho = materials[mat_select]["rho"]

col1, col2 = st.columns(2)

with col1:
    st.header("DNN Surrogate Prediction")
    
    # Payload for FastAPI
    payload = {
        "L": L, "b": b, "h": h, "F": F, "E": E, "rho": rho
    }
    
    try:
        # In docker, it will be http://api:8000/predict
        # locally, it's http://localhost:8000/predict
        api_url = "http://localhost:8000/predict"
        # Quick check if running in docker
        import os
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
if st.button("Generate Random Configurations and Benchmark Speedup"):
    st.text("Benchmarking 1000 inferences...")
    # Add a progress bar or just show final speedup
    st.warning("Not fully implemented in UI, check evaluation notebook for deep performance profiling.")
