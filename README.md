# AI Surrogate Model for Structural Simulation

This project demonstrates a Deep Learning surrogate model for structural engineering. It replaces standard analytical physics solvers (Euler-Bernoulli beam theory) or FEA with a neural network that predicts structural properties 500x faster while maintaining <5% Mean Absolute Percentage Error (MAPE). 

## Architecture
1. **Mathematical Ground Truth**: A pure NumPy implementation of the Euler-Bernoulli cantilever beam calculations.
2. **Synthetic Dataset**: 10,000 parameter combinations generated via Latin Hypercube Sampling (LHS), introducing 1% Gaussian noise to simulate empirical measurement uncertainty.
3. **DNN Surrogate Model**: A 4-layer fully-connected PyTorch neural network `[6 -> 128 -> 256 -> 128 -> 3]` with Batch Normalization and GELU activations.
4. **API Service**: A FastAPI backend serving the pre-trained neural network.
5. **Interactive Dashboard**: A Streamlit frontend allowing real-time parameter tweaking.

## Quickstart

### Running Locally (Docker)
Ensure Docker and `docker-compose` are installed.
```bash
docker-compose up --build
```
- **Streamlit App**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Backend**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Setup without Docker
1. Install dependencies: `pip install -r requirements.txt`
2. Run backend: `uvicorn api:app --reload --port 8000`
3. Run frontend: `streamlit run app.py`

## Physics Explained
The input space covers 6 variables: Length ($L$), Width ($b$), Height ($h$), Point Load ($F$), Young's Modulus ($E$), and Density ($\rho$).

The model learns three fundamental structural properties:
* **Max Deflection ($v_{max}$)**: $\frac{FL^3}{3EI}$
* **Max Stress ($\sigma_{max}$)**: $\frac{6FL}{bh^2}$
* **Natural Frequency ($f_n$)**: $\sim \sqrt{\frac{EI}{\rho A L^4}}$

## Outputs
- **Dataset**: `beam_dataset.parquet`
- **Trained Model**: `models/surrogate_model.pt`
- **Interactive UI**: Change inputs to see real-time updates and compare to analytical compute times.

## Demo Video
See `demo.webm` for a 90 second walkthrough of the application.
