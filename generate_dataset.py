import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from beam_physics import calculate_beam_properties

def generate_dataset(num_samples=10000):
    # Material properties
    # E: Young's Modulus (Pa), rho: Density (kg/m^3)
    materials = [
        {"name": "Steel", "E": 200e9, "rho": 7850},
        {"name": "Aluminum", "E": 69e9, "rho": 2700},
        {"name": "HDPE", "E": 1e9, "rho": 970}
    ]
    
    # 1. Latin Hypercube Sample
    # Dimensions: L, b, h, F, material_selector
    # material_selector is continuous [0, 1) and will be mapped to the 3 materials
    sampler = qmc.LatinHypercube(d=5)
    sample = sampler.random(n=num_samples)
    
    # Parameter bounds
    L_bounds = [0.1, 5.0]
    bh_bounds = [0.01, 0.2]
    F_bounds = [100, 50000]
    
    # Scale samples
    L = qmc.scale(sample[:, 0:1], [L_bounds[0]], [L_bounds[1]]).flatten()
    b = qmc.scale(sample[:, 1:2], [bh_bounds[0]], [bh_bounds[1]]).flatten()
    h = qmc.scale(sample[:, 2:3], [bh_bounds[0]], [bh_bounds[1]]).flatten()
    F = qmc.scale(sample[:, 3:4], [F_bounds[0]], [F_bounds[1]]).flatten()
    
    # Map material selector to materials
    mat_indices = np.floor(sample[:, 4] * len(materials)).astype(int)
    # Handle rare 1.0 case
    mat_indices[mat_indices == len(materials)] = len(materials) - 1
    
    E = np.array([materials[i]["E"] for i in mat_indices])
    rho = np.array([materials[i]["rho"] for i in mat_indices])
    
    df = pd.DataFrame({
        "L": L,
        "b": b,
        "h": h,
        "F": F,
        "E": E,
        "rho": rho
    })
    
    print("Computing physics outputs...")
    # 2. Compute outputs
    # Using our fast numpy solver
    outputs = df.apply(lambda row: calculate_beam_properties(
        row['L'], row['b'], row['h'], row['F'], row['E'], row['rho']), axis=1)
    
    df['max_deflection_m'], df['max_stress_Pa'], df['natural_freq_Hz'] = zip(*outputs)
    
    # 3. Add 1% Gaussian noise
    print("Adding 1% measurement noise...")
    for col in ['max_deflection_m', 'max_stress_Pa', 'natural_freq_Hz']:
        noise_std = 0.01 * np.abs(df[col])
        df[col] = df[col] + np.random.normal(0, noise_std)
        
    return df

def save_and_split(df):
    # 4. Split 80/10/10
    print("Splitting datasets...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    train_df.to_parquet("train_dataset.parquet", index=False)
    val_df.to_parquet("val_dataset.parquet", index=False)
    test_df.to_parquet("test_dataset.parquet", index=False)
    df.to_parquet("beam_dataset.parquet", index=False)
    print("Saved to Parquet.")
    
def plot_eda(df):
    print("Generating EDA plots...")
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap: Inputs vs Outputs")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
    plt.close()
    
    # Distribution of outputs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Using log scale for deflection and stress due to wide dynamic range
    sns.histplot(np.log10(df['max_deflection_m']), ax=axes[0], bins=50)
    axes[0].set_title("Log(Max Deflection [m]) Distribution")
    axes[0].set_xlabel("Log10(Deflection)")
    
    sns.histplot(np.log10(df['max_stress_Pa']), ax=axes[1], bins=50)
    axes[1].set_title("Log(Max Stress [Pa]) Distribution")
    axes[1].set_xlabel("Log10(Stress)")
    
    sns.histplot(np.log10(df['natural_freq_Hz']), ax=axes[2], bins=50)
    axes[2].set_title("Log(Natural Freq [Hz]) Distribution")
    axes[2].set_xlabel("Log10(Frequency)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "output_distributions.png"))
    plt.close()
    print("EDA plots saved in 'figures/' directory.")

if __name__ == "__main__":
    df = generate_dataset(10000)
    
    print("\nDataset Summary Stats:")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
    
    save_and_split(df)
    plot_eda(df)
