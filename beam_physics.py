import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_beam_properties(L, b, h, F, E, rho):
    """
    Calculate max deflection, max stress, and natural frequency for a cantilever beam
    with a point load at the free end.
    
    Parameters:
    - L: Length (m)
    - b: Width/base (m)
    - h: Height/depth (m)
    - F: Load at free end (N)
    - E: Young's Modulus (Pa)
    - rho: Density (kg/m^3)
    
    Returns:
    - max_deflection: (m)
    - max_stress: (Pa)
    - natural_frequency: (Hz)
    """
    # Cross-sectional area
    A = b * h
    
    # Area moment of inertia for rectangular section
    I = (b * h**3) / 12
    
    # Max deflection at the free end
    max_deflection = (F * L**3) / (3 * E * I)
    
    # Max bending stress at the fixed end
    # c is the distance from neutral axis to extreme fiber
    c = h / 2
    M = F * L # Max moment at fixed end
    max_stress = (M * c) / I
    
    # Natural frequency (Hz) for first mode of cantilever beam
    # f = (1.875104^2 / (2 * pi)) * sqrt(E * I / (rho * A * L^4))
    natural_frequency = (1.875104**2 / (2 * np.pi)) * np.sqrt((E * I) / (rho * A * L**4))
    
    return max_deflection, max_stress, natural_frequency

def plot_deflection_curves():
    # 5 different beam configurations
    # (L, b, h, F, E, rho, name)
    configs = [
        (2.0, 0.05, 0.1, 1000, 200e9, 7850, "Steel - Standard Load"),
        (1.0, 0.05, 0.1, 1000, 200e9, 7850, "Steel - Short length"),
        (2.0, 0.05, 0.1, 5000,  69e9, 2700, "Aluminum - Heavy Load"),
        (2.0, 0.1,  0.1, 1000, 200e9, 7850, "Steel - Wide base"),
        (2.0, 0.05, 0.1,  100,   1e9,  970, "HDPE - Flexible")
    ]
    
    plt.figure(figsize=(10, 6))
    
    for L, b, h, F, E, rho, name in configs:
        x = np.linspace(0, L, 100)
        I = (b * h**3) / 12
        # Deflection curve equation for point load at end: v(x) = (F * x^2 / (6 * E * I)) * (3 * L - x)
        v = (F * x**2 / (6 * E * I)) * (3 * L - x)
        
        # Plot downward deflection
        plt.plot(x, -v, label=name)
        
    plt.title("Cantilever Beam Deflection Curves (Point Load at End)")
    plt.xlabel("Position along beam length x (m)")
    plt.ylabel("Downward Deflection v(x) (m)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "deflection_curves.png"), dpi=300)
    print(f"Saved deflection curves to {os.path.join(figures_dir, 'deflection_curves.png')}")

if __name__ == "__main__":
    plot_deflection_curves()
    
    # Test calculation
    v_max, s_max, f_n = calculate_beam_properties(L=2.0, b=0.05, h=0.1, F=1000, E=200e9, rho=7850)
    print("\nTest Calculation Ground Truth (Steel - Standard):")
    print(f"Length = 2.0m, b = 0.05m, h = 0.1m, Force = 1000N")
    print(f"Max Deflection: {v_max*1000:.3f} mm")
    print(f"Max Stress: {s_max/1e6:.2f} MPa")
    print(f"Natural Frequency: {f_n:.2f} Hz")
