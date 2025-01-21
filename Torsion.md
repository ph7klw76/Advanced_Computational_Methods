

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Define the Ryckaert-Bellemans potential function
def ryckaert_bellemans(degree, c0, c1, c2, c3, c4, c5):
    """Ryckaert-Bellemans function for curve fitting."""
    cos_deg = np.cos(np.radians(degree))
    return c0 + c1 * cos_deg + c2 * cos_deg**2 + c3 * cos_deg**3 + c4 * cos_deg**4 + c5 * cos_deg**5

# Define a function to calculate R-squared
def calculate_r2(y_true, y_pred):
    """Calculate the R² value for a fit."""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Abstract function to load and sort data
def load_data(file_path):
    """Load and sort data by degree."""
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Degree", "Energy"])
    return data.sort_values("Degree")

# Abstract function to fit data and calculate R²
def fit_data(data):
    """Fit data to Ryckaert-Bellemans function and calculate R²."""
    params, _ = curve_fit(ryckaert_bellemans, data["Degree"], data["Energy"])
    y_pred = ryckaert_bellemans(data["Degree"], *params)
    r2 = calculate_r2(data["Energy"], y_pred)
    return params, r2

# Abstract function to generate fitted curve
def generate_fit(params, degrees):
    """Generate Ryckaert-Bellemans fitted curve."""
    return ryckaert_bellemans(degrees, *params)

# Main plotting function
def plot_data_with_fits(files, labels, colors, output_title="Energy vs. Degree"):
    """Plot data and Ryckaert-Bellemans fits."""
    plt.figure(figsize=(12, 8))
    degrees_fit = np.linspace(-180, 180, 500)  # Range for smooth curve fitting

    for file, label, color in zip(files, labels, colors):
        # Load and fit data
        data = load_data(file)
        params, r2 = fit_data(data)
        fit_curve = generate_fit(params, degrees_fit)

        # Plot data and fit
        plt.scatter(data["Degree"], data["Energy"], color=color, label=f"{label} Data", alpha=0.6)
        plt.plot(degrees_fit, fit_curve, color=color, linestyle="--", label=f"{label} Fit (R²={r2:.4f})")

    # Configure plot
    plt.xlabel("Degree", fontsize=16, fontweight="bold")
    plt.ylabel("Energy (eV)", fontsize=16, fontweight="bold")
    plt.title(output_title, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.tight_layout()
    plt.show()

# File paths and metadata
files = ['3potential.txt','5potential.txt', '7potential.txt']
labels = ["3-CrTri", "2-CrTri", "3,6-CzDiTri"]
colors = ["red", "green", "blue"]

# Generate the plot
plot_data_with_fits(files, labels, colors, "Donor-Acceptor Torsional Energy vs. Degree")
```
