# Torsional Potential

# 1. Theoretical Background

## 1.1 Definition of Torsional Potential

The torsional potential quantifies the energy change associated with the rotation around a single bond, also known as a dihedral rotation. It arises from:

- **Steric interactions:** Repulsion between atoms due to close contact.
- **Electrostatic interactions:** Favorable or unfavorable dipole alignments during rotation.
- **Electronic effects:** Changes in orbital overlap, such as conjugation or hyperconjugation.

The torsional potential is a periodic function of the dihedral angle ($\phi$), defined as the angle between two planes formed by four consecutive atoms (A–B–C–D):

- **Dihedral angle ($\phi$):** The angle between the planes formed by atoms A–B–C and B–C–D.

---

## 1.2 Mathematical Expression of Torsional Potential

In molecular mechanics, the torsional potential is often expressed as a Fourier series:
$$
V_{\text{torsion}}(\phi) = \sum_n \frac{V_n}{2} \left(1 + \cos(n\phi - \gamma)\right),
$$
where:
- $\phi$: Dihedral angle (degrees or radians).
- $V_n$: Amplitude (barrier height) of the $n$-th torsional term (in kcal/mol or kJ/mol).
- $n$: Periodicity (number of minima in one full rotation).
- $\gamma$: Phase angle, which determines the position of the minimum energy ($\phi = \gamma$).

### Key Points:
1. **Periodic Nature:**
   - The potential energy repeats every $360^\circ / n$.
2. **Phase Angle:**
   - $\gamma = 0^\circ$ typically corresponds to a trans (planar) configuration.
   - $\gamma = 90^\circ, 180^\circ$, etc., correspond to gauche or eclipsed conformations.
3. **Single-Term Approximation:**
   - For simplicity, if only one dominant periodicity exists (e.g., for small molecules), the torsional potential can be approximated as:
     $$
     V_{\text{torsion}}(\phi) = \frac{V}{2} \left(1 + \cos(n\phi)\right).
     $$

### Example: Ethane ($\text{H}_3\text{C} - \text{CH}_3$)
- $n = 3$, as the $\text{C}-\text{C}$ bond has threefold symmetry.
- The torsional potential is:
  $$
  V_{\text{torsion}}(\phi) = \frac{V}{2} \left(1 + \cos(3\phi)\right).
  $$

---

## 1.3 Contributions to the Torsional Potential

The torsional potential is influenced by:
- **Steric Hindrance:** Repulsion between non-bonded atoms or groups due to spatial crowding.
- **Electronic Effects:** $\pi$-conjugation or hyperconjugation can lower the energy for certain dihedral angles (e.g., planar or staggered conformations).
- **Van der Waals Interactions:** Dispersion and repulsion between non-bonded atoms.

These contributions combine to form the overall torsional profile of a molecule.

---

## 1.4 Importance of Torsional Potentials

Torsional potentials are critical for understanding:
- **Molecular Conformations:** Preferred dihedral angles determine the stability and geometry of molecules.
- **Reaction Pathways:** Many chemical reactions involve rotation around bonds (e.g., isomerization, ring opening/closing).
- **Molecular Dynamics:** Torsional motions contribute to flexibility in biological macromolecules like proteins, DNA, and polymers.

---

# 2. Calculating Torsional Potentials

The torsional potential can be computed theoretically using quantum chemistry methods. The standard approach involves scanning the dihedral angle while optimizing other degrees of freedom and calculating the energy at each step.

---

## 2.1 Methodology

### Step 1: Define the Dihedral Angle
Identify the four atoms defining the dihedral angle ($\phi$) in the molecule.

### Step 2: Generate a Torsional Energy Scan
1. Incrementally rotate the dihedral angle ($\phi$) in fixed steps (e.g., $10^\circ$, $15^\circ$, etc.) from $0^\circ$ to $360^\circ$.
2. Optimize the remaining molecular geometry (except the dihedral being scanned) at each step.

### Step 3: Plot the Torsional Profile
1. Plot the computed energies ($E$) as a function of the dihedral angle ($\phi$).
2. Fit the resulting energy curve to a Fourier series to extract $V_n$, $n$, and $\gamma$.

---

## 2.2 Quantum Mechanical Methods

The torsional potential can be calculated using:
- **Ab Initio Methods:**
  - Methods like Hartree-Fock (HF) or post-Hartree-Fock (MP2, CCSD) capture electronic effects at a high level of theory.
- **Density Functional Theory (DFT):**
  - Methods like B3LYP or $\omega$B97XD are widely used for torsional scans due to their balance of accuracy and computational cost.
- **Semiempirical Methods:**
  - Useful for large systems but less accurate.

---

## 2.3 Computational Steps in ORCA

### Step 1: Perform a Dihedral Scan
To perform a torsional energy scan, use the `%geom Scan` keyword in ORCA. Here’s an example input file:

```plaintext
! B3LYP def2-SVP TightSCF Opt
%geom
  Constraints
    { D 1 2 3 4; C; 0, 360, 15 }
  end
end
%output Print[P_BondOrders] 1
* xyz 0 1
C     0.00000     0.00000     0.00000
C     1.54000     0.00000     0.00000
H    -0.54000     0.93500     0.00000
H    -0.54000    -0.93500     0.00000
H     2.08000     0.93500     0.00000
H     2.08000    -0.93500     0.00000
*
```
### Explanation:

1. **Constraints**: Specifies the dihedral angle to scan.
   - **D 1 2 3 4**: Defines the dihedral angle between atoms 1, 2, 3, and 4.
   - **C**: Perform a constrained scan of the dihedral.
   - **0, 360, 15**: Scan from $0^\circ$ to $360^\circ$ in $15^\circ$ steps.

2. **Optimization**: At each step, the geometry is optimized with the dihedral angle fixed.

---

### Step 2: Extract Energies

After running the job, extract the energies for each step from the ORCA output file.

---

### 2.4 Example Output

**Data Table**: After running the dihedral scan, ORCA will output the energies for each dihedral angle. An example table might look like:

| Dihedral Angle ($\phi$) | Energy (Hartree) | Energy (kcal/mol) |
|--------------------------|------------------|-------------------|
| 0°                       | -154.3210       | 0.00              |
| 15°                      | -154.3195       | 0.94              |
| 30°                      | -154.3150       | 3.77              |
| ...                      | ...             | ...               |

---

**Energy Profile**:  
Plot the dihedral angle ($\phi$) vs. energy to obtain the torsional potential profile.

---

### 2.5 Fitting to a Fourier Series

To extract the Fourier coefficients $V_n$, use least-squares fitting. The energy values can be fit to:

$$
V_{\text{torsion}}(\phi) = \sum_n \frac{V_n}{2} \left(1 + \cos(n\phi - \gamma)\right).
$$

This step can be performed using software like Python, MATLAB, or specialized fitting tools.

---

### 3. Applications of Torsional Potentials

#### 3.1 Conformational Analysis
- Identifying the most stable conformations and their relative energies.

#### 3.2 Reaction Mechanisms
- Torsional barriers can dictate the rates of rotation around bonds, influencing reaction dynamics.

#### 3.3 Molecular Dynamics
- Torsional potentials are essential parameters in force fields (e.g., AMBER, CHARMM) for simulating the behavior of biomolecules and polymers.

---

### 4. Limitations and Challenges

#### Convergence Issues:
- Torsional scans require tight convergence criteria to avoid numerical noise in the energy profile.

#### Environmental Effects:
- Solvent effects or bulk environments may alter torsional potentials, requiring explicit solvation models or polarizable continuum models (e.g., CPCM).

---

### 5. Conclusion

Torsional potentials provide critical insights into the rotational barriers and conformational flexibility of molecules. By performing dihedral scans using quantum chemistry tools like ORCA, one can calculate the torsional energy profile with high accuracy. This information is essential for understanding molecular conformations, reaction mechanisms, and dynamics in diverse applications such as materials science, drug discovery, and catalysis. 

The methodology outlined here allows researchers to accurately compute torsional potentials and extract the key parameters governing molecular behavior.



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
