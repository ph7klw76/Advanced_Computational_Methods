# 1. Introduction and Motivation of planewave basis set

## 1.1 The Periodic Solid and Its Hamiltonian

In solid-state physics, we study electrons in a crystalline solid. The first-principles (or ab initio) description centers on solving the electronic Schrödinger equation within the Born–Oppenheimer approximation:

$$
\hat{H} \Psi(r_1, \dots, r_N) = E \Psi(r_1, \dots, r_N),
$$

where $\hat{H}$ is the many-electron Hamiltonian, and $\Psi$ is the many-electron wavefunction. In practical solid-state calculations—especially within Density Functional Theory (DFT)—we convert this many-electron problem into self-consistent, single-particle-like Kohn–Sham equations:

$$
\left[ -\frac{\hbar^2}{2m} \nabla^2 + V_{\text{eff}}(r) \right] \psi_{nk}(r) = \epsilon_{nk} \psi_{nk}(r),
$$

where $V_{\text{eff}}(r)$ is an effective, periodic potential comprising the nuclear Coulomb potential, Hartree potential, and exchange-correlation potential. The quantum number $n$ indexes bands, while $k$ is the wavevector in reciprocal space.

### Why Plane Waves?

In crystals, the potential $V_{\text{eff}}(r)$ is periodic over the Bravais lattice. Plane waves (Fourier series) are a natural way to represent periodic functions (and wavefunctions satisfying Bloch’s theorem). This representation simplifies many operations, especially via the Fast Fourier Transform (FFT). Moreover, plane-wave expansions are guaranteed orthonormal, systematically improvable by increasing a kinetic-energy cutoff, and trivial to differentiate.

# 2. Bloch’s Theorem and the Plane-Wave Expansion

## 2.1 Periodicity and Bloch Functions

A crystal lattice is defined by primitive lattice vectors $a_1, a_2, a_3$. Any lattice vector $R$ is an integer combination of these primitive vectors. The potential $V_{\text{eff}}(r)$ satisfies:

$$
V_{\text{eff}}(r+R) = V_{\text{eff}}(r), \quad \forall R.
$$

Bloch’s theorem states that for a periodic potential, the single-particle eigenstates (the “Bloch orbitals”) can be written as:

$$
\psi_{nk}(r) = e^{i k \cdot r} u_{nk}(r),
$$

where:

- $k$ is the wavevector in the first Brillouin zone (FBZ).
- $u_{nk}(r)$ has the same periodicity as the lattice:

  $$
  u_{nk}(r+R) = u_{nk}(r).
  $$

This separation into a phase factor $\exp(i k \cdot r)$ and a periodic function $u_{nk}(r)$ simplifies the solution of the Kohn–Sham equations under periodic boundary conditions.

## 2.2 Expanding the Periodic Part in a Fourier (Plane-Wave) Series

Since $u_{nk}(r)$ is periodic, it can be expanded in a discrete Fourier series over the reciprocal lattice vectors $G$:

$$
u_{nk}(r) = \sum_G C_{nk}(G) e^{i G \cdot r}.
$$

Hence, the Bloch orbital itself becomes:

$$
\psi_{nk}(r) = e^{i k \cdot r} \sum_G C_{nk}(G) e^{i G \cdot r} = \sum_G C_{nk}(G) e^{i (k + G) \cdot r},
$$

where each plane wave is labeled by $k + G$. This is the **plane-wave basis set**—an orthonormal set of functions $\exp[i (k + G) \cdot r]$, indexed by the discrete reciprocal lattice vectors $G$. In a typical code, we write:

$$
\psi_{nk}(r) \approx \sum_{|k+G| \leq G_{\text{max}}} C_{nk}(G) e^{i (k + G) \cdot r},
$$

where $|k+G|$ is limited by some kinetic-energy cutoff:

$$
\frac{\hbar^2}{2m} |k+G|^2 \leq E_{\text{cut}},
$$

thus truncating the sum to a finite number of plane waves.

# 3. Mathematics and Derivation: From Bloch’s Theorem to the Kohn–Sham Matrix Equation

Let us sketch the derivation that leads from the Kohn–Sham equation in real space to the generalized matrix eigenvalue problem in reciprocal space.

## 3.1 Kohn–Sham Equation in Real Space

The Kohn–Sham equation can be written as:

$$
\left[ -\frac{\hbar^2}{2m} \nabla^2 + V_{\text{eff}}(r) \right] \psi_{nk}(r) = \epsilon_{nk} \psi_{nk}(r).
$$

Given the potential $V_{\text{eff}}(r)$ is periodic, we assume Bloch’s form for $\psi_{nk}(r)$. Substituting

$$
\psi_{nk}(r) = \sum_G C_{nk}(G) e^{i (k + G) \cdot r}
$$

into the equation, we obtain:

$$
\sum_G C_{nk}(G) \left[ -\frac{\hbar^2}{2m} \nabla^2 + V_{\text{eff}}(r) \right] e^{i (k + G) \cdot r} = \epsilon_{nk} \sum_G C_{nk}(G) e^{i (k + G) \cdot r}.
$$

## 3.2 Expressing the Potential in Reciprocal Space

Because $V_{\text{eff}}(r)$ is also periodic, we can expand it in a Fourier series:

$$
V_{\text{eff}}(r) = \sum_{G'} \tilde{V}_{\text{eff}}(G') e^{i G' \cdot r},
$$

where $\tilde{V}_{\text{eff}}(G')$ are the Fourier coefficients of the potential (obtained from a Poisson solver or from direct discrete Fourier transforms in a plane-wave code).

## 3.3 Action of the Laplacian and Potential on Plane Waves

**Kinetic Term:**

$$
-\frac{\hbar^2}{2m} \nabla^2 e^{i (k + G) \cdot r} = \frac{\hbar^2}{2m} |k + G|^2 e^{i (k + G) \cdot r}.
$$

**Potential Term:**


![image](https://github.com/user-attachments/assets/12d8ef14-4d8a-48a7-8ef3-02040ea010fb)



Hence, the potential **couples** plane waves $k + G$ to plane waves $k + G + G'$.

## 3.4 Putting It All Together: Matrix Form

Collecting terms, the Kohn–Sham equation in reciprocal space becomes:

![image](https://github.com/user-attachments/assets/8f337c6c-3194-4cfa-96f9-49d68a69df4d)


This is a **generalized eigenvalue problem** in the plane-wave coefficients $C_{nk}(G')$. In practice, we:

1. **Truncate** $G$ and $G'$ by a kinetic-energy cutoff $E_{\text{cut}}$.
2. **Diagonalize** the resulting finite matrix for each $k$.

The **eigenvalues** $\epsilon_{nk}$ approximate the band structure, and the **eigenvectors** $C_{nk}(G')$ give the plane-wave composition of the wavefunction.

# 4. Practical Implementation in Solid-State Codes

## 4.1 The Role of FFT and Real-Space Grids

A hallmark of plane-wave DFT codes is the efficient use of the Fast Fourier Transform (FFT) to switch between reciprocal space and real space:

- **Reciprocal-Space Representation**: Multiplication by a reciprocal-space potential is just multiplication of the plane-wave coefficients by $\tilde{V}_{\text{eff}}(G)$.
- **Real-Space Representation**: Operators such as the local potential can sometimes be easier to handle directly in real space; the electron density $\rho(r)$ is typically evaluated in real space and then transformed to reciprocal space for convolution with the Coulomb kernel, etc.

Because FFT scales as $N \log N$ (where $N$ is the number of grid points), plane-wave DFT codes can treat systems with hundreds or even thousands of atoms (given sufficient computational resources).

## 4.2 Kinetic-Energy Cutoff $E_{\text{cut}}$

The largest computational control parameter in plane-wave codes is the kinetic-energy cutoff:

$$
E_{\text{cut}} \iff \frac{\hbar^2}{2m} |k+G|^2 \leq E_{\text{cut}}.
$$

- A higher $E_{\text{cut}}$ means more $G$-vectors, a larger basis set, and more accurate wavefunctions—but with higher CPU and memory costs.
- A lower $E_{\text{cut}}$ speeds up computations but may yield poor accuracy (e.g., Pulay stress, forces, and energies will be off).

Hence, one must **converge results** (total energy, structural parameters, etc.) with respect to $E_{\text{cut}}$.

## 4.3 Brillouin-Zone Sampling (k-Points)

Because wavefunctions depend on $k$, we must sample $k$-space in the first Brillouin zone (FBZ). Techniques like Monkhorst–Pack or Gamma-centered grids systematically choose discrete $k$-points to approximate integrals over $k$ (e.g., total energy, electron density). More $k$-points yield better sampling but a higher computational cost.

## 4.4 Boundary Conditions and Supercells

Plane-wave calculations typically assume **periodic boundary conditions (PBCs)**. For truly periodic solids, this is physically correct. For molecules or surfaces, we use a **large supercell with vacuum padding** in at least one direction to minimize spurious interactions with periodic images. A plane-wave code always sees a “repeating pattern,” so for non-periodic or finite systems, we ensure the cell is large enough that repeated images do not interact significantly.

# 5. Pseudopotentials and Beyond

## 5.1 The Core–Valence Problem

In a plane-wave basis, describing core electrons accurately would require an extremely high kinetic-energy cutoff because core orbitals vary rapidly near the nucleus. Moreover, these deep-lying electrons typically do not participate significantly in chemical bonding.

## 5.2 Norm-Conserving Pseudopotentials (NCPP) and Ultrasoft Pseudopotentials (USPP)

Pseudopotentials replace the all-electron potential near the nucleus by a “softer” effective potential that reproduces valence-electron scattering properties but does not demand an extremely fine plane-wave expansion. Key pseudopotential types include:

- **Norm-Conserving**: Ensure that the pseudo-wavefunction matches the all-electron wavefunction beyond a chosen cutoff radius $r_c$.
- **Ultrasoft**: Relax certain norm constraints to gain an even softer potential, reducing the plane-wave cutoff needed.

## 5.3 Projector-Augmented Wave (PAW) Method

The **PAW method** (Blöchl, 1994) effectively combines all-electron orbitals with a smooth pseudo-wavefunction in the interstitial region. It introduces “augmentation” expansions in spherical regions around atoms, bridging the gap between a pseudopotential approach and an all-electron approach. Nevertheless, the global basis in most PAW implementations remains plane waves.

# 6. Extensions and Variations

Although pure plane waves are standard in many solid-state packages, a few advanced methods combine or modify plane-wave expansions:

### **Linearized Augmented Plane Waves (LAPW)**

Used in codes like **WIEN2k**, the crystalline volume is partitioned into muffin-tin spheres (atomic regions) plus an interstitial region. Inside spheres, solutions are expanded in spherical harmonics; in the interstitial, plane waves are used.

### **Mixed-Basis Methods**

Sometimes local orbitals or atomic orbitals (Gaussians, numerical basis functions) are mixed with plane waves to capture strongly localized states more efficiently.

### **Real-Space Grids**

Instead of a plane-wave basis, some DFT codes discretize wavefunctions on uniform real-space grids. Mathematically akin to plane waves (due to equivalence via FFT), but the approach can simplify boundary conditions in certain contexts.

# 7. Advantages and Disadvantages of Plane-Wave Basis Sets

## **Advantages**

- **Orthogonality & Completeness**: Plane waves are naturally orthonormal in a periodic box.
- **Systematic Convergence**: Increasing the cutoff $E_{\text{cut}}$ systematically improves accuracy.
- **Simplicity of Derivatives**: Derivatives in reciprocal space are straightforward (multiplication by $iG$).
- **FFT Efficiency**: Transformations between real and reciprocal space are computationally efficient (**FFT**), enabling large-scale DFT.
- **Simple Implementation**: Many core DFT operations (e.g., convolution with the Coulomb kernel) become multiplications in reciprocal space.

## **Disadvantages**

- **Large Basis for Molecules**: Non-periodic or localized systems require large supercells and high cutoffs.
- **Inefficient Near the Nuclei**: Rapid oscillations of core states demand very high $E_{\text{cut}}$ unless pseudopotentials or PAW are employed.
- **Vacuum Regions**: For surfaces or isolated molecules, a significant fraction of the grid is effectively vacuum, adding computational overhead.

# 8. Summary of the Mathematical Structure

- **Bloch’s Theorem**:
  
$$
\psi_{nk}(r) = e^{i k \cdot r} u_{nk}(r).
$$

- **Periodic Part as a Fourier Series**:
  
$$
u_{nk}(r) = \sum_G C_{nk}(G) e^{i G \cdot r}.
$$

- **Kohn–Sham Hamiltonian in Reciprocal Space leads to the matrix equation**:

![image](https://github.com/user-attachments/assets/746f8651-e349-4c86-9826-1653623c4680)


  truncated by $G, G'$ up to some cutoff.

- **Pseudopotentials** drastically reduce the needed $G$-space size for valence electrons.

# 9. Practical Tips and Best Practices

- **Convergence Testing**
  - Always test convergence with respect to the plane-wave kinetic-energy cutoff $E_{\text{cut}}$.
  - Converge the k-point mesh for your property of interest (energy, forces, band gap, etc.).

- **Pseudopotential or PAW Choice**
  - Ensure the pseudopotential (or PAW dataset) is well-tested for your element.
  - Validate with known benchmarks or all-electron results when possible.

- **Pulay Stress**
  - Under-converged plane-wave basis can lead to unphysical stress in the cell, affecting geometry optimization.

- **Parallelization & FFT Grids**
  - Plane-wave codes parallelize well across $k$-points and also across 3D FFT grids.
  - Choose FFT grid parameters consistent with (or slightly beyond) the kinetic-energy cutoff.

# 10. Conclusion

Plane-wave basis sets form the foundation of many state-of-the-art solid-state DFT codes because they seamlessly handle the periodic boundary conditions of crystals, enable fast transforms between real and reciprocal space, and provide a systematic route to convergence. The method’s reliance on pseudopotentials or PAW methods elegantly circumvents the core-electron challenge, allowing plane waves to focus on valence states.

Mathematically, plane-wave expansions emerge directly from **Bloch’s theorem** for periodic potentials. Numerically, they are implemented by introducing a **kinetic-energy cutoff** that limits the number of reciprocal-lattice vectors (plane waves), turning the **Kohn–Sham equation** into a solvable matrix eigenproblem. With modern high-performance computing and efficient parallelization, **plane-wave DFT** can handle large-scale crystal systems (100s to 1000s of atoms) with impressive accuracy—provided that systematic convergence checks and high-quality pseudopotentials are employed.


# Toy Python Program: Plane-Wave Basis for a 1D Model Schrödinger Equation

Below is a toy Python program that illustrates the basic idea of using a **plane-wave basis** to solve (in one dimension) a model Schrödinger or Kohn–Sham-like equation with a simple **periodic potential**.

## Note:

- This code is **not** a fully fledged density-functional theory (DFT) code. Instead, it captures the essence of how one might:
  - Set up a **plane-wave Hamiltonian** in reciprocal space.
  - **Truncate it** by a kinetic-energy-like cutoff.
  - **Diagonalize** to obtain “band energies.”
- We use a **1D periodic potential** (a simple cosine). Real solid-state codes do this in **3D**, include **pseudopotentials**, **self-consistency loops**, etc.
- The code is **heavily commented** to explain each step.

```python
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1. Define the system and simulation parameters
##############################################################################

L = 2.0 * np.pi           # Length of the 1D "unit cell"
m = 1.0                   # Mass (set to 1 for simplicity)
hbar = 1.0                # Planck's constant / 2π = 1 for simplicity
v0 = 0.5                  # Amplitude of the cosine potential
g0 = 2                    # "Wave number index" for the potential (dimensionless)
                          #   The actual wave number is g0 * (2π / L).

Nmax = 5                  # Truncation for reciprocal lattice vectors:
                          #   We will include plane waves G = -Nmax .. +Nmax.
                          #   That means the Hamiltonian matrix is (2*Nmax+1) x (2*Nmax+1).

nk = 100                  # Number of k-points we will sample in the 1D Brillouin zone
                          #   We'll sweep k in [-π/L, +π/L] or something similar.

##############################################################################
# 2. Build the list of reciprocal lattice vectors G
##############################################################################

# G_base = 2π / L, the fundamental reciprocal lattice spacing.
G_base = 2.0 * np.pi / L

# We'll collect G values from -Nmax to +Nmax
G_vals = np.array([i * G_base for i in range(-Nmax, Nmax + 1)])
# This yields an array: [... -2*Nmax*pi/L, ..., 0, ..., 2*Nmax*pi/L].

##############################################################################
# 3. Represent the periodic potential in reciprocal space
##############################################################################
# We choose a simple 1D potential: V(x) = v0 * cos(g_actual * x),
#   where g_actual = g0 * G_base = g0 * (2π / L).
# The *Fourier coefficients* of cos( g_actual x ) are:
#   V(G = ± g_actual) = v0/2, and zero otherwise (neglecting constant shifts).
#
# We'll store these Fourier components in a dictionary for easy lookup.

V_recip = {}
g_actual = g0 * G_base

# The cosine can be written as cos(g_actual x) = 0.5(e^{i g_actual x} + e^{-i g_actual x}).
# Hence the nonzero Fourier amplitudes are at +g_actual and -g_actual, each with amplitude v0/2.
V_recip[+g_actual] = v0 / 2.0
V_recip[-g_actual] = v0 / 2.0

##############################################################################
# 4. Define a function to build the Hamiltonian matrix for a given k
##############################################################################

def build_hamiltonian(k):
    """
    Build the plane-wave Hamiltonian matrix H_{G,G'} for the 1D system at wavevector k.
    The matrix dimension is (2*Nmax+1) x (2*Nmax+1).
    
    Args:
        k (float): The wavevector in the 1D Brillouin zone.
    Returns:
        H (2D complex ndarray): The Hamiltonian matrix in the {e^{i(k+G)x}} basis.
    """
    size = 2*Nmax + 1  # dimension of the matrix
    H = np.zeros((size, size), dtype=complex)
    
    # Loop over all pairs (G, G')
    for i, G1 in enumerate(G_vals):
        for j, G2 in enumerate(G_vals):
            # 1) Kinetic term (only appears if i == j)
            if i == j:
                # E_kin = (hbar^2 / 2m) * (k + G1)^2
                H[i, j] += (hbar**2 / (2.0*m)) * (k + G1)**2
            
            # 2) Potential term couples plane waves differing by G_diff = G1 - G2
            #    We look up the Fourier coefficient V(G_diff).
            G_diff = G1 - G2
            # Retrieve the reciprocal-space component if it exists
            if np.isclose(G_diff, +g_actual, atol=1e-12):
                H[i, j] += v0/2.0
            elif np.isclose(G_diff, -g_actual, atol=1e-12):
                H[i, j] += v0/2.0
    
    return H

##############################################################################
# 5. Diagonalize the Hamiltonian at many k-points and store the band energies
##############################################################################

# We'll define a k-range around the "first Brillouin zone" for 1D, typically [-π/a, +π/a].
# Here a = L, so we do [-π/L, +π/L].
k_min = -np.pi / L
k_max = +np.pi / L
k_vals = np.linspace(k_min, k_max, nk)

# We'll store the eigenvalues in an array of shape (nBands, nk).
# nBands = 2*Nmax+1 is the dimension of our truncated basis.
nBands = 2*Nmax + 1
band_energies = np.zeros((nBands, nk))

for ik, kpt in enumerate(k_vals):
    # Build H(k)
    H_k = build_hamiltonian(kpt)
    
    # Diagonalize
    w, v = np.linalg.eigh(H_k)
    
    # Sort eigenvalues ascending and store
    w_sorted = np.sort(w.real)
    band_energies[:, ik] = w_sorted

##############################################################################
# 6. Plot the resulting “band structure”
##############################################################################

plt.figure(figsize=(6, 4))
for band_index in range(nBands):
    plt.plot(k_vals, band_energies[band_index, :], marker='o', linestyle='-')
plt.xlabel(r"$k$ (1D wavevector)")
plt.ylabel(r"Energy (arbitrary units)")
plt.title("1D Plane-Wave Band Structure for V(x) = v0 cos(g0 * x)")
plt.grid(True)
plt.tight_layout()
plt.show()
```

# Explanation of Key Steps

## Reciprocal Lattice & Plane-Wave Basis

- We define a **1D lattice** of length $L$. The reciprocal space is thus discretized by:

  $$
  G = n \times \frac{2\pi}{L}.
  $$

- We truncate these reciprocal lattice vectors to **$\pm N_{\max}$**, giving a finite matrix dimension of:

  $$
  (2N_{\max} +1) \times (2N_{\max} +1).
  $$

## Bloch’s Theorem

- For each wavevector $k$, the wavefunction is:

  $$
  \psi_k(x) = \sum_G C_k(G) e^{i(k+G)x}.
  $$

- We build the Hamiltonian in the basis:

  $$
  \{ e^{i(k+G)x} \}.
  $$

## Kinetic Term

- The **diagonal kinetic-energy part** is:

  $$
  \frac{\hbar^2}{2m} (k+G)^2.
  $$

## Potential Term

- Our potential is:

  $$
  V(x) = v_0 \cos(g_0 x).
  $$

- In reciprocal space, this has **nonzero Fourier components** at $\pm g_0$.
- This **couples plane waves** differing by $\pm g_0$. Hence the matrix elements:

  $$
  H_{G,G'} \propto \delta_{G-G', \pm g_0}.
  $$

## Diagonalization

- For each $k$, we form the matrix **$H(k)$** and solve the eigenvalue problem:

  $$
  H(k) C = E C.
  $$

- The resulting eigenvalues **$E$** are the band energies.
- Repeating over many $k$-points yields a **discrete representation of the band structure**.

## Plot

- We collect all the **eigenvalues** as functions of $k$ and **plot them**, mimicking a **1D band structure diagram**.

---

Although this script is a **simple educational illustration**, it shows the fundamental ideas behind **plane-wave expansions in periodic systems**:

- **Bloch states** split into a factor **$e^{ikx}$** times a sum over reciprocal lattice vectors.
- **We truncate the basis** in practice to manage the matrix size.
- **The periodic potential couples different $G$-values**.
- **Diagonalizing yields the energy eigenvalues** (bands).

In **real 3D solid-state codes**, the same principle is used (with **3D vectors** $k$ and $G$), plus:

- **Pseudopotentials** or **PAW** to handle atomic cores.
- **Self-consistent calculation** of the potential.
- More sophisticated **$k$-point sampling**.

Nonetheless, the **core concept remains the same** as in this **1D example**.

![image](https://github.com/user-attachments/assets/9ce4380e-2182-46dd-af0e-185742c43651)

