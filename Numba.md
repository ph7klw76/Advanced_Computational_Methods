# Speeding up calculation using Numba libraries with examples

# Quantifying Quantum Delocalization: The Role of

$$
\langle x^2 \rangle = \int_{-\infty}^{\infty} x^2 \, |\psi_0(x)|^2 \, dx
$$

In both foundational theory and cutting-edge applications, the second moment of a quantum wavefunction $\langle x^2 \rangle$ serves as a key diagnostic of spatial delocalization, zero-point motion, and coupling strengths. Below, we develop the concept rigorously, survey its real-world impact, and demonstrate how Numba can accelerate Monte Carlo estimates of $\langle x^2 \rangle$ even for complex potentials.

## 1. Theoretical Foundations

Consider a one-dimensional quantum system with normalized ground-state wavefunction $\psi_0(x)$. The variance in position is

$$
\text{Var}(x) = \langle x^2 \rangle - \langle x \rangle^2 = \int x^2 \, |\psi_0(x)|^2 \, dx \quad (\text{for } \langle x \rangle = 0).
$$

This rms displacement:

- Encodes zero-point motion—the irreducible quantum fluctuations even at $T=0$.
- Saturates the uncertainty bound, since $\Delta x = \sqrt{\langle x^2 \rangle}$ pairs with $\Delta p = \hbar / (2 \Delta x)$ in minimum-uncertainty states.
- Determines average potential energy in harmonic or anharmonic traps:

$$
\frac{1}{2} m \omega^2 \langle x^2 \rangle.
$$

Beyond pedagogy, $\langle x^2 \rangle$ appears directly in:

- Debye–Waller factors for X-ray and neutron scattering (solid-state physics).
- Transition dipole moments $\langle 0|x|1 \rangle$ in molecular spectroscopy.
- Nanomechanical resonator amplitudes, where RMS displacement informs force sensitivity.
- Quantum dot confinements, setting electron–phonon coupling constants.

## 2. Real-World Applications

### 2.1 Molecular Vibrations & Spectroscopy

<img width="500" height="433" alt="image" src="https://github.com/user-attachments/assets/074c44bb-8463-4ffe-bcd2-a99468f2c6f3" />

In diatomic molecules modeled by a Morse potential:

$$
V(x) = D_e (1 - e^{-\alpha x})^2,
$$

the ground-state $\langle x^2 \rangle$ predicts bond-length fluctuations, which shift infrared absorption lines and Raman scattering intensities.

### 2.2 Trapped Ion Qubits

<img width="332" height="165" alt="image" src="https://github.com/user-attachments/assets/a9e1bb14-f797-4dd1-9875-bcc63a4c9816" />

Ions in a Paul trap have motional ground states approximated by a harmonic potential. The rms width $\sqrt{\langle x^2 \rangle}$ sets the Lamb-Dicke parameter:

$$
\eta = k \sqrt{\langle x^2 \rangle},
$$

controlling coupling between internal qubit states and motional modes.

### 2.3 Nanomechanical Sensors

<img width="242" height="208" alt="image" src="https://github.com/user-attachments/assets/4af69b08-3048-4511-947b-d3ac31be79e5" />

Cantilever or membrane resonators operating near the quantum regime rely on zero-point displacement noise:

$$
\sqrt{\frac{\hbar}{2m\omega}}, \quad \text{i.e., } \sqrt{\langle x^2 \rangle},
$$


to define their ultimate force or displacement sensitivity.

## 3. Monte Carlo Estimation with Numba

Closed-form integrals exist for simple potentials, but for complex, multidimensional or anharmonic systems, we turn to importance-sampled Monte Carlo:

$$
\langle x^2 \rangle = \int x^2 \, p(x) \, dx = \int x^2 \, \frac{p(x)}{q(x)} \, q(x) \, dx \approx \frac{1}{N} \sum_{i=1}^{N} x_i^2 \, w(x_i),
$$

where $x_i \sim q(x)$, and $w(x) = p/q$. Here’s an accelerated implementation.

```python
import numpy as np
import time
from numba import njit, prange

# --- 3.1 Define target density p(x)=|ψ₀|² for HO or Morse --- #
@njit(fastmath=True)
def psi0_sq_harmonic(x):
    # m=ω=ℏ=1
    return (1/np.sqrt(np.pi)) * np.exp(-x*x)

@njit(fastmath=True)
def psi0_sq_morse(x, D_e=5.0, α=1.0):
    # Approximate ground-state |ψ₀|² via analytic form (for demonstration)
    z = 2 * np.sqrt(2*D_e)/α * np.exp(-α*x)
    # Proper normalization omitted for brevity
    return z**(2*np.sqrt(2*D_e)/α - 1) * np.exp(-z)

# --- 3.2 Importance sampler: proposal q(x)=N(0,σ²) --- #
σ = 2.0
@njit(fastmath=True)
def q_pdf(x):
    return 1/(np.sqrt(2*np.pi)*σ) * np.exp(-x*x/(2*σ*σ))

# --- 3.3 Numba-accelerated MC estimator --- #
@njit(parallel=True, fastmath=True)
def mc_expectation(num_samples, seed, target_type=0):
    np.random.seed(seed)
    total = 0.0
    for i in prange(num_samples):
        x = np.random.normal(0.0, σ)
        if target_type == 0:
            p = psi0_sq_harmonic(x)
        else:
            p = psi0_sq_morse(x)
        w = p / q_pdf(x)
        total += x*x * w
    return total / num_samples

# --- 3.4 Benchmarking --- #
if __name__ == "__main__":
    N = 2_000_000
    # Warm up Numba
    mc_expectation(10_000, 0, 0)
    for target in (0, 1):
        t0 = time.time()
        result = mc_expectation(N, seed=12345, target_type=target)
        dt = time.time() - t0
        label = "Harmonic" if target==0 else "Morse"
        print(f"{label:8s} ⟨x²⟩ ≈ {result:.5f}  in {dt:.3f}s")
```
@njit(parallel=True) and prange distribute the N-sample loop over CPU threads.
fastmath=True vectorizes exponentials and arithmetic.
Seeding once at entry ensures reproducibility (with thread-dependent interleaving).
On a 6-core machine this typically yields a 20–50× speed-up over a pure-Python/NumPy loop.

High-performance computing is at the heart of modern computational physics. While Python offers a rich ecosystem for scientific programming, its interpreter overhead can become a bottleneck when tight loops or complex algorithms are involved. Numba bridges this gap by compiling selected Python functions to optimized machine code at runtime, often delivering C- or Fortran-like performance with minimal code changes. In this post, we’ll explore how Numba works under the hood, illustrate its key features with Python examples, and demonstrate its impact on two canonical physics problems.
To learn more about Numba click [here](https://numba.pydata.org/)

### 4.1 Monte Carlo Sampling of $\langle r^2 \rangle$ in 3D

Physically, many problems reduce to asking “what is the average squared displacement” under some distribution of steps—e.g. thermal kicks in a fluid, zero-point vibrational amplitudes in a crystal, or fluctuations in a nanomechanical resonator. In 3D,

$$
\langle r^2 \rangle = \int_{\mathbb{R}^3} (x^2 + y^2 + z^2) \, p(x, y, z) \, d^3r \approx \frac{1}{N} \sum_{i=1}^{N} (x_i^2 + y_i^2 + z_i^2),
$$

where $(x_i, y_i, z_i)$ are drawn from the target distribution—or via importance sampling.

#### 4.1.1 Nested Loops with `prange`
```python
import numpy as np
from numba import njit, prange
import time

σ = 1.0  # assume unit-variance isotropic Gaussian

@njit(parallel=True, fastmath=True)
def mc_r2_3d_nested(n_particles, n_samples_per, seed):
    """
    Monte Carlo: for each 'particle' we draw n_samples_per
    3D Gaussian displacements and accumulate r^2.
    """
    np.random.seed(seed)
    total = 0.0
    for i in prange(n_particles):
        for j in range(n_samples_per):
            x, y, z = np.random.normal(0.0, σ, size=3)
            total += x*x + y*y + z*z
    return total / (n_particles * n_samples_per)

# benchmark
if __name__ == "__main__":
    Np, Ns = 5000, 2000
    # warm-up compile
    mc_r2_3d_nested(10, 10, 0)
    t0 = time.time()
    val = mc_r2_3d_nested(Np, Ns, seed=42)
    print(f"Nested: ⟨r²⟩ ≈ {val:.5f} in {time.time()-t0:.3f}s")
```

Outer prange distributes particles across threads.
Inner loop stays serial, but overall work scales as O(Np×Ns)

### 4.2 Extracting MSD from Trajectory Data

In molecular dynamics or Brownian‐motion experiments, you often have a trajectory $\{r_i(t_k)\}$ for $i = 1 \dots N$ particles over times $t_k$. The mean-square displacement (MSD) at lag $\Delta t = t_{k+\ell} - t_k$ is

$$
\text{MSD}(\Delta t) = \frac{1}{NM} \sum_{i=1}^{N} \sum_{k=1}^{M} \|r_i(t_{k+\ell}) - r_i(t_k)\|^2.
$$

Below is a Numba-accelerated routine to compute MSD for a single lag $\ell$. We assume `positions` is a `float64` array of shape `(N_particles, N_frames, 3)`.

```python
import numpy as np
from numba import njit, prange
import time

@njit(parallel=True, fastmath=True)
def compute_msd(positions, lag):
    Np, Nf, _ = positions.shape
    M = Nf - lag
    total = 0.0

    # Loop over particles in parallel
    for i in prange(Np):
        for k in range(M):
            dx = positions[i, k+lag, 0] - positions[i, k, 0]
            dy = positions[i, k+lag, 1] - positions[i, k, 1]
            dz = positions[i, k+lag, 2] - positions[i, k, 2]
            total += dx*dx + dy*dy + dz*dz

    # Average over all (i, k)
    return total / (Np * M)

# Example usage
if __name__ == "__main__":
    # synthetic random-walk data
    Np, Nf = 1000, 500
    rng = np.random.default_rng(0)
    steps = rng.normal(scale=0.1, size=(Np, Nf, 3))
    positions = np.cumsum(steps, axis=1)

    # warm-up
    compute_msd(positions, lag=10)

    t0 = time.time()
    msd10 = compute_msd(positions, lag=10)
    print(f"MSD(Δt=10) = {msd10:.5f}  computed in {time.time()-t0:.3f}s")
```
## Why This Matters

The long-time slope of $\text{MSD}(\Delta t)$ vs.\ $\Delta t$ gives the diffusion coefficient:

$$
D = \frac{1}{6} \frac{d\langle r^2 \rangle}{dt}.
$$

- In biophysics, one tracks particle motions in cells to extract viscosity or active transport rates.
- In materials science, MSD of atoms over time reveals melting transitions or glassy arrest.

## 4.3 Performance Tips & Best Practices

- **Seed once at function entry** to maintain reproducibility—accept that interleaving across threads may differ if you change thread-count.
- **Flatten loops** when per-iteration work is uniform; nested `prange` can lead to imbalance if inner loop counts vary.
- **Contiguous arrays**: ensure `positions` and any large arrays are C-contiguous floats for best memory throughput.
- **`fastmath`** yields significant vectorization of exponentials and divisions—essential in thermal-motion or Coulomb‐force loops.
- **Cache warming**: the first call pays compilation overhead; subsequent calls run at full speed.



