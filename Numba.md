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
