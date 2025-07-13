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

In diatomic molecules modeled by a Morse potential:

$$
V(x) = D_e (1 - e^{-\alpha x})^2,
$$

the ground-state $\langle x^2 \rangle$ predicts bond-length fluctuations, which shift infrared absorption lines and Raman scattering intensities.

### 2.2 Trapped Ion Qubits

Ions in a Paul trap have motional ground states approximated by a harmonic potential. The rms width $\sqrt{\langle x^2 \rangle}$ sets the Lamb-Dicke parameter:

$$
\eta = k \sqrt{\langle x^2 \rangle},
$$

controlling coupling between internal qubit states and motional modes.

### 2.3 Nanomechanical Sensors

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
