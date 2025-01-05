# Vibrational Analysis and the Hessian Matrix: Using Quantum Methods to Predict and Analyze IR and Raman Spectra

**Vibrational spectroscopy**—encompassing both infrared (IR) and Raman spectroscopies—is one of the most powerful tools for elucidating molecular structure, bonding, and dynamics. At the heart of understanding and predicting these vibrational spectra lies the **Hessian matrix**, which encodes the second derivatives of the potential energy surface (PES) with respect to atomic displacements. Modern quantum chemical methods (e.g., Hartree–Fock, DFT, post-Hartree–Fock methods) compute this Hessian to obtain vibrational frequencies and related intensities.

## 1. Overview of Molecular Vibrations

In molecular systems composed of $N$ atoms, each atom has three spatial degrees of freedom, yielding a total of $3N$ Cartesian coordinates ($x_i$, $y_i$, $z_i$ for each atom $i$). However, a subset of these degrees of freedom corresponds to overall translations and rotations of the molecule as a rigid body, which do not contribute to vibrational motion:

- **For a nonlinear molecule**, there are 3 translational and 3 rotational degrees of freedom, leaving $3N - 6$ vibrational modes.
- **For a linear molecule**, there are 3 translational and 2 rotational degrees of freedom, leaving $3N - 5$ vibrational modes.

### 1.1 Potential Energy Surface (PES)

The **Potential Energy Surface (PES)**, $E(R)$, describes the electronic energy of the molecule as a function of its nuclear coordinates $R = \{x_1, y_1, z_1, \dots, x_N, y_N, z_N\}$. Physically, it represents how the electronic structure (and thus total energy) changes as atoms move.

- **Equilibrium Geometry** $R^{(0)}$: The point on the PES where the gradient of $E(R)$ vanishes, i.e.,

$$
  \frac{\partial E}{\partial R_\alpha} \bigg|_{R^{(0)}} = 0 \quad
$$

for all coordinates. This stationary point is usually a minimum for a stable molecule.

### 1.2 Small-Amplitude Vibrations and the Harmonic Approximation

Near the equilibrium geometry, the molecular potential $E(R)$ can be approximated by a Taylor series expansion. Let $\Delta R_\alpha = R_\alpha - R_\alpha^{(0)}$ denote small displacements from equilibrium:

- **Zeroth-order term**: $E(R^{(0)})$ is the energy at equilibrium.
- **First-order terms**: Proportional to the first derivatives $\partial E / \partial R_\alpha$ at equilibrium, which vanish because we are at a minimum.
- **Second-order (harmonic) terms**:
  
![image](https://github.com/user-attachments/assets/a58d4611-d7e8-4f22-bd89-411d221f2a41)

  
Thus, near the equilibrium, we can write:

![image](https://github.com/user-attachments/assets/fd13ee0e-4de7-46a9-a554-fe22f60a38e3)


where the subscript $0$ indicates evaluation at $R^{(0)}$. This quadratic expansion leads to the **harmonic approximation** for molecular vibrations.

### 1.3 Normal Mode Concept

In the harmonic (quadratic) approximation, the coupled vibrational problem simplifies into a set of independent harmonic oscillators called **normal modes**. Each normal mode is a collective motion in which all atoms vibrate in phase, with a characteristic frequency $\omega_k$.

- Physically, normal modes are eigenvectors of the mass-weighted Hessian (see below).
- Each normal mode has a distinct pattern of atomic displacements and a corresponding vibrational frequency.
- These normal modes form an orthonormal basis for the vibrational subspace, allowing us to decompose any small-amplitude nuclear motion into a superposition of these modes.

---

## 2. Hessian Matrix and Normal Modes

### 2.1 Definition of the Hessian Matrix (Force Constant Matrix)

The key quantity for analyzing small-amplitude vibrations is the **Hessian matrix** $F$, whose elements are given by:

$$
F_{\alpha \beta} = \frac{\partial^2 E}{\partial R_\alpha \partial R_\beta} \bigg|_{R^{(0)}},
$$

where $\alpha$ and $\beta$ index the $3N$ Cartesian coordinates $\{x_1, y_1, z_1, \dots \}$. This matrix is sometimes referred to as the **force constant matrix**, reflecting that $\frac{\partial^2 E}{\partial R_\alpha \partial R_\beta}$ is related to the “spring constant” coupling coordinate $\alpha$ and $\beta$.

#### 2.1.1 Interpretation of the Hessian

- **Diagonal elements** $F_{\alpha \alpha}$ measure the curvature of the potential energy in the direction of coordinate $\alpha$.
- **Off-diagonal elements** $F_{\alpha \beta}$ measure coupling between coordinates $\alpha$ and $\beta$; if large in magnitude, displacements in one coordinate significantly affect forces in the other coordinate.

### 2.2 Mass-Weighted Hessian

Because different atoms can have very different masses (e.g., hydrogen vs. heavier elements), we typically transform from the standard Cartesian coordinates $\{R_\alpha\}$ to mass-weighted coordinates $\{Q_\alpha\}$. Define:

$$
Q_\alpha = \sqrt{m_\alpha} \Delta R_\alpha,
$$

where $m_\alpha$ is the mass of the atom associated with coordinate $\alpha$. In matrix form, this transformation can be written using a mass matrix $M = \text{diag}(m_1, m_2, \dots, m_{3N})$. The **mass-weighted Hessian** $H$ is:

$$
H = M^{-1/2} F M^{-1/2},
$$

with elements:

$$
H_{\alpha \beta} = \frac{1}{\sqrt{m_\alpha m_\beta}} F_{\alpha \beta}.
$$

### 2.3 Normal Modes from Eigenvalue Problem

#### 2.3.1 Harmonic Oscillator Framework

In the harmonic approximation, the vibrational Hamiltonian can be written (in one dimension for simplicity) as:

![image](https://github.com/user-attachments/assets/610d9907-04e7-4882-a718-c1fb8c8fe025)


After switching to mass-weighted coordinates $Q_\alpha$, it simplifies to a set of decoupled harmonic oscillators if we diagonalize $H$.

#### 2.3.2 Diagonalization and Frequencies

To find the normal modes and their frequencies, we solve the generalized eigenvalue problem in mass-weighted coordinates:

$$
H e_k = \lambda_k e_k,
$$

where $e_k$ is the eigenvector for mode $k$, and $\lambda_k$ is the corresponding eigenvalue. The harmonic vibrational frequency $\omega_k$ (in radians per second) is obtained from:

$$
\omega_k = \sqrt{\lambda_k}.
$$

Often we express frequency in wavenumbers (cm$^{-1}$):

$$
\tilde{\nu}_k = \frac{\omega_k}{2 \pi c} = \frac{1}{2 \pi c} \sqrt{\lambda_k},
$$

where $c$ is the speed of light.

---
#### 2.4 Practical Implications
Dimensionality: For molecules with many atoms, the Hessian matrix can be very large (3𝑁×3𝑁3N×3N), requiring computationally efficient methods for evaluation and diagonalization.

Symmetry: Molecular symmetry can be exploited to reduce the computational cost and help classify normal modes according to symmetry species.

Interpretation and Scaling: Real molecules exhibit anharmonic effects, so harmonic frequencies are often slightly higher than experimental values. Empirical scaling factors (e.g., 0.96 to 0.99 for typical DFT) are commonly applied to improve agreement with experimental data.

Coupled Coordinates: Off-diagonal Hessian elements capture coupling between local motion (like bond stretching) and other degrees of freedom (like angle bending), resulting in more complex normal modes.

# 3. Quantum Chemical Computation of the Hessian

## 3.1 Electronic Structure Methods

To obtain $F_{\alpha \beta}$, quantum chemical methods approximate the electronic wavefunction or electron density and compute the energy derivatives with respect to nuclear coordinates. Some of the most commonly used methods include:

- **Hartree–Fock (HF)**: A mean-field approximation that solves the Schrödinger equation for a single Slater determinant.
- **Density Functional Theory (DFT)**: Uses functionals of the electron density (rather than wavefunctions) to calculate electronic energies.
- **Post-Hartree–Fock methods (e.g., MP2, CCSD)**: Include electron correlation corrections to the HF reference.

### 3.1.1 Analytical vs Numerical Derivatives

- **Analytical second derivatives**: Many electronic structure packages can compute the Hessian via direct evaluation of the second derivative of the energy with respect to coordinates.

- **Numerical second derivatives**: Alternatively, one can approximate second derivatives from finite differences of first derivatives:
  
![image](https://github.com/user-attachments/assets/ccb66a4e-c0c3-4b2e-b96f-aae75ac5f0d1)


Analytical derivatives are typically more accurate and often more efficient, but they are more complicated to implement. Numerical derivatives are straightforward in principle but can introduce errors from finite-difference spacing $\delta$.

## 3.2 Geometry Optimization and Hessian Computation

When performing a vibrational analysis in practice:

1. **Geometry Optimization**: One first locates the equilibrium geometry $\{x_i^{(0)}\}$ by setting the gradient ($\nabla E$) to zero.
2. **Hessian Calculation**: Next, at the optimized geometry, the Hessian matrix is computed (either analytically or numerically).
3. **Diagonalization**: The mass-weighted Hessian is diagonalized to yield normal modes and harmonic frequencies.

# 4. IR and Raman Intensities

**Vibrational spectroscopy** provides not only frequencies of molecular vibrations but also intensities—i.e., how strongly a given vibrational mode absorbs IR radiation (IR spectroscopy) or scatters (Raman spectroscopy). Intensity in both spectroscopies hinges on how an external electromagnetic field couples to the dynamic molecular property that the field interacts with: the dipole moment in IR and the polarizability in Raman.

## 4.1 IR Intensities

### 4.1.1 Theoretical Background

An infrared-active vibration requires that the molecular dipole moment $\mu$ changes as the nuclei move along the normal mode coordinate $Q_k$. The classical selection rule for IR activity can be stated as:

$$
\left( \frac{\partial \mu_\alpha}{\partial Q_k} \right) \neq 0 \quad \text{for at least one Cartesian component } \alpha \in \{x, y, z\}.
$$

If the dipole moment does not change for that vibrational motion, the mode is IR-inactive.

### 4.1.2 Transition Dipole Moment and Absorption Intensity

In quantum mechanical terms, the intensity of an IR absorption from the vibrational ground state $\vert 0 \rangle$ to the first excited state $\vert 1_k \rangle$ of mode $k$ is proportional to the square of the transition dipole moment:

$$
I_{\text{IR}, k} \propto \left| \langle 0 \vert \hat{\mu} \vert 1_k \rangle \right|^2.
$$

For a harmonic oscillator, we can expand the dipole operator $\hat{\mu}$ in a Taylor series around the equilibrium geometry (implicitly around the normal coordinate $Q_k = 0$):

![image](https://github.com/user-attachments/assets/76222b0f-3adc-42d9-9894-710c78744215)

Since $\mu(0)$ is independent of $\hat{Q}_k$, it gives no contribution to the transition from $\vert 0 \rangle$ to $\vert 1_k \rangle$. The first derivative term typically dominates:

$$
\hat{\mu} \approx \mu(0) + \left( \frac{\partial \mu}{\partial Q_k} \right)_0 \hat{Q}_k.
$$

Using ladder operators or the standard result for harmonic oscillator matrix elements, we find:

$$
\langle 0 \vert \hat{Q}_k \vert 1_k \rangle \neq 0, \quad \langle 0 \vert \hat{Q}_k^2 \vert 1_k \rangle = 0, \dots
$$

Therefore:

![image](https://github.com/user-attachments/assets/1d524546-5096-4933-b462-ffae4f9a4cd0)


**Key Takeaways**:
- IR intensity is governed by how strongly the dipole moment changes along the vibrational coordinate.
- Only modes for which $\frac{\partial \mu}{\partial Q_k} \neq 0$ are IR-active.
- In practice, quantum chemical software computes these partial derivatives by evaluating the dipole moment at geometries displaced along each normal mode direction.

## 4.2 Raman Intensities

### 4.2.1 Physical Origin of Raman Scattering

Raman scattering arises when an incoming photon induces an oscillating dipole in the molecule via its polarizability tensor $\alpha$. The scattered photon emerges at frequencies $\omega_0 \pm \omega_k$ (Stokes and anti-Stokes lines), where $\omega_0$ is the incident photon frequency and $\omega_k$ corresponds to the vibrational frequency of mode $k$. The Raman activity of a mode depends on how $\alpha$ (the molecular polarizability) changes along that vibration.

### 4.2.2 Polarizability Tensor and Raman Selection Rule

The polarizability tensor $\alpha_{ij}$ maps the electric field $E_j$ onto the induced dipole $P_i$:

$$
P_i = \sum_j \alpha_{ij} E_j.
$$

A mode is Raman-active if at least one component $\alpha_{ij}$ changes during the vibrational motion:

$$
\left( \frac{\partial \alpha_{ij}}{\partial Q_k} \right) \neq 0 \quad \text{for some } i, j.
$$

### 4.2.3 Raman Intensity Expressions

Classical treatments of Raman scattering often rely on the time-dependent induced dipole. However, in a more quantum mechanical picture, the Raman scattering cross-section (and thus the intensity) for mode $k$ can be shown to be proportional to the square of the first derivative of the polarizability tensor with respect to $Q_k$. One common simplified expression for totally symmetric modes in a depolarized limit is:

$$
I_{\text{Raman}, k} \propto \left| \left( \frac{\partial \bar{\alpha}}{\partial Q_k} \right)_0 \right|^2 \quad \text{where } \bar{\alpha} = \frac{1}{3} \text{Tr}(\alpha).
$$

In a more complete form, considering both the isotropic ($\bar{\alpha}$) and the anisotropic ($\gamma$) parts of the polarizability, the Raman intensity for each mode can be written as:

$$
I_{\text{Raman}, k} \propto 45 \left( \frac{\partial \bar{\alpha}}{\partial Q_k} \right)^2 + 7 \left( \frac{\partial \gamma}{\partial Q_k} \right)^2,
$$

where:

$$
\gamma = \frac{1}{2} \sum_{i, j} \left( \alpha_{ij} - \delta_{ij} \bar{\alpha} \right)^2.
$$

**Key Takeaways**:
- Raman intensity depends on how strongly the polarizability tensor changes along a normal mode.
- For a given mode $k$, $\frac{\partial \alpha}{\partial Q_k} \neq 0$ must hold for the vibration to be Raman-active.
- In practice, quantum chemical software computes $\alpha$ or relevant polarizability derivatives for small displacements along each normal mode.

# 5. Mathematical Derivations of Key Equations

Vibrational frequencies and their corresponding IR or Raman intensities can be derived from fundamental quantum mechanical principles combined with harmonic oscillator approximations. Below, we delve deeper into these derivations.

## 5.1 Harmonic Approximation and the Hessian

**Recap of the Hessian**: Around the equilibrium geometry $\{x_i^{(0)}\}$, the molecular potential energy $E(\{x_i\})$ can be expanded in a Taylor series. Because the first derivatives vanish at equilibrium, the second-order term dominates:

![image](https://github.com/user-attachments/assets/163dd2a8-eee1-4a6d-a33d-af3be74ec940)

where $F_{ij}$ is the Hessian matrix element $\frac{\partial^2 E}{\partial x_i \partial x_j}$ evaluated at equilibrium.

For vibrational analysis, we move to mass-weighted coordinates $Q_\alpha$ such that:

$$
Q_\alpha = \sqrt{m_\alpha} \Delta x_\alpha,
$$

leading to the mass-weighted Hessian $H$ (see earlier sections). Solving the eigenvalue problem:

$$
H e_k = \lambda_k e_k \implies \omega_k = \sqrt{\lambda_k},
$$

gives the harmonic vibrational frequencies $\omega_k$.

---

## 5.2 IR Selection Rules via Transition Dipole Moments

### 5.2.1 Vibrational Wavefunctions in the Harmonic Approximation

Within the harmonic approximation, the vibrational wavefunction of a single normal mode $Q_k$ resembles the quantum harmonic oscillator states $\vert n \rangle$, where $n = 0, 1, 2, \dots$. The Hamiltonian for mode $k$ (neglecting zero-point energy constants) is:

$$
\hat{H}_k = -\frac{\hbar^2}{2} \frac{d^2}{dQ_k^2} + \frac{1}{2} \omega_k^2 Q_k^2,
$$

in appropriate units (assuming mass-weighted coordinates absorb the mass factor).

The ground state $\vert 0_k \rangle$ and first excited state $\vert 1_k \rangle$ have wavefunctions:

$$
\psi_{0_k}(Q_k) \propto e^{-\frac{1}{2} \omega_k Q_k^2 / \hbar}, \quad \psi_{1_k}(Q_k) \propto Q_k e^{-\frac{1}{2} \omega_k Q_k^2 / \hbar},
$$

up to normalization constants.

### 5.2.2 Transition Dipole Moment Operator

The dipole moment operator can be expanded around $Q_k = 0$:

$$
\hat{\mu}(Q_k) \approx \mu(0) + \left( \frac{\partial \mu}{\partial Q_k} \right)_0 Q_k + \dots
$$

The IR transition from $\vert 0_k \rangle \to \vert 1_k \rangle$ involves matrix elements:

$$
\langle 0_k \vert \hat{\mu} \vert 1_k \rangle \approx \left( \frac{\partial \mu}{\partial Q_k} \right)_0 \langle 0_k \vert Q_k \vert 1_k \rangle.
$$

Since $\langle 0_k \vert Q_k \vert 1_k \rangle \neq 0$, the transition moment is non-zero if $\frac{\partial \mu}{\partial Q_k} \neq 0$. Its magnitude, squared, is proportional to the IR intensity.

---

## 5.3 Raman Scattering Cross Section: Polarizability Derivatives

### 5.3.1 Vibrational Dependence of Polarizability

In analogy to the dipole moment, the polarizability tensor can be expanded in normal coordinates:

$$
\alpha_{ij}(Q_k) \approx \alpha_{ij}(0) + \left( \frac{\partial \alpha_{ij}}{\partial Q_k} \right)_0 Q_k + \dots
$$

### 5.3.2 Scattering Induced by Polarizability Changes

When a monochromatic electric field $E = E_0 e^{i \omega_0 t}$ interacts with the molecule, the induced dipole is:

$$
P(t) = \alpha(Q_k) E(t).
$$

If the molecular geometry is vibrating with frequency $\omega_k$, the instantaneous value of $Q_k$ will vary with time. This time-dependent polarizability leads to a scattered field (Raman scattering). In the simplest classical treatment:

![image](https://github.com/user-attachments/assets/997eb384-4e3f-4426-9fa3-3545d5ff8baa)


where $Q_{k0}$ is the amplitude. Consequently, the induced dipole oscillates at frequencies $\omega_0$ (Rayleigh scattering) and $\omega_0 \pm \omega_k$ (Raman scattering).

### 5.3.3 Intensities in Terms of $\bar{\alpha}$ and $\gamma$

A more rigorous quantum-mechanical derivation yields formulas relating the Raman differential cross section $\frac{d\sigma}{d\Omega}$ to derivatives of $\bar{\alpha}$ (the mean polarizability) and $\gamma$ (the anisotropy). The Placzek polarizability theory (or Placzek approximation) is commonly used:

$$
\frac{d\sigma}{d\Omega}(\omega_0 \to \omega_0 \pm \omega_k) \propto \omega_0^4 \left[ 45 \left( \frac{\partial \bar{\alpha}}{\partial Q_k} \right)^2 + 7 \left( \frac{\partial \gamma}{\partial Q_k} \right)^2 \right],
$$

where:

$$
\bar{\alpha} = \frac{1}{3} (\alpha_{xx} + \alpha_{yy} + \alpha_{zz}),
\quad \gamma^2 = \frac{1}{2} \sum_{i,j} (\alpha_{ij} - \delta_{ij} \bar{\alpha})^2.
$$

Hence, if $\frac{\partial \alpha_{ij}}{\partial Q_k} = 0$ for all $i, j$, that mode does not scatter photons (Raman-inactive). If these derivatives are non-zero, the mode is Raman-active.

---

## 6. Putting It All Together

**Frequency Computation**:
1. The Hessian $F_{\alpha \beta}$ is computed at the equilibrium geometry using quantum chemistry.
2. Transforming to mass-weighted coordinates yields the matrix $H$, which is diagonalized to give eigenvalues $\lambda_k$.
3. The harmonic vibrational frequency is $\omega_k = \sqrt{\lambda_k}$.

**IR Intensities**:
- Proportional to $\left| \frac{\partial \mu}{\partial Q_k} \right|^2$.
- A non-zero derivative of the dipole moment w.r.t. the normal mode implies the mode absorbs IR radiation.

**Raman Intensities**:
- Depend on $\left| \frac{\partial \alpha}{\partial Q_k} \right|^2$.
- Changes in polarizability with the vibrational coordinate cause inelastic light scattering, yielding Raman lines.

In sum, these mathematical frameworks—centered on the harmonic approximation, the Hessian matrix, and the expansions of dipole and polarizability in normal coordinates—provide a robust theoretical basis for predicting both frequencies and intensities of molecular vibrations. Modern computational chemistry software automates these steps, allowing scientists to compare simulated IR/Raman spectra to experimental data, confirm molecular structures, and gain deeper insight into chemical bonding and molecular dynamics.





