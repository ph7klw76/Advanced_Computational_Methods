# 1. Introduction

Molecular spectroscopy in the UV-Vis range probes electronic transitions between quantized energy levels in molecules. When a photon of appropriate energy (frequency) interacts with a molecule, it can induce an absorption transition from a lower electronic state (often the ground state) to a higher electronic state. Conversely, when the molecule returns to lower energy levels, it may emit photons in a process commonly known as **fluorescence** or **phosphorescence**, depending on spin multiplicity changes.

To rigorously understand these processes, we rely on quantum mechanics:

- **Time-Dependent Perturbation Theory** to derive transition probabilities.
- **Born–Oppenheimer Approximation** to separate electronic and nuclear (vibrational) coordinates.
- **Franck–Condon Principle** for vibronic transitions.
- **Solvent Models** (e.g., PCM) to incorporate environmental effects.
- **Jablonski Diagram** to visualize all radiative and non-radiative processes.
- **Quantum Yield** and **Lifetime** expressions to quantify emission efficiency and rates.

---

# 2. Electronic Transitions: Selection Rules

## 2.1 Time-Dependent Perturbation Theory and Fermi’s Golden Rule

The total molecular Hamiltonian can be written as:

![image](https://github.com/user-attachments/assets/32dda946-a23a-4d61-a1b4-7677ead37b83)

where:

- $\hat{H}_0$ is the unperturbed (time-independent) Hamiltonian of the molecule,
- $\hat{H}_{\text{int}}(t)$ is the time-dependent perturbation, typically the interaction with the electromagnetic field.

For an electric dipole transition in the semi-classical approximation, the perturbation Hamiltonian is:

$$
\hat{H}_{\text{int}}(t) = -\hat{\mu} \cdot E(t),
$$

where:

- $\hat{\mu}$ is the electric dipole operator,
- $E(t)$ is the electric field of the light wave.

---

### 2.1.1 Transition Probability

Using first-order time-dependent perturbation theory, the transition probability per unit time ($W_{i \to f}$) from an initial state $\vert i \rangle$ to a final state $\vert f \rangle$ with energies $E_i$ and $E_f$ is given by **Fermi’s Golden Rule**:

$$
W_{i \to f} = \frac{2\pi}{\hbar} \left| \langle f \vert \hat{H}_{\text{int}} \vert i \rangle \right|^2 \delta(E_f - E_i - \hbar \omega).
$$

---

## 2.2 Spin and Laporte Selection Rules

### Spin Selection Rule:

$$
\Delta S = 0.
$$

Transitions generally must conserve the total spin quantum number. In purely electric dipole transitions (no strong spin-orbit coupling), spin cannot flip.

---

### Laporte Selection Rule (in centrosymmetric molecules):

Allowed transitions must involve a change in parity (gerade $\leftrightarrow$ ungerade). For example, in octahedral complexes, $d \to d$ transitions are often “Laporte forbidden” (but may become partially allowed through mixing of orbitals or vibration).

---

### Other Selection Rules:

- **Angular Momentum Rule**: $\Delta l = \pm 1$ in a one-electron picture for atomic orbitals.
- **Dipole Selection Rule**: $\langle f \vert \hat{\mu} \vert i \rangle \neq 0$ must hold for the transition to be dipole-allowed.

# 3. Franck–Condon Principle: Vibronic Transitions

## 3.1 Overview and Born–Oppenheimer Approximation

Molecular electronic transitions typically occur on the order of femtoseconds, while nuclear (vibrational) motions are significantly slower (on the order of picoseconds or longer). Hence, during an electronic transition, the nuclear coordinates are effectively **frozen**. This is the essence of the Franck–Condon (FC) principle.

Under the Born–Oppenheimer approximation, the total molecular wavefunction $\Psi(r, R)$ factorizes into an electronic part $\psi_e(r; R)$ and a nuclear (vibrational) part $\chi_v(R)$:

$$
\Psi(r, R) = \psi_e(r; R) \chi_v(R),
$$

where $r$ are electronic coordinates, and $R$ are nuclear coordinates. Each electronic state $e$ can have multiple vibrational energy levels $v$.

---

## 3.2 Franck–Condon Principle and Transition Intensities

### 3.2.1 Vibronic Transitions

An electronic transition from a vibrational level $\vert v_i(e) \rangle$ in the ground electronic state $e$ to a vibrational level $\vert v_f(e') \rangle$ in the excited electronic state $e'$ has a transition dipole moment:

$$
\langle v_f(e') \vert \langle \psi_e(e') \vert \hat{\mu} \vert \psi_e(e) \rangle \vert v_i(e) \rangle = \langle \psi_e(e') \vert \hat{\mu} \vert \psi_e(e) \rangle \times \langle v_f(e') \vert v_i(e) \rangle,
$$

assuming that the electronic and vibrational wavefunctions factorize:

- $\langle \psi_e(e') \vert \hat{\mu} \vert \psi_e(e) \rangle$ is the purely electronic transition dipole moment.
- $\langle v_f(e') \vert v_i(e) \rangle$ is the vibrational overlap integral, also called a **Franck–Condon factor**.

Thus, the intensity $I_{v_i \to v_f}$ for the vibronic transition is proportional to the product:

$$
I_{v_i \to v_f} \propto \left| \langle \psi_e(e') \vert \hat{\mu} \vert \psi_e(e) \rangle \right|^2 \times \left| \langle v_f(e') \vert v_i(e) \rangle \right|^2.
$$

---

### 3.2.2 Franck–Condon Region

On an energy diagram (one-dimensional cut for the nuclear coordinate), the vertical transition from the ground-state potential energy surface (PES) to the excited-state PES at the same nuclear geometry is often referred to as the **Franck–Condon transition**. The highest-intensity transitions occur when there is maximum overlap between the vibrational wavefunctions $\chi_{v_i}(e)$ and $\chi_{v_f}(e')$.

---

## 3.3 Huang–Rhys Factor and Franck–Condon Factors

### 3.3.1 Physical Meaning of the Huang–Rhys Factor $S$

The Huang–Rhys factor, denoted $S$, is a dimensionless parameter that quantifies the displacement between the ground-state and excited-state harmonic oscillators in a one-dimensional (or sometimes effective) model. Specifically:

- Consider a single vibrational mode of frequency $\omega_{\text{vib}}$.
- Let the equilibrium position of this mode in the ground state be $Q_0$, and the equilibrium position in the excited state be $Q_1$.
- Define the dimensionless displacement $\Delta Q$ in terms of the oscillator’s characteristic length (for a harmonic oscillator of frequency $\omega_{\text{vib}}$).

The Huang–Rhys factor $S$ is related to the square of that dimensionless displacement:

$$
S = \frac{1}{2} \left( \frac{M \omega_{\text{vib}}}{\hbar} \right) (Q_1 - Q_0)^2 \equiv \text{(dimensionless measure of PES displacement)},
$$

where $M$ is the effective mass of the vibrational coordinate.

---

### 3.3.2 Relationship Between $S$ and Franck–Condon Intensities

When the harmonic oscillator approximation is valid for both ground and excited electronic states, the Franck–Condon factor from the vibrational ground state $v_i = 0$ in the ground electronic state to the $v_f$-th vibrational level in the excited electronic state follows a Poisson-like distribution in terms of $S$:

$$
\left| \langle v_f(e') \vert 0(e) \rangle \right|^2 = e^{-S} \frac{S^{v_f}}{v_f!}.
$$

This result comes from expanding the displaced harmonic oscillator wavefunctions and integrating.

**Key Takeaway**:

- **$S$ Large** $\implies$ The excited-state PES is significantly displaced from the ground-state PES, leading to significant vibronic structure and strong intensity for higher $v_f$ transitions.
- **$S$ Small** $\implies$ Minimal displacement, meaning transitions predominantly occur from $v_i = 0$ to $v_f = 0$ (the so-called 0-0 transition), with little vibrational progression.

![image](https://github.com/user-attachments/assets/63ce1394-941e-4953-973a-67d44922252f)

---

### 3.3.3 Significance of FC Factors and Huang–Rhys Factor

- **Franck–Condon Factors**: Determine which vibrational bands will be most intense in both absorption and emission. They encapsulate the overlap of nuclear wavefunctions in the two electronic manifolds.
- **Huang–Rhys Factor**: Quantifies how strongly a particular vibrational mode is excited (or relaxed) during electronic transitions. A high $S$ indicates large nuclear reorganization upon excitation, typical of systems with significant geometry changes (e.g., large bond length changes, big dipole changes).

# 4. Emission Processes

When a molecule in an excited electronic state relaxes to a lower-energy state, emission may occur. Emission can be **radiative** (photon emission) or **non-radiative** (internal conversion, intersystem crossing, etc.).

---

## 4.1 Radiative Transitions: Fluorescence and Phosphorescence

### 4.1.1 Spontaneous Emission and Einstein’s $A$ Coefficient

For a radiative transition $e' \to e$ from excited-state energy $E_{e'}$ to ground-state energy $E_e$, the rate of spontaneous emission is often written as the Einstein $A$ coefficient ($A_{e' \to e}$):

$$
A_{e' \to e} = \frac{\omega_{e'e}^3}{3\pi \epsilon_0 \hbar c^3} \left| \mu_{e'e} \right|^2,
$$

where:

- $\omega_{e'e} = (E_{e'} - E_e) / \hbar$ is the transition angular frequency,
- $\mu_{e'e} = \langle \psi_e(e) \vert \hat{\mu} \vert \psi_e(e') \rangle$ is the electronic transition dipole (possibly multiplied by appropriate Franck–Condon factors for vibronic transitions).

---

#### 4.1.1.1 Derivation Sketch

1. **Time-Dependent Perturbation Theory**: The spontaneous emission probability can be interpreted as the system coupling to the vacuum electromagnetic modes.
2. **Planck Radiation Law and Detailed Balance**: Einstein introduced constants $A$ and $B$ to relate absorption and emission rates.
3. **Final Expression**: The spontaneous emission rate is proportional to $\omega^3 \left| \mu \right|^2$, as shown above.

---

### 4.1.2 Fluorescence vs. Phosphorescence

- **Fluorescence**: Refers to singlet-to-singlet transitions, e.g., $S_1 \to S_0$. Spin selection rules $\Delta S = 0$ allow for relatively fast emission (nanoseconds).
- **Phosphorescence**: Involves triplet-to-singlet transitions ($T_1 \to S_0$). Because this is spin-forbidden in the absence of spin-orbit coupling, it often has a longer timescale (microseconds to seconds), although heavier atoms (with stronger spin-orbit coupling) can enhance phosphorescence rates.

---

## 4.2 Non-Radiative Transitions

### 4.2.1 Internal Conversion (IC)

#### 4.2.1.1 Conceptual Overview

Internal Conversion (IC) is a radiationless process where a molecule in an upper electronic state ($e'$) transitions to a lower electronic state ($e$) of the same spin multiplicity (e.g., $S_1 \to S_0$ or $S_2 \to S_1$). It typically occurs at near-degenerate (or closely spaced) vibrational levels of the upper and lower electronic states and is facilitated by **vibronic coupling**—the mixing of electronic and vibrational wavefunctions.

---

#### 4.2.1.2 Hamiltonian Partition and Fermi’s Golden Rule

The total Hamiltonian $\hat{H}$ can be written as:

![image](https://github.com/user-attachments/assets/9dc7d657-c26c-4589-828c-cbac48160af5)

where:

- $\hat{H}_0$ is the Born–Oppenheimer (BO) part, describing each electronic state (and its associated vibrational structure) independently.
- $\hat{H}_{\text{vib}}$ is the vibronic coupling (off-diagonal in the electronic basis) that enables transitions between different electronic states at certain nuclear configurations.

For initial $\vert i \rangle \equiv \vert e', v_i \rangle$ and final $\vert f \rangle \equiv \vert e, v_f \rangle$ states, the transition rate $k_{i \to f}$ is given by Fermi’s Golden Rule:

$$
k_{i \to f} = \frac{2\pi}{\hbar} \left| \langle f \vert \hat{H}_{\text{vib}} \vert i \rangle \right|^2 \delta(E_i - E_f),
$$

where:

- $\hat{H}_{\text{vib}}$ contains terms depending on the derivative of the electronic Hamiltonian with respect to nuclear coordinates, enabling transitions across PESs.
- $\delta(E_i - E_f)$ enforces energy conservation within vibrational manifolds.

---

#### 4.2.1.3 Summation Over Final States and Density of States

The total IC rate $k_{\text{IC}}$ is obtained by summing over all possible final vibrational states:

![image](https://github.com/user-attachments/assets/dff9c4ea-8684-4da3-add5-1c4ff4782e63)


In a dense manifold of vibrational levels, this sum can become an integral, defining a vibrational density of states $\rho(E)$:

![image](https://github.com/user-attachments/assets/2bf7c996-992c-4527-a867-afed07d647ad)


The magnitude of this rate depends on:

- The magnitude of vibronic coupling $\hat{H}_{\text{vib}}$.
- The overlap in energy between the initial state $(e', v_i)$ and final vibrational states $(e, v_f)$.
- The vibrational density of states $\rho(E)$, which generally increases with vibrational energy for large molecules.

---

#### 4.2.1.4 Kasha’s Rule and Vibronic Considerations

Kasha’s Rule in photochemistry states:

**"Emission (fluorescence) generally occurs from the lowest excited singlet state, regardless of the initially excited singlet."**

The fast IC rates between $S_n \to S_1$ are attributed to large vibronic coupling and dense vibrational manifolds, enabling efficient radiationless relaxation to $S_1$.

---

### 4.2.2 Intersystem Crossing (ISC)

#### 4.2.2.1 Conceptual Overview

Intersystem Crossing (ISC) is a radiationless transition between different spin states (e.g., $S_1 \leftrightarrow T_1$). It is spin-forbidden in the absence of spin-orbit coupling, but the presence of heavy atoms (high atomic number) or specific orbital mixing can enhance spin-orbit coupling, making ISC more probable.

---

#### 4.2.2.2 Spin-Orbit Coupling Hamiltonian

The spin-orbit coupling Hamiltonian $\hat{H}_{\text{SO}}$ is given by:

$$
\hat{H}_{\text{SO}} = \frac{\alpha^2}{2} \sum_i \hat{S}_i \cdot (\hat{r}_i \times \nabla V(r_i)),
$$

where:

- $\alpha$ is the fine-structure constant ($\approx 1/137$),
- $\hat{S}_i$ is the spin operator for electron $i$,
- $V(r_i)$ is the electrostatic potential (often from the nucleus).

---

#### 4.2.2.3 Fermi’s Golden Rule for ISC

The rate of ISC from an initial state $\vert S, v_i \rangle$ (singlet) to final states $\vert T, v_f \rangle$ (triplet) is:

![image](https://github.com/user-attachments/assets/a3fbfa32-09a3-41c9-a93a-658195aca00a)

---

### 4.2.3 Energy Gap Law

The **Energy Gap Law** states:

- Larger the energy gap $\Delta E = E_{e'} - E_e$, the lower the non-radiative rate.
- High $\Delta E$ implies many vibrational quanta must be involved, reducing vibrational wavefunction overlap.

---

### 4.2.4 Total Non-Radiative Rate $k_{\text{nr}}$

The total population decay out of an excited electronic state $e'$ is:

$$
\frac{dN_{e'}}{dt} = -(k_r + k_{\text{nr}}) N_{e'},
$$

where:

$$
k_{\text{nr}} = k_{\text{IC}} + k_{\text{ISC}} + \dots
$$

Depending on the system, one or more non-radiative channels dominate.

# 5.Foundational Models of Solvent–Solute Interactions

There are two broad approaches:

1. **Continuum (Implicit) Solvent Models**: Treat the solvent as a dielectric continuum characterized by $ϵ$. A cavity (often with a shape approximating the solute’s electron density contour) is formed around the solute, and the reaction field (i.e., the solvent’s response) is computed self-consistently.  
   Examples: Polarizable Continuum Model (PCM), Onsager Reaction Field Model.

2. **Explicit Solvent Models**: Represent individual solvent molecules explicitly (e.g., via molecular dynamics or Monte Carlo simulations). Although more accurate, these are computationally expensive.

Below, we focus on continuum models, which yield analytic or semi-analytic equations that elucidate spectral shifts in a rigorous yet computationally feasible manner.

---

## 5.1. Born/Onsager-Type Reaction Field Concepts

### 5.2 Sphere-in-Continuum (Classical Born Model)

#### 5.2.1 Basic Electrostatics Setup

Consider a point charge $q$ at the center of a spherical cavity of radius $a$, embedded in a continuum of dielectric constant $ϵ$. The electrostatic (solvation) free energy $G_{\text{solv}}$ of placing the charge in the solvent is found by solving Poisson’s equation with boundary conditions at the cavity surface:

$$
\nabla^2 \phi(r) = -\frac{\rho(r)}{ϵ_0 ϵ}.
$$

For a point charge in a spherical cavity, the resulting Born solvation energy (classical approximation) is:

$$
\Delta G_{\text{solv}} = -\frac{q^2}{8 \pi ϵ_0 a} \left( 1 - \frac{1}{ϵ} \right).
$$

**Physical Meaning**:
- If $ϵ > 1$, the solvation energy is negative, indicating the solvent stabilizes the charge by polarizing around it.
- As $ϵ \to \infty$,
  
$$
\Delta G_{\text{solv}} \to -\frac{q^2}{8 \pi ϵ_0 a}.
$$

Although simplistic (point charge, spherical cavity), this formula captures the qualitative essence of solvent stabilization.

---

#### 5.2.2 Extension to Dipoles

For a dipole $\mathbf{p}$ in a spherical cavity, the solvation free energy (Onsager model) becomes proportional to $\mathbf{p}^2$ and depends on the reaction field $\mathbf{R}$ created by the dielectric:

$$
\Delta G_{\text{solv}} = -\frac{1}{2} \mathbf{p} \cdot \mathbf{R}.
$$

One can show that:

$$
\mathbf{R} = \frac{2 (ϵ - 1)}{(2ϵ + 1) a^3} \mathbf{p},
$$

leading to:

$$
\Delta G_{\text{solv}} = -\frac{(ϵ - 1)}{(2ϵ + 1) a^3} \mathbf{p}^2.
$$

---

## 5.3. Lippert–Mataga (or Lippert) Equation for Solvatochromism

When a molecule absorbs or emits light, it often changes its dipole moment from the ground state $\mu_g$ to the excited state $\mu_e$. The difference $\Delta \mu = \mu_e - \mu_g$ interacts differently with the solvent, causing energy-level shifts. A popular semi-empirical formula for the Stokes shift $\Delta \nu = \nu_{\text{abs}} - \nu_{\text{em}}$ is the Lippert–Mataga equation:

$$
\Delta \nu = \nu_{\text{abs}} - \nu_{\text{em}} = \frac{2 h c}{a^3} \left( \frac{ϵ - 1}{2ϵ + 1} - \frac{n^2 - 1}{2n^2 + 1} \right) (\mu_e - \mu_g)^2 + \text{constant},
$$

where:

- $\nu_{\text{abs}}$ and $\nu_{\text{em}}$ are the wavenumbers $(\text{cm}^{-1})$ of absorption and emission maxima, respectively,
- $h$ is Planck’s constant, $c$ is the speed of light,
- $ϵ$ is the static dielectric constant of the solvent, $n$ is the refractive index,
- $a$ is an effective cavity radius of the solute molecule,
- $\Delta \mu^2 = (\mu_e - \mu_g)^2$ is the square of the difference in dipole moments between excited and ground states.

---

### 5.4 Derivation Sketch

1. **Initial and Final Energies**:
   - Ground-state energy: $G_{\text{solv}}(g) = -\alpha \mu_g^2$,
   - Excited-state energy: $G_{\text{solv}}(e) = -\alpha \mu_e^2$,  
     where $\alpha$ is a constant that depends on $ϵ$, $n$, and $a$.

2. **Transition Energies**:

![image](https://github.com/user-attachments/assets/82c18385-bab8-48c9-ba5d-1e728d66d6f4)

where $\tilde{G}_{\text{solv}}$ represents the solvation free energies after solvent relaxation.
     
3. Linear response or simple reaction field assumption: the difference in 𝜇g  vs. 𝜇𝑒 
  under different states leads to solvatochromic shifts that can be captured in an expression linear in 𝜖

4. **Subtraction**:
   Subtracting $\nu_{\text{em}}$ from $\nu_{\text{abs}}$ yields an expression proportional to $\Delta \mu^2$ and a function of $ϵ$ and $n$.

Hence, the Lippert–Mataga equation emerges.

---

## 5.5. Polarizable Continuum Model (PCM)

### 5.5.1 General Theory

The Polarizable Continuum Model (PCM) is a quantum chemistry approach where:

1. **Molecular Cavity**: The solute is enclosed by a cavity based on the van der Waals surface or another chosen surface.
2. **Reaction Field**: The solute polarizes the surrounding dielectric, which then interacts back with the solute.
3. **Self-Consistency**: This interaction modifies the solute's energy levels, which are solved self-consistently.

The Hamiltonian for the solvated system becomes:

![image](https://github.com/user-attachments/assets/c863085f-e900-470a-9745-f186ed0d7a74)


where:

- $\hat{H}_0$ is the gas-phase Hamiltonian (electrons + nuclei),
- $\hat{V}_{\text{reaction}}$ is the reaction-field potential depending on the solute charge distribution $\rho(r)$, cavity boundary, and $ϵ$.

---

### 5.5.2 PCM Equations: Self-Consistent Reaction Field (SCRF)

1. **Surface Charge**:
   The molecular charge density $\rho(r)$ induces a surface charge $\sigma(s)$ on the cavity boundary $s$:
   
$$
\sigma(s) = -ϵ_0 [ϵ - 1] \left[ \frac{\partial \phi(s)}{\partial n_s} \right],
$$
   
   where $\phi(s)$ is the solute’s electrostatic potential on the surface.

3. **Reaction Potential**:
   The reaction potential $\phi_{\text{reaction}}(r)$ felt by the solute is:
   
$$
\phi_{\text{reaction}}(r) = \int_s \frac{\sigma(s')}{4 \pi ϵ_0 ϵ \vert r - s' \vert} ds'.
$$

5. **Self-Consistency**:
   Solve the Schrödinger equation:
   
![image](https://github.com/user-attachments/assets/145816f7-2338-4a0f-82db-f0b0a44da5cf)


Iterate until $\rho(r)$ and $\sigma(s)$ converge.

---

### 5.5.3 Computing Spectral Shifts with PCM

1. **Ground State ($S_0$)**: Solve the SCF or post-SCF equation for the ground-state wavefunction $\Psi_g$ and energy $E_g$.
2. **Excited State ($S_1, S_2, \dots$)**: Use TD-DFT or CIS with PCM to obtain excited-state energies $E_e$.
3. **Absorption Energy**:
   
$$
\Delta E_{\text{abs}} \approx E_e - E_g.
$$

5. **Emission Energy**:
   Include solvent relaxation effects to compute the relaxed excited-state energy $E_e(\text{relaxed})$.

---

# 7. Modeling Singlet and Triplet State Transitions

## 7.1 Electronic Spin States: Singlet vs. Triplet

For a two-electron system (typical in many electronic transitions):

- **Singlet State ($S$)**: The total spin $S = 0$. The two electron spins are anti-parallel (spin wavefunction is antisymmetric), leading to a single ($m_s = 0$) overall spin state.
- **Triplet State ($T$)**: The total spin $S = 1$. The two electron spins are parallel (spin wavefunction is symmetric), yielding three degenerate spin sub-levels ($m_s = -1, 0, +1$) in the absence of spin-orbit coupling.

### 7.1.1 Spin Selection Rules

In purely electric dipole transitions (without heavy-atom effects), the usual spin selection rule is:

$$
\Delta S = 0.
$$

Hence, singlet–triplet transitions (e.g., $S_1 \leftrightarrow T_1$) are formally spin-forbidden unless spin-orbit coupling (SOC) is present to mix singlet and triplet characters.

---

## 7.2 Spin-Orbit Coupling and State Mixing

### 7.2.1 Spin-Orbit Hamiltonian

The spin-orbit coupling (SOC) operator $\hat{H}_{SO}$ couples the spin $\hat{S}$ and orbital angular momentum $\hat{L}$. In a one-electron approximation:

$$
\hat{H}_{SO} = \alpha^2 \sum_i \left( \frac{Z_i}{r_i} \right) \hat{L}_i \cdot \hat{S}_i,
$$

where:

- $\alpha \approx 1/137$ is the fine-structure constant,
- $Z_i$ is the nuclear charge for electron $i$,
- $r_i$ is the distance of electron $i$ from the nucleus.

For heavier atoms (large $Z$), spin-orbit coupling becomes stronger, enabling enhanced singlet–triplet mixing.

---

### 7.2.2 Perturbative Treatment of Singlet–Triplet Mixing

If $\hat{H}_{SO}$ is small relative to the electronic energy spacings, we can treat SOC as a perturbation:

1. **Unperturbed Hamiltonian $\hat{H}_0$**: Yields spin-pure singlet ($\vert S \rangle$) and triplet ($\vert T \rangle$) eigenstates.
2. **Perturbation $\hat{H}_{SO}$**: Mixes these states, creating new eigenstates $\vert \psi_1 \rangle, \vert \psi_2 \rangle, \dots$ that are linear combinations:
   
$$
   \vert \psi \rangle = c_S \vert S \rangle + c_T \vert T \rangle.
$$

4. The coefficients $(c_S, c_T)$ depend on the magnitudes of spin-orbit matrix elements, e.g., $\langle S \vert \hat{H}_{SO} \vert T \rangle$ and $E_S - E_T$. Larger matrix elements or smaller energy gaps enhance mixing.

---

## 7.3 Intersystem Crossing (Singlet $\leftrightarrow$ Triplet)

Intersystem Crossing (ISC) refers to the radiationless transition between states of different spin multiplicities, for instance:

$$
S_1 \xrightarrow{\text{ISC}} T_1, \quad T_1 \xrightarrow{\text{ISC}} S_0,
$$

depending on which states are energetically accessible.

---

### 7.3.1 Fermi’s Golden Rule for ISC

Using time-dependent perturbation theory, the rate $k_{ISC}$ from an initial state $\vert S, v_i \rangle$ (singlet, vibrational level $v_i$) to final states $\vert T, v_f \rangle$ (triplet, vibrational level $v_f$) can be written:

![image](https://github.com/user-attachments/assets/c54aca44-30ae-4c30-a166-5d5cdf9558db)


where:

- $\hat{H}_{SO}$ is the spin-orbit coupling operator,
- $\hat{H}_{vib}$ includes the vibronic coupling (dependence of electronic states on nuclear coordinates),
- The Dirac delta $\delta(\dots)$ enforces energy conservation across vibrational manifolds.

If $\langle S \vert \hat{H}_{SO} \vert T \rangle$ is large (e.g., heavy atoms, strong orbital angular momentum), ISC rates can be significant. Conversely, for molecules composed of only light atoms (e.g., H, C, N, O), ISC can be much slower unless specific structural features enhance SOC (e.g., carbonyl groups).

---

## 7.4 Phosphorescence (Triplet $\to$ Singlet Radiative Decay)

When a triplet state ($T_1$) transitions radiatively back to the singlet ground state ($S_0$), this is phosphorescence. It is typically spin-forbidden, but again made possible by spin-orbit coupling mixing in some singlet character to $T_1$. This leads to longer emission lifetimes (microseconds to seconds) compared to fluorescence (nanoseconds).

---

### 7.4.1 Einstein $A$-Coefficient for Triplet–Singlet Emission

Analogous to singlet–singlet transitions:

$$
A_{T_1 \to S_0} = \frac{\omega_{T_1 S_0}^3}{3\pi ϵ_0 \hbar c^3} \vert \langle S_0 \vert \hat{\mu} \vert T_1 \rangle \vert^2,
$$

where:

- $\omega_{T_1 S_0} = (E_{T_1} - E_{S_0}) / \hbar$ is the transition frequency,
- $\langle S_0 \vert \hat{\mu} \vert T_1 \rangle = \sum_k c_k \langle S_0 \vert \hat{\mu} \vert S_k \rangle$,
  with $\vert T_1 \rangle$ partially mixed with $\vert S_k \rangle$ states via SOC, giving a (typically small) but non-zero dipole transition moment.

---

# 8. Quantum Yield and Lifetime

Quantum yield ($\Phi$) and lifetime ($\tau$) are critical measures of how efficiently a molecule emits light and how long it remains in an excited state before de-excitation.

---

## 8.1 Definitions and Key Equations

### Quantum Yield $\Phi$:

$$
\Phi = \frac{\text{number of photons emitted}}{\text{number of photons absorbed}} = \frac{k_r}{k_r + k_{nr}}.
$$

- $k_r$ is the radiative rate (sum of all radiative channels, e.g., fluorescence, phosphorescence).
- $k_{nr}$ is the non-radiative rate (sum of internal conversion $k_{IC}$, intersystem crossing $k_{ISC}$, and other non-radiative pathways).

---

### Lifetime $\tau$:

$$
\tau = \frac{1}{k_{tot}}, \quad k_{tot} = k_r + k_{nr}.
$$

This $\tau$ is the mean or average time the molecule spends in the excited state before returning to the ground state (or another lower-energy state).

---

## 8.2 Radiative Rate $k_r$: Einstein $A$-Coefficients

For a singlet–singlet electronic transition ($S_1 \to S_0$), the radiative rate is often expressed via the Einstein $A$-coefficient:

$$
k_r = A_{S_1 \to S_0} = \frac{\omega^3}{3\pi ϵ_0 \hbar c^3} \vert \langle S_0 \vert \hat{\mu} \vert S_1 \rangle \vert^2,
$$

where $\omega = (E_{S_1} - E_{S_0}) / \hbar$.

---

### 8.3 Non-Radiative Rate $k_{nr}$

$$
k_{nr} = k_{IC} + k_{ISC} + \dots
$$

Each of these terms can be estimated using Fermi’s Golden Rule approaches, as described earlier for internal conversion and intersystem crossing. The net effect is to reduce the population available for radiative emission, thus lowering the quantum yield.



## analyse HR and reorganization energy

```python

import pandas as pd
import matplotlib.pyplot as plt

# Load the data (replace 'Book1.csv' with the actual path if necessary)
file_path = 'Book1.csv' #Freq	HR-factor	reorganization energy
data = pd.read_csv(file_path)

# Extract relevant columns for plotting
frequency = data['Freq']
hr_factor = data['HR-factor']
reorganization_energy = data['reorganization energy']

# Create the plot with thicker bars
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot HR-factor as thicker blue bars on the left y-axis
bar1= ax1.bar(frequency, hr_factor, label='HR-factor', color='tab:blue', alpha=0.8, width=5, align='center')
ax1.set_xlabel('Frequency (cm⁻¹)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Huang-Rhys factor', color='tab:blue', fontsize=16, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.yaxis.label.set_fontweight('bold')
ax1.xaxis.label.set_fontweight('bold')
# Set the x-axis limits
ax1.set_xlim(0, 2000)
# Create a second y-axis for reorganization energy with far thicker red bars
ax2 = ax1.twinx()
bar2=ax2.bar(frequency, reorganization_energy, label='Reorganization Energy (cm⁻¹)', color='tab:red', alpha=0.8, width=5, align='edge')
ax2.set_ylabel('Reorganization Energy (cm⁻¹)', color='tab:red', fontsize=16, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
ax2.yaxis.label.set_fontweight('bold')

# Title and grid
# fig.legend([bar1, bar2], ['HR-factor', 'Reorganization Energy'], loc='center left', fontsize=12)
plt.title('', fontsize=16, fontweight='bold')
fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Show the plot
plt.show()
```


## analyse FC and energy shift


![image](https://github.com/user-attachments/assets/e2e9220d-6d47-436c-95ae-8fefb8ad88d9)

```python
import pandas as pd
import numpy as np
import math  # Import the standard library math module
import matplotlib.pyplot as plt

# Load the data
file_path = 'Book1.csv'
data = pd.read_csv(file_path)

# Extract relevant columns
freq = data['Freq']
hr_factors = data['HR-factor']

# Define constants
vibrational_spacing = 1400  # Vibrational energy spacing in cm⁻¹
sigma = vibrational_spacing / 500  # Gaussian width (adjust for broadening)
num_points = 100000  # Number of points in the spectrum
energy_range = np.linspace(0, 10000, num_points)  # Energy range in cm⁻¹

# Function to calculate Franck–Condon Factors
def calculate_fcf(hr_factor, max_transition=100):
    """
    Calculate Franck–Condon Factors for transitions 0-0 to 0-n.

    Parameters:
        hr_factor (float): Huang–Rhys Factor
        max_transition (int): Maximum transition level

    Returns:
        list: FCF values for transitions 0-0 to 0-n
    """
    fcf = [(hr_factor**n * np.exp(-hr_factor)) / math.factorial(n) for n in range(max_transition + 1)]
    return fcf

# Simulate the spectrum
spectrum = np.zeros_like(energy_range)

for _, row in data.iterrows():
    # Calculate FCFs for the current HR-factor
    fcf_values = calculate_fcf(row['HR-factor'], max_transition=4)
    
    # Add contributions from each transition to the spectrum
    for n, fcf in enumerate(fcf_values):
        peak_position = row['Freq'] + n * vibrational_spacing  # Energy of the n-th transition
        spectrum += fcf * np.exp(-((energy_range - peak_position)**2) / (2 * sigma**2))

# Normalize the spectrum
spectrum /= spectrum.max()

# Plot the simulated spectral profile
plt.figure(figsize=(10, 6))
plt.plot(energy_range, spectrum, label='Simulated Spectrum', color='blue')
plt.xlabel('Energy (cm⁻¹)', fontsize=16, fontweight='bold')
plt.ylabel('Intensity (Normalized)', fontsize=16, fontweight='bold')
plt.title('Simulated Spectral Profile Based on FCFs', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
```

### ORCA ESD Module
[replot](https://github.com/HenriqueCSJ/ORCASpectrumPlot)


## OVERSHOOT in Tail side of emission Spectra

In many vibronic simulations (for example, when calculating Franck–Condon factors for emission spectra) one finds that very low‐frequency modes often come with very large displacements. These modes are typically associated with large‐amplitude motions (such as torsions or soft deformations) that are not well described by a harmonic approximation. In ORCA, one practical remedy is to set the `TCUTFREQ` flag so that modes with frequencies below a certain value (in cm⁻¹) are removed from the calculation. In addition, one can examine the magnitude of the (dimensionless) displacements associated with these modes. Here’s how one can rationalize a choice for both thresholds.

## 1. Displacements and the Harmonic Approximation

For each normal mode, one may define a **dimensionless displacement parameter**, $d$, that is related to the actual coordinate displacement $\Delta Q$ and the zero‐point amplitude of that mode. A common definition is:

$$
d = \frac{\Delta Q}{\sqrt{\frac{\hbar}{2\mu\omega}}}
$$

where:

- $\mu$ is the **reduced mass** associated with the mode,
- $\omega$ is the **vibrational angular frequency**, and
- $\sqrt{\frac{\hbar}{2\mu\omega}}$ is the **zero‐point amplitude**.

The **Huang–Rhys factor** $S$ (which gives an idea of how strongly a particular mode couples to an electronic transition) is then:

$$
S = \frac{d^2}{2}.
$$

In many systems, a **Huang–Rhys factor** on the order of **1** (i.e., $d \approx 2$ or roughly $1.4$) already signals significant coupling. When $d$ becomes substantially larger than about **1.5** (in dimensionless units), the displacement is so large that the harmonic oscillator approximation may no longer be reliable. In practice, one might say that:

- **Displacements with** $d \gtrsim 1.5$ **are “too large”**, meaning the corresponding mode is likely to be highly anharmonic.
- This corresponds to **$S \gtrsim 1.1$**.
- Modes with such large dimensionless displacements contribute significantly to the vibrational progression and—in the harmonic treatment—can overly broaden the simulated spectral tails.


# 1. Low-Frequency Modes and Their Role

## A. Large Amplitude Motions and Anharmonicity

### **Large Zero-Point Amplitudes:**
For a given vibrational mode, the zero‐point amplitude is given by:

$$
Q_{\text{zpt}} = \sqrt{\frac{\hbar}{2\mu\omega}},
$$

where:

- $\mu$ is the **reduced mass** and  
- $\omega$ is the **angular frequency**.

When $\omega$ is very low (e.g., below **100 cm⁻¹**), $Q_{\text{zpt}}$ becomes large. This means that even modest displacements **$\Delta Q$** lead to very large **dimensionless displacements**:

$$
d = \frac{\Delta Q}{Q_{\text{zpt}}} = \Delta Q \sqrt{\frac{2\mu\omega}{\hbar}}.
$$

A small **physical displacement** translates into a **large $d$** when $\omega$ is small.

### **Impact on Franck–Condon Factors:**
The **Franck–Condon (FC) factors**, which dictate the **intensity distribution** in the emission spectrum, are given by:

$$
FC_{if} = \left| \langle \chi_f (g) | \chi_i (e) \rangle \right|^2.
$$

When the **dimensionless displacement** $d$ is large (commonly $d \gtrsim 1.5$), the corresponding **Huang–Rhys factor**:

$$
S = \frac{d^2}{2}
$$

becomes significant. A **high $S$** means that the **transition intensity** is spread over many vibrational quanta, thereby **broadening** the overall spectrum.

---

## B. Inadequacy of the Harmonic Approximation

### **Anharmonic Behavior:**
Low-frequency modes typically correspond to **soft, large-amplitude motions** (e.g., torsions, librations) that are **not well described by the harmonic oscillator model**. The **harmonic approximation** assumes a **parabolic potential energy surface**, but for these modes, the **true potential** is more **anharmonic**. Consequently, using the harmonic oscillator model leads to **overestimated FC factors** in the **tail region** of the spectrum.

### **Over-Broadening:**
Since these **low-frequency modes** are contributing **unphysically large displacements**, their **inclusion causes the simulated vibrational progression to be overly spread out**. The **tails of the emission spectrum**, which are **sensitive** to transitions involving **high vibrational quantum numbers**, are thus **overpopulated**, leading to an **overshoot relative to experimental observations**.

### **Critical Note:**
One must be cautious because some modes naturally have large displacements even when they are physically relevant. However, if these modes also have **very low frequencies**, then the combination suggests they are **“floppy”** and likely beyond the harmonic regime.

---

## 2. Choice of Frequency Threshold (`TCUTFREQ`)

Low-frequency modes are often problematic because:

- **Low-frequency (soft) modes** (e.g., those with frequencies below **100 cm⁻¹**) have very shallow potential energy surfaces.
- They tend to be **more sensitive to environmental effects** (e.g., solvent fluctuations) and usually display anharmonic behavior.
- Their vibrational quanta are **very small** (on the order of **0.012 eV** at **100 cm⁻¹**), so even modest displacements result in large relative changes.

Because of these reasons, many practitioners choose to **remove modes** below a certain cutoff frequency. Based on both the **literature** and **practical experience**, a `TCUTFREQ` value of **approximately 100 cm⁻¹** is a common choice. This value is justified because:

- Modes below **100 cm⁻¹** often represent **large-amplitude, low-energy motions** (such as torsions or collective deformations) that are not well described by a harmonic oscillator model.
- Removing these modes **prevents the simulation from artificially overpopulating the spectral tail** (i.e., extending intensity to energies far from the **0–0 transition**).

### **Critical Note:**
One must **balance** the removal of modes with the risk of excluding physically meaningful contributions. Setting `TCUTFREQ` too **high** may discard modes that, while soft, still play a role in the **vibronic structure**. Conversely, **too low** a threshold may leave in modes whose anharmonicity distorts the simulated spectrum.

---

## 3. Recommended Thresholds and Their Justification

### **a) Displacement Threshold:**
- **Threshold:** $d \gtrsim 1.5$ (in dimensionless normal coordinate units).
- **Justification:**
  - A dimensionless displacement of about **1.5** implies a **Huang–Rhys factor**:

$$
S \approx \frac{(1.5)^2}{2} \approx 1.125.
$$

  - Values of $S \gtrsim 1$ already indicate significant geometry change between the ground and excited states.
  - When $d$ **exceeds** 1.5, the **harmonic approximation becomes questionable**, and the vibrational progression (and hence spectral tails) may be **artificially broadened**.

### **b) Frequency Threshold (`TCUTFREQ`):**
- **Threshold:** `TCUTFREQ` set to **approximately 100 cm⁻¹**.
- **Justification:**
  - Modes with frequencies **below 100 cm⁻¹** are typically associated with **large-amplitude, anharmonic motions**.
  - Their **low energy quanta** mean that even a **modest displacement** produces a **significant change** in the Franck–Condon envelope, often leading to an **overestimation of the spectral tail intensity**.
  - **Empirical practice** in many computational studies supports the removal of modes in this region to improve the agreement between **theory and experiment**.

---

## 4. Final Remarks

While these threshold values **($d \gtrsim 1.5$ and `TCUTFREQ` ≈ 100 cm⁻¹)** are a **good starting point**, they should be regarded as **guidelines rather than strict rules**. The optimal choice may depend on:

- The **specific molecule** under study,
- The **nature of the electronic transition**,
- The **quality of the underlying potential energy surfaces**, and
- How **sensitive** the final spectral features are to **low-frequency contributions**.

**Critically**, one must always **validate these choices** by comparing **simulated spectra** with **experimental results**. If the **tail region** is **overshot** due to contributions from modes with **very large displacements** and **very low frequencies**, then **removing them** (via the `TCUTFREQ` flag) is justified. However, one should also **consider alternative treatments** (such as **anharmonic corrections**) if these modes are suspected to play a **genuine role in the dynamics**.

### **In summary:**
For many systems, it is advisable to **remove** modes with **frequencies below 100 cm⁻¹** (using `TCUTFREQ = 100 cm⁻¹`) **when their computed dimensionless displacements exceed approximately 1.5**. This practice **minimizes** the risk of **overestimating the spectral tails** due to **anharmonic low-frequency motions** while maintaining a balance between **physical realism** and **computational tractability**.

## The Orca command to calculate the emission spectrum
```text
! DEF2-SVP ESD(FLUOR)
%TDDFT
	NROOTS 1
	IROOT 1
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.07295
END
%ESD
  GSHESSIAN  "S0.hess"
  ESHESSIAN  "S1.hess"
  LINES      VOIGT
  LINEW      25
  INLINEW    50
  PrintLevel 4
  UNIT       NM
  TCUTFREQ   100
  IFREQFLAG  REMOVE
END
%maxcore 4000
%pal nprocs 16 end
* XYZFILE 0 1 EHBIPO0.07294535118797844.xyz


if doing any Numerical fgrequency calculation in particularly the excited state , if there is an error, remove those files associetd with it by running

```text
# List of filenames to match
patterns=(
    "S1_D00384"
 bla..bla..bla
)

# Loop through patterns and delete matching files
for pattern in "${patterns[@]}"; do
    find . -type f -name "*$pattern*" -exec rm -v {} +
done

echo "Deletion process completed."
```

and restart the calculation by adding this option

```text
%FREQ
	restart true
END
```


## plot the curve
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "EHBIPO.txt"
data = pd.read_csv(file_path, sep="\t", header=None, names=["Frequency (cm$^{-1}$)", "Huang-Rhys factor", "Reorganization Energy (cm$^{-1}$)"])

# Plot the data
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for Huang-Rhys factor (left Y-axis)
ax1.bar(data["Frequency (cm$^{-1}$)"], data["Huang-Rhys factor"], width=10, alpha=0.6, label="Huang-Rhys factor", color='b')
ax1.set_xlabel("Frequency (cm$^{-1}$)", fontsize=18, fontweight='bold')
ax1.set_ylabel("Huang-Rhys factor", fontsize=18, fontweight='bold', color='b')
ax1.tick_params(axis='y', labelcolor='b', labelsize=21)
ax1.tick_params(axis='x', labelsize=21)
ax1.set_xlim([0, 2000])
# Create second Y-axis for Reorganization Energy
ax2 = ax1.twinx()
ax2.bar(data["Frequency (cm$^{-1}$)"], data["Reorganization Energy (cm$^{-1}$)"], width=10, alpha=0.6, label="Reorganization Energy", color='r')
ax2.set_ylabel("Reorganization Energy (cm$^{-1}$)", fontsize=18, fontweight='bold', color='r')
ax2.tick_params(axis='y', labelcolor='r', labelsize=21)

# Title and show plot
plt.title("")
fig.tight_layout()
plt.show()
```



