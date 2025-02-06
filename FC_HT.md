# Understanding IntensityFC and IntensityHT in Vibrationally Resolved Spectroscopy

In vibrationally resolved spectroscopy, the intensity of individual vibrational transitions between electronic states is a critical factor that determines the appearance of absorption or fluorescence spectra. Two important contributions to these intensities are the Franck-Condon (FC) intensities ($\text{Intensity}_{\text{FC}}$) and Herzberg-Teller (HT) intensities ($\text{Intensity}_{\text{HT}}$). These terms arise from different physical origins and describe how vibrational and electronic interactions influence the spectrum.

This comprehensive guide delves into the theoretical foundations, equations, physical meaning, and practical applications of $\text{Intensity}_{\text{FC}}$ and $\text{Intensity}_{\text{HT}}$, with a focus on their implementation in quantum chemical software like ORCA.

---

## 1. Introduction to Vibrationally Resolved Spectroscopy

In electronic transitions (e.g., absorption or fluorescence), vibrational substructure often appears in the spectrum due to nuclear motion. The total intensity of a transition is influenced by:

- **The Franck-Condon principle**: Assumes that the electronic transition occurs so quickly that the nuclei remain stationary during the process.
- **The Herzberg-Teller coupling**: Accounts for the modulation of the electronic transition dipole moment by nuclear displacements.

The observed spectrum is a sum of individual vibrational transitions between quantum states of the molecule. Each vibrational transition has an intensity determined by:

- The overlap between vibrational wavefunctions in the initial and final electronic states ($\text{Intensity}_{\text{FC}}$).
- The effect of nuclear motion on the electronic transition dipole moment ($\text{Intensity}_{\text{HT}}$).

### Total Intensity Expression:

$$
I_{\text{total}} = I_{\text{FC}} + I_{\text{HT}}
$$

Where:

- $I_{\text{FC}}$: Contribution from the Franck-Condon factor.
- $I_{\text{HT}}$: Contribution from Herzberg-Teller coupling.

---

## 2. Franck-Condon Intensity ($\text{Intensity}_{\text{FC}}$)

### 2.1 Theoretical Framework

The Franck-Condon principle assumes that the nuclei do not move during the electronic transition because the excitation happens on a much shorter timescale than nuclear motion (Born-Oppenheimer approximation). The intensity is determined solely by the overlap of vibrational wavefunctions in the initial and final states.
![image](https://github.com/user-attachments/assets/3077d975-2ac2-48b1-a8ea-925cfca6d63e)

#### Mathematical Expression:

$$
I_{\text{FC}} \propto \left| \int \phi_i(r) \phi_f(r) \, dr \right|^2
$$

Where:

- $\phi_i(r)$: Vibrational wavefunction in the ground electronic state ($S_0$).
- $\phi_f(r)$: Vibrational wavefunction in the excited electronic state ($S_1$).
- $\int \phi_i(r) \phi_f(r) \, dr$: Overlap integral of vibrational wavefunctions.

The square of the overlap integral represents the **Franck-Condon factor**, which measures the alignment of vibrational wavefunctions between the ground and excited states.

---

### 2.2 Origin of Vibrational Progression

- **Vertical Transitions**: When the equilibrium geometry of the excited state differs from the ground state, nuclear displacement leads to a progression of vibrational peaks (vibrational substructure).
- **Harmonic Oscillator Approximation**: Under this approximation, vibrational wavefunctions are Hermite polynomials, and Franck-Condon factors can be computed analytically.

---

### 2.3 Key Assumptions

- The electronic transition dipole moment ($\mu$) is constant and does not depend on nuclear displacements.
- The intensity is entirely determined by nuclear overlap.

---

## 3. Herzberg-Teller Intensity ($\text{Intensity}_{\text{HT}}$)

### 3.1 Theoretical Framework

In the Herzberg-Teller framework, the transition dipole moment ($\mu$) is modulated by nuclear motion and can be expanded as a Taylor series around the equilibrium nuclear geometry:

$$
\mu(Q) = \mu_0 + \sum_k \left( \frac{\partial \mu}{\partial Q_k} \right) Q_k + \dots
$$

Where:

- $Q_k$: Normal mode displacement along vibrational mode $k$.
- $\mu_0$: Transition dipole moment at equilibrium geometry.
- $\frac{\partial \mu}{\partial Q_k}$: Derivative of the dipole moment with respect to nuclear displacement.

The HT intensity for a transition between vibrational states $i$ and $f$ is given by:

$$
I_{\text{HT}} \propto \left| \int \phi_i(r) \left( \frac{\partial \mu}{\partial Q_k} \right) \phi_f(r) \, dr \right|^2
$$

Where:

- $\phi_i(r)$ and $\phi_f(r)$: Vibrational wavefunctions (same as in the Franck-Condon case).
- $\frac{\partial \mu}{\partial Q_k}$: Modulates the transition intensity by coupling electronic and vibrational degrees of freedom.

---

### 3.2 Physical Interpretation

The HT contribution arises from the modulation of the electronic transition dipole moment by nuclear motion. This is particularly important when:

- The electronic transition is symmetry forbidden (e.g., $n \to \pi^*$ transitions).
- Charge transfer or symmetry breaking leads to large variations in the transition dipole moment with nuclear displacement.

---

### 3.3 Combined Contributions

The total intensity of a transition combines Franck-Condon and Herzberg-Teller contributions:

$$
I_{\text{total}} \propto \left| \mu_0 \int \phi_i(r) \phi_f(r) \, dr \right|^2 + \sum_k \left| \int \phi_i(r) \left( \frac{\partial \mu}{\partial Q_k} \right) \phi_f(r) \, dr \right|^2
$$

- The first term corresponds to Franck-Condon intensity ($I_{\text{FC}}$).
- The second term corresponds to Herzberg-Teller intensity ($I_{\text{HT}}$).

# 4. Differences Between IntensityFC and IntensityHT

| **Property**                | **Franck-Condon (IntensityFC)**              | **Herzberg-Teller (IntensityHT)**           |
|-----------------------------|---------------------------------------------|---------------------------------------------|
| **Dipole Dependence**       | Assumes a fixed transition dipole moment.  | Accounts for variation of dipole moment with nuclear displacement. |
| **Key Origin**              | Overlap of vibrational wavefunctions.      | Modulation of dipole moment by vibrational motion. |
| **Applications**            | Vibrational progressions in allowed electronic transitions. | Forbidden or weak electronic transitions (e.g., $n \to \pi^*$, charge-transfer states). |
| **Dominates When**          | The electronic transition is strong (allowed). | The electronic transition is weak (forbidden). |

---

# 5. Applications in ORCA

In **ORCA**, vibrationally resolved spectra include both **IntensityFC** and **IntensityHT** contributions. The `DOHT` keyword in the `%ESD` block controls whether Herzberg-Teller coupling is included in the calculation:

- `DOHT TRUE`: Includes Herzberg-Teller contributions (**IntensityHT**).
- `DOHT FALSE`: Ignores Herzberg-Teller coupling and uses only Franck-Condon factors (**IntensityFC**).

---

## Output in ORCA

ORCA provides a detailed breakdown of **IntensityFC** and **IntensityHT** in the output file. For each vibrational transition, the following information is typically reported:

| **Transition** | **Initial Vibrational State (S₀)** | **Final Vibrational State (S₁)** | **IntensityFC** | **IntensityHT** | **Total Intensity** |
|----------------|------------------------------------|-----------------------------------|------------------|------------------|----------------------|
| 1              | 0 → 0                             | 0 → 0                            | 0.85            | 0.01            | 0.86                 |
| 2              | 0 → 1                             | 0 → 1                            | 0.05            | 0.02            | 0.07                 |

---

## Visualization

The final spectrum combines both contributions to produce the vibrationally resolved electronic spectrum.

---

# 6. Conclusion

Both **IntensityFC** and **IntensityHT** play crucial roles in determining the appearance of vibrationally resolved electronic spectra:

- **IntensityFC** dominates for strong, allowed transitions and is purely based on vibrational overlap.
- **IntensityHT** is significant for weak or forbidden transitions and arises from the coupling between vibrational motion and the electronic dipole moment.

By including both contributions, software like ORCA provides accurate spectra that capture both Franck-Condon and Herzberg-Teller effects, enabling detailed analysis of experimental results and theoretical predictions.

---

