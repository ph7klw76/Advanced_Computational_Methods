# Quantum‐Mechanical Underpinnings of Thermally Activated Delayed Fluorescence (TADF)

## Abstract

Thermally activated delayed fluorescence (TADF) revolutionises exciton utilisation in organic optoelectronics by converting non‑emissive triplet excitons into emissive singlets without heavy metals.  
This blog offers a rigorous, quantum‑mechanical treatment of the phenomenon, weaving together spin–orbit, vibronic and environmental effects within a unified rate‑theory framework.  
Readers will find a bridge between first‑principles Hamiltonians, kinetic models, computational protocols and experimental observables.

## Table of Contents

- [Motivation and Historical Context](#1--motivation-and-historical-context)
- [Electronic States & Energy Landscape](#2--electronic-states--energy-landscape)
- [Multistate Vibronic Hamiltonian](#3--multistate-vibronic-hamiltonian)
- [Spin–Orbit and Spin–Vibronic Coupling](#4--spinorbit-and-spin–vibronic-coupling)
- [Rate Theories](#5--rate-theories-fermi-marcus-levichjortner)
- [Computational Tool‑Kit](#6--computational-toolkit)
- [Experimental Signatures of TADF](#7--experimental-signatures)
- [Molecular Design Strategies](#8--molecular-design-strategies)
- [Frontier Challenges & Emerging Concepts](#9--frontier-challenges--emerging-concepts)
- [Concluding Remarks](#10--concluding-remarks)
- [Selected References](#selected-references)

---

## 1  Motivation and Historical Context

Organic light‑emitting diodes (OLEDs) are intrinsically limited by spin statistics: electrical excitation yields 25 % singlets and 75 % triplets.  
Phosphorescent emitters solve this with heavy metals but at the cost of scarcity and stability.  
TADF, first demonstrated in 2012 by Adachi’s group, circumvents the bottleneck by thermal up‑conversion of triplets to singlets, enabling 100 % internal quantum efficiency (IQE) with earth‑abundant elements.

---

## 2  Electronic States & Energy Landscape

### 2.1  Charge‑Transfer (CT) Versus Local Excitation (LE)

TADF scaffolds adopt donor‑acceptor (D–A) architectures where:

- HOMO localises on the donor  
- LUMO on the acceptor  

The resulting CT excited states feature:

- Small spatial overlap → reduced exchange integral → small singlet–triplet gap  
- Long radiative lifetime due to weak oscillator strength — later mitigated by hybridising with LE states.

### 2.2  Exchange‑Energy Expression

A practical design rule is:

$$
\Delta E_{\text{ST}} \lesssim k_B T \approx 25\, \text{meV at } 300\,\text{K}
$$

---

## 3  Multistate Vibronic Hamiltonian

A minimal Hamiltonian capturing TADF comprises the lowest CT singlet ($^1\text{CT}$), CT triplet ($^3\text{CT}$) and a nearby LE triplet ($^3\text{LE}$):

$$
\hat{H} = \hat{H}_\text{el} + \hat{H}_\text{vib} + \hat{H}_\text{SO} + \hat{H}_\text{NA}
$$

- $\hat{H}_\text{SO}$: one‑electron spin–orbit operator, first‑order in relativistic perturbation.  
- $\hat{H}_\text{NA}$: non‑adiabatic vibronic coupling mediating internal conversion between CT and LE triplets.  
- $\hat{H}_\text{vib}$: Franck–Condon active vibrational modes.

Diagonalisation within a Born–Huang expansion yields spin‑mixed vibronic states enabling efficient intersystem crossing (ISC) and reverse ISC (RISC).

---

## 4  Spin–Orbit and Spin–Vibronic Coupling

### 4.1  Direct Spin–Orbit (El‑Sayed) Mechanism

For pure $^3\text{CT} \leftrightarrow ^1\text{CT}$ transitions, one‑electron $\hat{H}_\text{SO}$ is small ($\sim 0$).  
El‑Sayed’s rules thus disfavour direct ISC.

### 4.2  Second‑Order Spin–Vibronic Pathway

A higher‑order perturbative term couples states via an intermediate LE configuration:

$$
\langle ^1\text{CT} | \hat{H}_\text{SO} \frac{1}{E - \hat{H}_0} \hat{H}_\text{NA} | ^3\text{LE} \rangle
$$

Modest LE–CT energy offsets ($\Delta E_{\text{LE–CT}}$) maximise this term.  
This “hybridisation funnel” underpins record RISC rates up to $\sim 10^6$ s$^{-1}$.

---

## 5  Rate Theories (Fermi, Marcus, Levich–Jortner)

### 5.1  Fermi’s Golden Rule

$$
k = \frac{2\pi}{\hbar} |\langle f | \hat{H} | i \rangle|^2 \rho(E)
$$

where $\rho(E)$ is the Franck–Condon weighted density of states.

### 5.2  Marcus–Levich–Jortner Expression for RISC

$$
k = \frac{2\pi}{\hbar} |H_{\text{eff}}|^2 \sum_{v=0}^\infty \frac{e^{-S} S^v}{v!} \frac{1}{\sqrt{4\pi \lambda k_BT}} \exp\left[ -\frac{(\Delta G + \lambda + v \hbar \omega)^2}{4\lambda k_BT} \right]
$$

**Key Parameters**

| Symbol | Typical Range | Design Lever |
|--------|----------------|------------------|
| $\Delta E_{\text{ST}}$ | 10–100 meV | D–A torsion angle |
| $\lambda$ | 150–350 meV | Backbone rigidity |
| $H_{\text{SO}}^{\text{eff}}$ | 0.1–5 cm$^{-1}$ | LE admixture, heteroatoms |

### 5.3  Master Equation

Coupling the singlet and triplet reservoirs:

$$
\frac{d[S]}{dt} = -k_r[S] + k_{\text{RISC}}[T], \quad \frac{d[T]}{dt} = -k_{\text{RISC}}[T]
$$

Closed‑form solutions reproduce bi‑exponential decays observed in time‑resolved photoluminescence.

---

## 6  Computational Tool‑Kit

| Phenomenon | Recommended Level | Notes |
|------------|-------------------|-------|
| $S_1$, $T_1$, $T_2$ | TD‑DFT (wpbeh, optimally tuned range‑separated) | Validate with ADC(2) or EOM‑CCSD |
| SOC Matrix Elements | Breit–Pauli one‑electron SOC within TD‑DFT or quadratic response | Spin–vibronic coupling via PySOC, Q‑Chem |
| FC & Huang–Rhys factors | DFT normal‑mode analysis | Needed for Marcus/MLJ rates |
| Non‑adiabatic dynamics | Surface hopping (SHARC, PYXAID) | Captures temperature‑dependent RISC |

---

## 7  Experimental Signatures

- **Transient PL**: Bimodal decay with prompt ($\tau_p$) and delayed ($\tau_d$) components  
- **Temperature Dependence**: Arrhenius plot of delayed yield gives activation energy $\approx \Delta E_{\text{ST}}$  
- **Magneto‑EL/PL**: Triplet harvesting suppressed by magnetic field (Zeeman effect)  
- **Thermally Stimulated Current Spectroscopy**: Probes trap‑mediated reverse crossing

---

## 8  Molecular Design Strategies

- **Twisted D–A Conformation** for small $\Delta E_{\text{ST}}$  
- **Multi‑Resonance Emitters (MRE)**: Rigid frameworks, FWHM < 30 nm  
- **Through‑Space Charge Transfer**: “U‑shaped” or “spiro” linkers  
- **Heavy‑Atom‑Assisted TADF**: Halogens/chalcogens increase SOC while preserving organic character

---

## 9  Frontier Challenges & Emerging Concepts

- **Hot Exciton & Up‑Conversion TADF**: Leveraging higher triplet states ($T_n$)  
- **Hyperfluorescence**: TADF sensitiser + fluorescent dopant → high IQE & colour purity  
- **Blue TADF Stability**: Mitigating exciton‑polaron annihilation  
- **Device Physics**: Charge balance, exciton diffusion and optical out‑coupling affect EQE

---

## 10  Concluding Remarks

TADF epitomises how molecular‑level quantum engineering translates into macro‑scale device breakthroughs.  
Developing a predictive, multiscale modelling pipeline—from relativistic perturbation theory through kinetic Monte Carlo—is vital for pushing efficiencies beyond today’s benchmarks.

---

# Derivation of the Three-State Spin–Vibronic Hamiltonian Underpinning TADF

## 1 Start from the Exact Molecular Hamiltonian

$$
\hat{H}_{\text{tot}} =
\underbrace{-\sum_\alpha \frac{\hbar^2}{2 M_\alpha} \frac{\partial^2}{\partial Q_\alpha^2}}_{\hat{T}_{\text{nuc}}}
+ \underbrace{\hat{H}_{\text{el}}(r; Q)}_{\text{Coulomb}}
+ \underbrace{\hat{H}_{\text{SO}}(r)}_{\text{Breit–Pauli}}
$$

Where:

- $Q_\alpha$: dimensionless normal-mode coordinates  
- $\hat{T}_{\text{nuc}}$: nuclear kinetic energy  
- $\hat{H}_{\text{el}}$: electronic Coulomb Hamiltonian (parametrised by $Q$)  
- $\hat{H}_{\text{SO}}$: one-electron spin–orbit operator (first order in $c^{-2}$)

---

## 2 Born–Huang Expansion and Projection onto a Diabatic Subspace

Express the exact wavefunction as:

$$
\Psi(r, Q) = \sum_I \Phi_I(r; Q)\, \chi_I(Q)
$$

Project onto three diabatic electronic states relevant for TADF:

- $\lvert S_{\text{CT}} \rangle$: lowest singlet, charge-transfer character  
- $\lvert T_{\text{CT}} \rangle$: lowest triplet, same CT character  
- $\lvert T_{\text{LE}} \rangle$: nearby triplet, local-excitation character  

The projected nuclear-space Hamiltonian matrix becomes:

$$
H_{IJ}(Q) =
\langle I \lvert \hat{H}_{\text{el}} \rvert J \rangle +
\langle I \lvert \hat{H}_{\text{SO}} \rvert J \rangle -
\sum_\alpha \frac{\hbar^2}{2 M_\alpha} \left[
F^\alpha_{IJ}(Q)\, \frac{\partial}{\partial Q_\alpha} +
\delta_{IJ} \frac{\partial^2}{\partial Q_\alpha^2}
\right]
$$

Where the non-adiabatic (derivative) coupling is:

$$
F^\alpha_{IJ}(Q) = \langle I \lvert \frac{\partial}{\partial Q_\alpha} \rvert J \rangle
$$

---

## 3 Interpret Each Block of the Matrix

### 3.1 Diagonal Elements – Adiabatic Potentials

$$
E_I(Q) = \langle I \lvert \hat{H}_{\text{el}} \rvert I \rangle
$$

These occupy the (1,1), (2,2), and (3,3) entries for $S_{\text{CT}}$, $T_{\text{CT}}$, and $T_{\text{LE}}$, respectively.

---

### 3.2 Off-Diagonal Elements from Spin–Orbit Coupling (SOC)

The one-electron Breit–Pauli SOC operator:

$$
\hat{H}_{\text{SO}} = \sum_i \frac{1}{2 c^2} \frac{1}{r_i} \frac{\partial V}{\partial r_i} \hat{L}_i \cdot \hat{S}_i
$$

- Changes spin by ±1 but preserves spatial symmetry  
- Only non-zero matrix element at first order:

$$
H_{\text{SO}}(1,2) = \langle S_{\text{CT}} \lvert \hat{H}_{\text{SO}} \rvert T_{\text{CT}} \rangle
$$

- Appears in (1,2) and (2,1)  
- SOC between $S_{\text{CT}}$ and $T_{\text{LE}}$ is zero (spin + orbital change forbidden)

---

### 3.3 Off-Diagonal Elements from Vibronic (Non-Adiabatic) Coupling

Expanding to first order in normal modes:

$$
\hat{H}_{\text{vib}} = \sum_\alpha \lambda_\alpha^{(IJ)} Q_\alpha
$$

With:

$$
\lambda_\alpha^{(IJ)} = \sqrt{\frac{\hbar}{2 M_\alpha \omega_\alpha}} \langle I \lvert \frac{\partial \hat{H}_{\text{el}}}{\partial Q_\alpha} \rvert J \rangle
$$

- Vibronic coupling conserves spin  
- Leading term couples $T_{\text{CT}}$ and $T_{\text{LE}}$:

$$
V_{23}(Q) = \langle T_{\text{CT}} \lvert \hat{H}_{\text{vib}} \rvert T_{\text{LE}} \rangle
$$

- Appears in (2,3) and (3,2)  
- No coupling between $S_{\text{CT}}$ and $T_{\text{LE}}$ at first order

---

## 4 Assemble the $3 \times 3$ Hamiltonian

Ordering: $(S_{\text{CT}}, T_{\text{CT}}, T_{\text{LE}})$

$$
\hat{H}(Q) =
\begin{pmatrix}
E_{S_{\text{CT}}}(Q) & H_{\text{SO}}(1,2) & 0 \\
H_{\text{SO}}^*(1,2) & E_{T_{\text{CT}}}(Q) & V_{23}(Q) \\
0 & V_{23}^*(Q) & E_{T_{\text{LE}}}(Q)
\end{pmatrix}
$$

- Zeros enforced by spin and orbital selection rules  
- The matrix is Hermitian

---

## 5 Second-Order “Super-Exchange” Path

Direct $S_{\text{CT}} \leftrightarrow T_{\text{LE}}$ coupling is forbidden.  
However, a two-step pathway exists:

$$
S_{\text{CT}} \xrightarrow{\hat{H}_{\text{SO}}} T_{\text{CT}} \xrightarrow{\hat{H}_{\text{vib}}} T_{\text{LE}}
$$

Löwdin partitioning yields the effective interaction:

$$
H_{\text{eff}}(1,3) = \frac{H_{\text{SO}}(1,2)\, V_{23}}{E_{S_{\text{CT}}} - E_{T_{\text{CT}}}} + \text{h.c.}
$$

This term underlies efficient RISC, allowing:

$$
k_{\text{RISC}} \sim 10^6 - 10^7 \ \text{s}^{-1}
$$

---

## 6 Why Every Element Is Exactly Where It Is

| Matrix Position | Operator Responsible | Selection Rule |
|-----------------|----------------------|----------------|
| (1,1), (2,2), (3,3) | $\hat{H}_{\text{el}}$ | Diagonal energies |
| (1,2), (2,1) | $\hat{H}_{\text{SO}}$ | Spin flip ($S = 0 \leftrightarrow 1$), same orbital |
| (2,3), (3,2) | $\hat{H}_{\text{vib}}$ | Spin-conserved, orbital change |
| (1,3), (3,1) | *None* | Forbidden: spin + orbital change |

---

###  Quick Sanity Check

- Set $H_{\text{SO}}(1,2) = 0$ ⇒ No first-order ISC/RISC  
- Set $V_{23} = 0$ ⇒ No path to $T_{\text{LE}}$, delayed fluorescence slows  
- Keep both non-zero ⇒ Efficient, Arrhenius-activated RISC

---

##  Bottom Line

The structure of the multistate vibronic Hamiltonian is **not an ansatz**.  
It follows unambiguously from:

1. The spin multiplicity of the states  
2. El-Sayed’s rules for SOC  
3. The spinlessness of the nuclear kinetic operator  

Thus, the $3 \times 3$ matrix in the blog post is a **symmetry-dictated reduction** of the full molecular Hamiltonian.


