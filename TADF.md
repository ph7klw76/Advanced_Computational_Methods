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

## Selected References

*(To be inserted based on citations and literature used.)*
