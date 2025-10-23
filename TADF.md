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

![image](https://github.com/user-attachments/assets/e1b0ae9c-18e5-4895-878e-b028f9479d3f)


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

![image](https://github.com/user-attachments/assets/9bab9421-5402-41b1-83cd-9de7d0652ee6)


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

# Below is a step-by-step derivation of the three-state spin–vibronic Hamiltonian that underpins TADF

## 1 Start from the exact molecular Hamiltonian

![image](https://github.com/user-attachments/assets/75cb96fe-1487-4bf3-847c-23e3c9e0b593)


- $Q_\alpha$ – dimensionless normal-mode coordinates  
- $\hat{T}_\text{nuc}$ – nuclear kinetic energy  
- $\hat{H}_\text{el}$ – electronic Coulomb Hamiltonian, parameterised by $Q$  
- $\hat{H}_\text{SO}$ – one-electron spin–orbit operator (first order in $c^{-2}$)

---

## 2 Born–Huang expansion and projection onto a diabatic subspace

Write the exact wavefunction as:

$$
\Psi(r, Q) = \sum_I \Phi_I(r; Q)\, \chi_I(Q)
$$

Then project onto three diabatic electronic states that are most relevant for TADF:

- $\lvert S_\text{CT} \rangle$ – lowest singlet, charge-transfer character  
- $\lvert T_\text{CT} \rangle$ – lowest triplet, same CT character  
- $\lvert T_\text{LE} \rangle$ – nearby triplet, local-excitation character

The diabatic choice fixes the electronic basis so that each state’s character (CT vs LE, singlet vs triplet) is preserved as the nuclei move. Inside this subspace we obtain a nuclear-space Hamiltonian matrix:

![image](https://github.com/user-attachments/assets/f8e0fe4b-3db1-4024-a406-ec3cb1864deb)


Where the derivative (non-adiabatic) coupling is:

$$
F_{IJ}^\alpha(Q) = \langle I \lvert \frac{\partial}{\partial Q_\alpha} \rvert J \rangle
$$

---

## 3 Interpret each block of the matrix

### 3.1 Diagonal elements – adiabatic potentials

$$
E_I(Q) = \langle I \lvert \hat{H}_\text{el} \rvert I \rangle
$$

These populate the (1,1), (2,2), and (3,3) entries and represent the potential-energy surfaces for $S_\text{CT}$, $T_\text{CT}$, and $T_\text{LE}$ respectively.

### 3.2 Off-diagonal elements from spin–orbit coupling (SOC)

The one-electron Breit–Pauli operator in atomic units is:

$$
\hat{H}_\text{SO} = \sum_i \frac{1}{2c^2} \frac{1}{r_i} \frac{\partial V}{\partial r_i} \, \hat{L}_i \cdot \hat{S}_i
$$

It changes spin by ±1 but leaves spatial symmetry unchanged.

Only the singlet–triplet pair with identical spatial character satisfies the Wigner–Eckart rules in first order. Hence the sole non-zero matrix element at first order is:

![image](https://github.com/user-attachments/assets/9224c4c6-3e3d-4041-86cc-b61b70966cd0)


This goes into the (1,2) entry; its Hermitian conjugate fills the (2,1) entry.

SOC between $S_\text{CT}$ and $T_\text{LE}$ is zero in first order because it would simultaneously change spin and spatial configuration.

### 3.3 Off-diagonal elements from vibronic (non-adiabatic) coupling

Analytically expanding the derivative coupling to first order in normal modes gives a linear vibronic operator:

![image](https://github.com/user-attachments/assets/895267ff-29b1-40e0-8db3-16140274949b)


Vibronic coupling preserves total spin (nuclear momentum carries no spin quantum number). Therefore the leading non-zero term couples the two triplets:

![image](https://github.com/user-attachments/assets/c3a24e5a-8105-4ba6-900c-970092cb522f)


Which sits in (2,3) and (3,2).

Singlet–triplet vibronic terms are zero to first order, leaving (1,3) and (3,1) empty.

---

## 4 Assemble the $3 \times 3$ Hamiltonian

Ordering the basis as $(S_\text{CT}, T_\text{CT}, T_\text{LE})$, one obtains:

$$
\hat{H}(Q) =
\begin{pmatrix}
E_{S_\text{CT}}(Q) & H_\text{SO}(1,2) & 0 \\
H_\text{SO}(1,2)^* & E_{T_\text{CT}}(Q) & V_{23}(Q) \\
0 & V_{23}^*(Q) & E_{T_\text{LE}}(Q)
\end{pmatrix}
$$

Zero elements are enforced by the combination of spin selection (SOC) and orbital selection (vibronic coupling).

The matrix is Hermitian because each physical operator is Hermitian and we retain Hermitian conjugates explicitly.

---

## 5 Second-order “super-exchange” path

Because the direct singlet–LE-triplet coupling is zero, communication proceeds by a two-step pathway:

![image](https://github.com/user-attachments/assets/bfd46234-7aaf-4eb7-b9cb-68e75248bc66)


Löwdin partitioning (or a canonical Schrieffer–Wolff transformation) projects out $T_\text{CT}$ and yields an effective singlet–triplet interaction:

$$
H_\text{eff}(1,3) = \frac{H_\text{SO}(1,2) \, V_{23}}{E_{S_\text{CT}} - E_{T_\text{CT}}} + \text{h.c.}
$$

That second-order term is the microscopic origin of the fast reverse inter-system crossing (RISC) observed in efficient TADF molecules: even though each first-order operator is small, their product divided by a modest energy denominator can be large enough to drive:

$$
k_\text{RISC} \sim 10^{6-7} \, \text{s}^{-1}
$$

---

## 6 Why every element is exactly where it is

| Matrix position | Operator responsible | Selection rule (reason for being non-zero) |
|-----------------|----------------------|--------------------------------------------|
| (1,1) / (2,2) / (3,3) | $\hat{H}_\text{el}$ | Trivial: diagonal energies |
| (1,2) & (2,1)         | $\hat{H}_\text{SO}$ | Spin change $S = 0 \leftrightarrow 1$ with same spatial symmetry |
| (2,3) & (3,2)         | $\hat{H}_\text{vib}$ | Spin conserved ($S=1$), different orbital character |
| (1,3) & (3,1)         | None (0) | Would need simultaneous spin flip and orbital change – forbidden at first order |

---

## Quick sanity check

- Set $H_\text{SO}(1,2) = 0$ ⇒ no first-order ISC/RISC possible – the system reduces to two uncoupled blocks.  
- Set $V_{23} = 0$ ⇒ SOC alone can shuffle population between $S_\text{CT}$ and $T_\text{CT}$, but no pathway connects to $T_\text{LE}$ – delayed fluorescence slows dramatically.  
- Keep both non-zero ⇒ the experimentally observed fast, Arrhenius-activated RISC emerges.

---

## Bottom line:

The positions of every non-zero element in the multistate vibronic Hamiltonian follow directly from:

1. The spin multiplicity of the electronic states  
2. El-Sayed’s rules for SOC  
3. The fact that the nuclear kinetic energy operator carries no spin  

Once these rules are laid out, the $3 \times 3$ matrix in the blog post is not an *ansatz* but an unambiguous, symmetry-dictated reduction of the exact Hamiltonian.


<img width="1300" height="492" alt="image" src="https://github.com/user-attachments/assets/ff0b41cb-764e-42d0-ad5d-d580673a1e41" />
<img width="950" height="825" alt="image" src="https://github.com/user-attachments/assets/8ad86a55-d996-4ff4-b7e8-5348790b05f7" />


see https://pubmed.ncbi.nlm.nih.gov/31841330/
