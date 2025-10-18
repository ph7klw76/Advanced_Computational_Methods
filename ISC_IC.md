# Intersystem Crossing Rates, Internal Conversion Rates, Nonadiabatic Coupling Matrix Elements, and Spin-Orbit Matrix Elements in Organic Emitters

In organic photophysics, the processes of intersystem crossing (ISC), internal conversion (IC), and their underlying mechanisms—mediated by nonadiabatic coupling matrix elements (NACMEs) and spin-orbit coupling (SOC)—play fundamental roles in controlling the efficiency of light-emitting devices, particularly in organic light-emitting diodes (OLEDs) and thermally activated delayed fluorescence (TADF) systems. Understanding these rates and coupling mechanisms is essential for designing efficient organic emitters.

This blog explores the theoretical framework, relevant equations, and computational implementation of ISC, IC, NACMEs, and SOC in ORCA, one of the most popular quantum chemistry packages for studying photophysical phenomena.

---

## 1. Introduction to Nonradiative Processes in Organic Molecules

### 1.1 Intersystem Crossing (ISC)

![image](https://github.com/user-attachments/assets/e51a5bed-a014-4e17-ad5e-c8a7143995db)


ISC refers to the radiationless transition between electronic states of different spin multiplicity, e.g., from a singlet excited state ($S_1$) to a triplet excited state ($T_1$). It is mediated by spin-orbit coupling (SOC), which mixes the wavefunctions of singlet and triplet states, enabling such transitions despite their otherwise spin-forbidden nature.

**Key Applications**:
- **OLEDs**: Efficient ISC from $S_1 \to T_1$ promotes triplet harvesting in phosphorescent emitters.
- **TADF**: Reverse ISC (RISC) from $T_1 \to S_1$ enhances emission in TADF materials.

---

### 1.2 Internal Conversion (IC)

Internal conversion involves a nonradiative transition between electronic states of the same spin multiplicity (e.g., $S_1 \to S_0$). IC occurs via vibronic coupling, mediated by nonadiabatic coupling matrix elements (NACMEs), which account for the mixing of electronic and nuclear motions.

**Key Applications**:
- **Nonradiative Loss**: IC can quench fluorescence by promoting relaxation to the ground state.
- **TADF Materials**: Competition between IC and RISC governs TADF efficiency.

---

### 1.3 Nonadiabatic Coupling Matrix Elements (NACMEs)

NACMEs quantify the strength of coupling between two electronic states due to nuclear motion. They play a critical role in internal conversion (IC) and dictate how effectively a molecule can transition between electronic states of the same spin multiplicity.

---

### 1.4 Spin-Orbit Coupling (SOC)

SOC represents the interaction between an electron’s spin and its orbital angular momentum. In the context of ISC, SOC allows spin-forbidden transitions (e.g., $S_1 \to T_1$) by mixing singlet and triplet wavefunctions.

---

## 2. Theoretical Framework

### 2.1 Intersystem Crossing (ISC) and Spin-Orbit Coupling

#### Rate of Intersystem Crossing

The rate of ISC between an initial state $\vert S \rangle$ (singlet) and a final state $\vert T \rangle$ (triplet) is given by Fermi’s Golden Rule:

$$
k_{ISC} = \frac{2\pi}{\hbar} \vert \langle T \vert \hat{H}_{SO} \vert S \rangle \vert^2 \rho(E),
$$

where:
- $\hat{H}_{SO}$ is the spin-orbit Hamiltonian.
- $\langle T \vert \hat{H}_{SO} \vert S \rangle$ is the spin-orbit coupling matrix element (SOCME).
- $\rho(E)$ is the density of vibrational states at the crossing point.

---

#### Spin-Orbit Coupling Matrix Element (SOCME)

The SOCME is defined as:

$$
\langle T \vert \hat{H}_{SO} \vert S \rangle = \sum_i \xi_i \langle \psi_T \vert \mathbf{L}_i \cdot \mathbf{S}_i \vert \psi_S \rangle,
$$

where:
- $\mathbf{L}_i$ and $\mathbf{S}_i$ are the orbital and spin angular momentum operators, respectively.
- $\xi_i$: Spin-orbit coupling constant, which depends on the atomic number $Z$ of heavy atoms present (e.g., $\xi \propto Z^2$).

In ORCA, SOCMEs are computed using spin-orbit integrals between wavefunctions of singlet and triplet states.

---

#### Factors Affecting ISC Rates

1. **Spin-Orbit Coupling (SOC)**:
   - Strong SOC enhances ISC rates. Molecules with heavy atoms (e.g., Ir, Pt) exhibit strong SOC.
   
2. **Energy Gap ($\Delta E_{ST}$)**:
   - Smaller singlet-triplet gaps promote ISC by enhancing vibronic coupling.

3. **Overlap of Wavefunctions**:
   - Orbital symmetry and configuration interaction (CI) between $S_1$ and $T_1$ states affect $\langle T \vert \hat{H}_{SO} \vert S \rangle$.

# A. Understanding the SOCME: ⟨T ∣ 𝐻̂_SO ∣ S⟩

The spin-orbit coupling matrix element (SOCME), ⟨T ∣ 𝐻̂_SO ∣ S⟩, is a key term in the rate of intersystem crossing (ISC) as described by Fermi’s Golden Rule:

$$
k_{ISC} = \frac{2\pi}{\hbar} \vert \langle T \vert \hat{H}_{SO} \vert S \rangle \vert^2 \rho(E),
$$

where:
- ∣S⟩: The singlet excited state wavefunction ($S_1$).
- ∣T⟩: The triplet excited state wavefunction ($T_1$).
- $𝐻̂_{SO}$: The spin-orbit Hamiltonian, which couples the spin and orbital angular momenta.
- $\langle T \vert 𝐻̂_{SO} \vert S \rangle$: The spin-orbit coupling between $S_1$ and $T_1$.
- $\rho(E)$: The density of vibrational states at the crossing point.

The SOCME depends on:
1. The molecular orbital (MO) symmetry and overlap between the $S_1$ and $T_1$ states.
2. The configuration interaction (CI) of $S_1$ and $T_1$ states, which determines how the wavefunctions of these states are constructed.
3. Spin-orbit coupling constants, which are determined by atomic properties (e.g., heavier atoms exhibit stronger SOC).

Let’s explore the role of orbital symmetry and CI in shaping ⟨T ∣ 𝐻̂_SO ∣ S⟩.

---

## B. Orbital Symmetry and Its Effect on ⟨T ∣ 𝐻̂_SO ∣ S⟩

### B.1 Molecular Orbital Contributions to $S_1$ and $T_1$

Both $S_1$ and $T_1$ states arise from electronic excitations between molecular orbitals (MOs). For example:
- $S_1$: Singlet state where one electron is excited from a HOMO (Highest Occupied Molecular Orbital) to a LUMO (Lowest Unoccupied Molecular Orbital) with paired spin.
- $T_1$: Triplet state with a similar excitation but with unpaired spin.

**Orbital Symmetry in $S_1$ and $T_1$:**
- For ⟨T ∣ 𝐻̂_SO ∣ S⟩ to be non-zero, the symmetries of the MOs involved in the $S_1$ and $T_1$ states must allow overlap when acted upon by the spin-orbit operator $𝐻̂_{SO}$.
- Specifically, the spin-orbit operator couples states where the spatial symmetries of the MOs in $S_1$ and $T_1$ differ by angular momentum.

**Example: π–π* vs. n–π***
- **Strong SOC (allowed by symmetry):**
  - If $S_1$ involves a π–π* excitation and $T_1$ involves a π–π* excitation (similar orbital symmetry), SOC may be weak because of limited angular momentum coupling.
  - However, if $S_1$ involves an n–π* excitation (from a non-bonding orbital to a π* orbital), SOC with $T_1$ (π–π*) can be strong because n–π* and π–π* differ in angular momentum, enhancing coupling.
- **Weak SOC (forbidden by symmetry):**
  - If $S_1$ and $T_1$ arise from completely orthogonal orbital configurations with incompatible symmetries, ⟨T ∣ 𝐻̂_SO ∣ S⟩ may be small or zero.

---

### B.2 Effect of Overlap Between Molecular Orbitals

The magnitude of ⟨T ∣ 𝐻̂_SO ∣ S⟩ depends on the overlap between the molecular orbitals (MOs) involved in the $S_1$ and $T_1$ transitions. Strong overlap enhances the spin-orbit coupling and ISC rate. For example:
- $S_1$: HOMO → LUMO excitation.
- $T_1$: HOMO → LUMO excitation.

If the HOMO and LUMO have significant spatial overlap, the SOCME will be enhanced.

---

## C. Configuration Interaction (CI) and Its Effect on ⟨T ∣ 𝐻̂_SO ∣ S⟩

### C.1 Configuration Interaction in Excited States

In quantum chemistry, the excited states ($S_1$, $T_1$) are not purely single excitations from one MO to another but are instead linear combinations of multiple electronic configurations. This mixing is called configuration interaction (CI) and arises naturally in methods like CASSCF (Complete Active Space SCF) and TD-DFT (Time-Dependent DFT).

The wavefunctions for $S_1$ and $T_1$ can be written as:

$$
\vert S_1 \rangle = \sum_i c_i \vert \psi_i \rangle,
$$

$$
\vert T_1 \rangle = \sum_j d_j \vert \psi_j \rangle,
$$

where:
- $\vert \psi_i \rangle$: Individual electronic configurations.
- $c_i, d_j$: CI coefficients for the $i$-th and $j$-th configurations.

---

### C.2 Role of CI in SOCME

The SOCME, ⟨T ∣ 𝐻̂_SO ∣ S⟩, depends on how much the configurations in $S_1$ and $T_1$ overlap:

![image](https://github.com/user-attachments/assets/7400b013-0272-4968-8d18-4afc99927cb9)


**Key Factors:**
1. **Orbital Mixing via CI:**
   - If $S_1$ and $T_1$ share significant CI contributions (i.e., $c_i$ and $d_j$ are large for overlapping configurations), ⟨T ∣ 𝐻̂_SO ∣ S⟩ increases.

2. **Intermediate Configurations:**
   - SOC is often enhanced by intermediate configurations that mix singlet and triplet states through $𝐻̂_{SO}$.
   - For example, in $S_1 \to T_1$, if both states share intermediate $n–π*$ or $π–π*$ configurations, SOCME is strengthened.

3. **Symmetry Relaxation via CI:**
   - CI can mix configurations of different symmetry, making spin-forbidden transitions (e.g., $S_1 \to T_1$) partially allowed.

---

## D. Practical Implications for Organic Emitters

In organic emitters, the ISC efficiency depends critically on:

1. **Orbital Symmetry:**
   - Molecules with closely aligned $S_1$ and $T_1$ states (e.g., small $\Delta E_{ST}$) and appropriate symmetry for SOC exhibit high ISC rates.
   - For example, TADF molecules are designed to minimize $\Delta E_{ST}$ and maximize orbital overlap.

2. **CI Contributions:**
   - Configuration mixing increases the density of states that can couple $S_1$ and $T_1$.
   - Molecules with strong multi-configurational character (e.g., those with lone pairs or extended π-systems) often exhibit enhanced ISC.

# 3. Theoretical Framework of Internal Conversion

## 3.1 Definition of Internal Conversion

Internal conversion (IC) is a nonradiative transition between two electronic states (∣i⟩ and ∣j⟩) of the same spin multiplicity, typically a singlet ($S_1 \to S_0$) or triplet ($T_2 \to T_1$) transition. During IC, the electronic energy lost by the molecule is converted into vibrational energy, and the molecule relaxes to the lower electronic state.

### Example Process: $S_1 \to S_0$
1. A molecule absorbs light and reaches the first singlet excited state $S_1$.
2. Through IC, the molecule transitions to the ground state $S_0$ without emitting a photon.
3. The excess energy is transferred to vibrational modes of $S_0$.

---

## 3.2 Governing Equation: Fermi’s Golden Rule

The rate of internal conversion ($k_{IC}$) is determined by Fermi’s Golden Rule:

$$
k_{IC} = \frac{2\pi}{\hbar} \vert \langle i \vert \hat{H}_{NA} \vert j \rangle \vert^2 \rho(E),
$$

where:
- $\hat{H}_{NA}$: The nonadiabatic coupling Hamiltonian, describing coupling between electronic states via nuclear motion.
- $\langle i \vert \hat{H}_{NA} \vert j \rangle$: The nonadiabatic coupling matrix element (NACME), which quantifies the strength of coupling between states ∣i⟩ and ∣j⟩.
- $\rho(E)$: The vibrational density of states (VDOS) in the final electronic state ($S_0$), which determines how many vibrational modes are available to accept the energy difference between $S_1$ and $S_0$.

### Key Factors Affecting $k_{IC}$:
1. **Nonadiabatic Coupling ($\langle i \vert \hat{H}_{NA} \vert j \rangle$):**
   - Strong NAC enhances the IC rate.
2. **Energy Gap ($\Delta E_{ij}$):**
   - Smaller energy gaps lead to stronger vibronic coupling, facilitating IC.
3. **Vibrational Density of States ($\rho(E)$):**
   - Higher $\rho(E)$ at the crossing point increases IC rates by providing more vibrational modes to accept the electronic energy.

---

## 3.3 The Role of the Nonadiabatic Coupling Hamiltonian ($\hat{H}_{NA}$)

The nonadiabatic coupling Hamiltonian arises from the breakdown of the Born-Oppenheimer approximation, which assumes that nuclear and electronic motions can be separated. When electronic states are close in energy, this approximation fails, and coupling between electronic and nuclear motions becomes significant.

### Nonadiabatic Coupling Hamiltonian:

$$
\hat{H}_{NA} = -\frac{\hbar}{M} \nabla_R,
$$

where:
- $M$: Nuclear mass.
- $\nabla_R$: Gradient operator with respect to nuclear coordinates $R$.

This term couples electronic wavefunctions through the nuclear displacement $R$, allowing transitions between electronic states mediated by nuclear motion.

### Nonadiabatic Coupling Matrix Element (NACME):
The strength of nonadiabatic coupling between states ∣i⟩ and ∣j⟩ is quantified by:

$$
\langle i \vert \hat{H}_{NA} \vert j \rangle = \int \psi_i^*(r; R) \nabla_R \psi_j(r; R) \, dr,
$$

where:
- $\psi_i(r; R)$: Electronic wavefunction of state ∣i⟩ at nuclear coordinate $R$.
- $\nabla_R$: Nuclear derivative operator, which quantifies the sensitivity of the electronic wavefunction to nuclear displacement.

---

## 3.4 Energy Gap Law for IC

The energy gap law describes how the rate of IC depends on the energy difference $\Delta E_{ij}$ between the initial ($i$) and final ($j$) electronic states:

- **Small $\Delta E_{ij}$:**
  - Small energy gaps lead to stronger vibronic coupling, increasing $k_{IC}$.
  - The molecule requires less vibrational energy to bridge the gap between the two states.
- **Large $\Delta E_{ij}$:**
  - Large energy gaps reduce vibronic coupling, suppressing IC.
  - Fewer vibrational modes in the final state ($S_0$) match the energy gap, reducing $\rho(E)$.

### Implication:
IC is most efficient when the initial and final states are close in energy (e.g., at conical intersections or near avoided crossings).

---

# 4. Physical Mechanisms of IC

## 4.1 Vibronic Coupling

Vibronic coupling describes the interaction between electronic and vibrational degrees of freedom. It enables IC by allowing the electronic wavefunction to mix with nuclear vibrations. Vibronic coupling depends on:
- **Overlap of Vibrational Wavefunctions:** Stronger overlap enhances IC.
- **Mode-Specific Coupling:** Certain vibrational modes (e.g., bond-stretching modes) contribute more strongly to IC.

---

## 4.2 Conical Intersections and Avoided Crossings

Conical intersections (CIs) and avoided crossings are critical points where two electronic potential energy surfaces come close in energy, leading to strong nonadiabatic coupling.

### Conical Intersections (CIs):
- At a CI, the energy gap ($\Delta E_{ij}$) between $S_1$ and $S_0$ is nearly zero, and the NACME $\langle i \vert \hat{H}_{NA} \vert j \rangle$ becomes large.
- CIs act as "funnels" for IC, allowing the molecule to efficiently transfer electronic energy into vibrational modes.

---

# 5. Factors Influencing IC Rates

## 5.1 Molecular Size
- Larger molecules have more vibrational modes, resulting in higher $\rho(E)$ at the crossing point.
- This enhances IC rates because more vibrational states can accept the electronic energy.

## 5.2 Energy Gap ($\Delta E_{ij}$)
- Small $\Delta E_{ij}$ enhances vibronic coupling and increases $k_{IC}$.

## 5.3 Vibrational Density of States ($\rho(E)$)
- Higher $\rho(E)$ at the crossing point provides more pathways for the energy redistribution required for IC.

- To calculate intersystem crossing S1 to T1 assuming harmoonicity K*K<30


```text
  ! DEF2-SVP ESD(ISC) CPCM(Toluene)
%scf
  moinp "P1.gbw" # triplet energy
end
%TDDFT NROOTS 1
       SROOT 1
       TROOT 1
       TROOTSSL 0
       DOSOC TRUE
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
        ExtParamXC "_omega" 0.062445651649735014
END
%ESD
   ISCISHESSIAN "PI1_S1.hess"
   ISCFSHESSIAN "PI1_T1.hess"
   USEJ TRUE
   DOHT TRUE
   DELE 5949.86896 # energy difference between diabatic S1 T1 positive dowbhill
   SOCME 0.0, 1.96e-6 # S1 to T1 SOC real and Imag
END
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 PI1.xyz #triplet relaxed Geometry
```
if fails try

```text
  ! DEF2-SVP ESD(ISC) CPCM(Toluene)
%scf
  moinp "P1.gbw"
end
%TDDFT NROOTS 1
       SROOT 1
       TROOT 1
       TROOTSSL 0
       DOSOC TRUE
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
        ExtParamXC "_omega" 0.062445651649735014
END
%ESD
   ISCISHESSIAN "PI1_S1.hess"
   ISCFSHESSIAN "PI1_T1.hess"
   USEJ TRUE
   DOHT TRUE
   HESSFLAG   AHAS
   PrintLevel 4
   DELE 5949.86896 # energy difference between diabatic S1 T1
   SOCME 0.0, 1.96e-6 # S1 to T1 SOC real and Imag
END
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 PI1.xyz
```

