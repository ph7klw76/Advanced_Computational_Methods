# Reorganization between singlet and triplet


# Reorganization Energy for Reverse Intersystem Crossing in TADF Molecules

Thermally Activated Delayed Fluorescence (TADF) has emerged as a powerful mechanism to harvest triplet excitons in organic light‐emitting diodes (OLEDs). The process relies on reverse intersystem crossing (RISC) from the triplet ($T_1$) to the singlet ($S_1$) excited state. A key factor that influences the efficiency of RISC is the **reorganization energy**—the energetic cost of reorganizing the nuclear framework when transitioning between the two electronic states. 

In this post, we explore the concept in detail, derive the relevant equations, and outline how to calculate these quantities using the ORCA quantum chemistry package.

## 1. Theoretical Background

### 1.1. TADF and Reverse Intersystem Crossing (RISC)

In TADF molecules, the typical photophysical process involves:

- **Excitation**: Generation of excited states (predominantly forming triplets under electrical excitation).
- **RISC**: Thermal upconversion of triplet excitons to the singlet manifold.
- **Emission**: Radiative decay from the singlet state (delayed fluorescence).

Because the $T_1 \to S_1$ transition is spin‐forbidden, the process is usually slow. However, if the energy gap ($\Delta E_{ST}$) is small and the reorganization energy ($\lambda$) is low, vibronic coupling (often assisted by spin–orbit interactions) can facilitate RISC. Under the framework of nonadiabatic transition theory, the RISC rate can be expressed in a Marcus-like form:

$$
k_{\text{RISC}} = \frac{2\pi}{\hbar} \, \lvert H_{\text{SO}} \rvert^2 \, \text{FCWD}
$$

where $H_{\text{SO}}$ is the spin–orbit coupling matrix element and the Franck–Condon weighted density (FCWD) is given (within the classical Marcus picture) by

$$
\text{FCWD} \propto \frac{1}{4\pi \lambda k_B T} \exp \left[ -\frac{(\Delta E + \lambda)^2}{4\lambda k_B T} \right].
$$

Here, $\Delta E$ is the electronic energy gap between $T_1$ and $S_1$, $k_B$ is Boltzmann’s constant, and $T$ is the temperature.

### 1.2. Defining Reorganization Energy

The **reorganization energy** ($\lambda$) quantifies the energy required to reorganize the molecular geometry from the equilibrium structure of one electronic state to that of the other without a change in the electronic configuration. 

In a two-state (harmonic) approximation, the potential energy surfaces (PES) for the $S_1$ and $T_1$ states along a generalized coordinate $Q$ can be written as:

$$
V_S(Q) = \frac{1}{2} k (Q - Q_S)^2 + E_S,
$$

$$
V_T(Q) = \frac{1}{2} k (Q - Q_T)^2 + E_T,
$$

where:

- $Q_S$ and $Q_T$ are the equilibrium coordinates for $S_1$ and $T_1$, respectively,
- $k$ is the force constant (assumed similar for both states),
- $E_S$ and $E_T$ are the electronic energies at their respective minima.

The reorganization energy for, say, the singlet state is then:

$$
\lambda_{S_1} = V_S(Q_T) - V_S(Q_S) = \frac{1}{2} k (Q_S - Q_T)^2.
$$

In multi-dimensional space (i.e., including all vibrational modes), the total reorganization energy is the sum over all normal modes:

$$
\lambda = \sum_{j}^{3N-6} \lambda_j = \sum_{j}^{3N-6} \frac{1}{2} k_j (\Delta Q_j)^2,
$$

where:

- $k_j$ is the force constant for mode $j$,
- $\Delta Q_j = Q_{S_1,j} - Q_{T_1,j}$.

Often, the vibrational contribution is expressed in terms of the Huang–Rhys factor $S_j$:

$$
S_j = \frac{\lambda_j}{\hbar \omega_j} \Rightarrow \lambda_j = \hbar \omega_j S_j,
$$

with $\omega_j$ being the vibrational frequency of mode $j$.

### 1.3. Extracting $\lambda$ from Computed Energies

An alternative approach for extracting $\lambda$ from computed energies is to perform **vertical energy evaluations** at non-equilibrium geometries. For example:

For the singlet state:

$$
\lambda_{S_1} = E_{S_1}(R_{T_1}) - E_{S_1}(R_{S_1}),
$$

For the triplet state:

$$
\lambda_{T_1} = E_{T_1}(R_{S_1}) - E_{T_1}(R_{T_1}).
$$

In some analyses, the effective reorganization energy for the RISC process is taken as the sum of the two contributions:

$$
\lambda_{\text{eff}} = \lambda_{S_1} + \lambda_{T_1}.
$$

Furthermore, in the Marcus formalism, the **activation free energy** for the transition is given by:

$$
\Delta G^{\ddagger} = \frac{(\Delta E + \lambda)^2}{4\lambda}.
$$

A lower $\lambda$ implies a smaller barrier, thereby facilitating a faster RISC rate.


Calculate the  opt singlet
```python
! DEF2-SVP OPT CPCM(toluene)  # Opt Singlet excited Geo
%TDDFT  NROOTS  2
        IROOT  1
        IROOTMULT SINGLET
        LRCPCM True
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0645
end           
%maxcore 6000
%pal nprocs 16 end
* XYZFILE 0 1 235-tPBisICz.xyz
```


claculate singlet energy based on opt triplet geometry as t.inp file
```python
! DEF2-SVP OPT CPCM(toluene)  # Opt 1nd triplet excited Geo
%TDDFT  NROOTS  1
        IROOT  1
        IROOTMULT TRIPLET
        LRCPCM True
END
%scf
  MaxIter 300
end
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0505
end           
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 EHBIPOAc0.050460446612374865.xyz
$new_job
!DEF2-SVP OPT CPCM(toluene)    # Cal Singlet excited Energy based on 2nd Opt triplet Geo
%TDDFT  NROOTS  1
        IROOTMULT SINGLET
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0505
end           
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 T.xyz
```
