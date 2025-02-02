# Excited States via RPA, CIS, and SF-TDA

## 1. Introduction

Excited state phenomena are central to understanding spectroscopy, photochemistry, and electronic properties in molecules and materials. Standard ground-state methods such as Hartree–Fock (HF) or density functional theory (DFT) do not adequately capture the physics of excited states. Several post–ground-state methods have been developed; among the simplest and most instructive are:

- **Random Phase Approximation (RPA):** Emerging from many-body perturbation theory and the linear response of the electron density.
- **Configuration Interaction Singles (CIS):** A wavefunction method that approximates excited states by including only singly excited determinants.
- **Spin-Flip Tamm–Dancoff Approximation (SF-TDA):** A variant of TDA (which itself is a simplified version of RPA) adapted to systems where a spin flip is essential to access low-spin states from a high-spin reference.

In the following sections, we describe the theoretical foundations of these methods, derive their governing equations, and explain each term in detail.

---

## 2. Random Phase Approximation (RPA)

### 2.1 Overview

The Random Phase Approximation (RPA) is a method rooted in linear response theory. In its time-dependent formulation, RPA describes how a many-electron system responds to an external perturbation, thereby providing access to excitation energies and response functions. In the context of excited state calculations, RPA is often formulated as an eigenvalue problem that couples both excitation (forward) and de-excitation (backward) amplitudes.

---

### 2.2 The RPA Eigenvalue Equation

In matrix form, the RPA eigenvalue problem is expressed as:

![image](https://github.com/user-attachments/assets/b2ceedd9-2f65-445b-8b2d-4e045a4366ce)


Here:

- $X$ and $Y$ are vectors of excitation and de-excitation amplitudes, respectively.
- $\omega$ represents the excitation energies.

The structure of Eq. (1) shows that RPA accounts for both upward (excitation) and downward (de-excitation) processes, with the coupling mediated by the off-diagonal block $B$.

---

### 2.3 Detailed Definition of the Matrices

For a system with **occupied orbitals** (indexed by $i, j$) and **virtual orbitals** (indexed by $a, b$), the matrix elements are given by:

![image](https://github.com/user-attachments/assets/a08a20fc-641f-45f0-8278-de7803179362)


where:

- $\epsilon_p$ denotes the orbital energy of orbital $p$.
- $\delta_{ij}$ is the Kronecker delta, ensuring that the orbital energy differences are only included when the indices match.

The **antisymmetrized two-electron integral** is defined as:

$$
\langle ia || jb \rangle = \langle ia | jb \rangle - \langle ia | bj \rangle.   [4]
$$

Here, $\langle ia | jb \rangle$ is the **standard Coulomb integral**, while $\langle ia | bj \rangle$ is the **exchange integral**.

---

### 2.4 Connection to Tamm–Dancoff Approximation (TDA)

The Tamm–Dancoff Approximation (TDA) is obtained from RPA by **neglecting the de-excitation amplitudes**, effectively setting $Y=0$ or equivalently $B=0$. The eigenvalue problem then simplifies to:

$$
A X = \omega X.  [5]
$$

While TDA is computationally simpler, it omits coupling terms that can be significant in cases where de-excitations are non-negligible.

---

## 3. Configuration Interaction Singles (CIS)

### 3.1 Overview

Configuration Interaction Singles (CIS) is a **wavefunction-based method** for excited states that considers **linear combinations of singly excited determinants** relative to a single-determinant reference (typically the Hartree–Fock ground state). Although CIS neglects electron correlation beyond single excitations, it provides a useful **first approximation** for excited state energies and transition properties.

---

### 3.2 The CIS Wavefunction Ansatz

In CIS, the excited state wavefunction is expanded as:

$$
| \Psi_{\text{CIS}} \rangle = \sum_{i,a} c_{ia} | \Phi_{ia} \rangle.  [6]
$$

where:

- $| \Phi_{ia} \rangle$ is a Slater determinant obtained by promoting an electron from an occupied orbital $i$ to a virtual orbital $a$.
- $c_{ia}$ are **variational coefficients** that are determined by **diagonalizing the Hamiltonian** in the space of singly excited determinants.

---

### 3.3 The CIS Hamiltonian Matrix

The matrix elements in the CIS basis are given by:

$$
H_{ia,jb} = \langle \Phi_{ia} | \hat{H} | \Phi_{jb} \rangle.  [7]
$$

Under the CIS approximation, these matrix elements take the form:

$$
H_{ia,jb} = ( \epsilon_a - \epsilon_i ) \delta_{ij} \delta_{ab} + \langle ia || jb \rangle. [8]
$$

Notice the resemblance between Eq. (8) and the matrix $A$ in the RPA formulation (Eq. (2)). In CIS, there is **no coupling to de-excitations** (i.e., no counterpart to $B$) because only **singly excited configurations** are included.

---

### 3.4 The CIS Eigenvalue Equation

The excited state energies $E$ and coefficients $c_{ia}$ are obtained by solving the **standard eigenvalue problem**:

$$
H c = E c.  [9]
$$

---

## 4. Spin-Flip Tamm–Dancoff Approximation (SF-TDA)

### 4.1 Motivation and Overview

The **standard CIS or TDA approaches sometimes fail** in cases where the nature of the excited state **involves a change in spin**. This is particularly true for systems with **diradical character** or for **bond-breaking processes**, where a spin-flip can lead from a **high-spin reference** to a **low-spin excited state**. The **Spin-Flip Tamm–Dancoff Approximation (SF-TDA)** adapts the TDA framework to **include spin-flip excitations**.

---

### 4.2 The SF-TDA Eigenvalue Equation

Restricting the excitation space to **spin-flip configurations** and applying the TDA (i.e., **neglecting coupling to de-excitations**), we arrive at an **eigenvalue equation** analogous to Eq. (5):

$$
A_{\text{SF}} r = \omega r. [11]
$$

Here, the matrix elements $A_{ia,jb}^{\text{SF}}$ are defined by:

$$
A_{ia,jb}^{\text{SF}} = ( \epsilon_a - \epsilon_i ) \delta_{ij} \delta_{ab} + \langle ia || jb \rangle_{\text{SF}}. [12]
$$

where $\langle ia || jb \rangle_{\text{SF}}$ denotes the **two-electron integrals** computed with the appropriate **spin constraints** (only those integrals that involve a **spin flip** are retained).

---

## 5. Conclusion

Understanding excited states remains a **central challenge** in theoretical and computational chemistry. In this blog, we have presented a rigorous exploration of **three methods**—RPA, CIS, and SF-TDA—each offering a different **balance between computational cost and accuracy**. 

Each method has its **domain of applicability**, and a deep understanding of their derivations and limitations is essential for their proper use in research. As computational techniques continue to evolve, these methods serve as **foundational tools** for exploring the rich and complex world of electronic excitations in matter.
