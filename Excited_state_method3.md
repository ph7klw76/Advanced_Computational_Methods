# 1. STEOM-CCSD

## 1.1 Overview

The Similarity-Transformed Equation-of-Motion Coupled Cluster with Singles and Doubles (STEOM-CCSD) method is an efficient alternative to standard EOM-CCSD. It is based on a similarity transformation of the Hamiltonian using the ground state coupled cluster (CC) wavefunction and employs additional transformations to decouple the excitation manifolds. One of its major advantages is that it produces excitation energies with reduced cost while retaining a high degree of accuracy.

---

## 1.2 Ground-State Coupled Cluster and Similarity Transformation

We begin with the ground state CCSD wavefunction written as:

$$
\vert \Psi_0 \rangle = e^{\hat{T}} \vert \Phi_0 \rangle, (1)
$$


where

- $\vert \Phi_0 \rangle$ is the reference determinant (usually Hartree–Fock),
- $\hat{T} = T_1 + T_2$ is the cluster operator with single and double excitation components:

$$
T_1 = \sum_{i,a} t_{i}^{a} a_a^\dagger a_i, \quad T_2 = \frac{1}{4} \sum_{i,j,a,b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i. (2)
$$


The similarity-transformed Hamiltonian is defined as:

$$
\bar{H} = e^{-\hat{T}} H e^{\hat{T}}  (3)
$$

which—although non-Hermitian—has eigenvalues that coincide with those of the full Hamiltonian $H$.

---

## 1.3 STEOM Transformation: Decoupling the Excitation Manifolds

In standard EOM-CCSD, one solves the eigenvalue problem:

$$
\bar{H} R_k \vert \Phi_0 \rangle = E_k R_k \vert \Phi_0 \rangle, (4)
$$

with the excitation operator $R_k$ expanded in the full singles and doubles space. In STEOM-CCSD, the strategy is to introduce an additional similarity transformation that decouples a selected “target” subspace (typically the neutral excitations) from the complementary parts of the Hilbert space. 

This is achieved by first computing ionization potentials (IP) and electron affinities (EA) via separate EOM-CCSD (or similar) calculations. The idea is to use these results to construct an effective Hamiltonian for neutral excitations.

Let $S$ denote an operator that connects the “active” space (built from a set of IP and EA eigenstates) with the remainder. One then defines a further transformation:

$$
\tilde{H} = (1 + S)^{-1} \bar{H} (1 + S), (5)
$$

so that the effective eigenvalue problem in the “target” (STEOM) space becomes:

$$
\tilde{H} X = \omega X. (6)
$$

Here:

- $X$ represents the eigenvector expressed in the active (transformed) basis,
- $\omega = E_k - E_0$ is the excitation energy.

The operator $S$ is chosen so that the off-diagonal coupling between the active space and its complement is minimized (or eliminated), thereby “folding” the influence of high-energy excitations into an effective Hamiltonian that is much smaller than the full EOM space. In practical implementations, the matrix elements of $\tilde{H}$ are computed using the previously determined IP and EA amplitudes.

---

## 1.4 Summary of STEOM-CCSD Equations

To recapitulate, the STEOM-CCSD procedure involves:

1. Solving the CCSD amplitude equations to obtain $\hat{T}$ (Eq. (2)).
2. Constructing the similarity-transformed Hamiltonian $\bar{H}$ (Eq. (3)).
3. Computing separate IP and EA excitation energies and amplitudes.
4. Defining the transformation operator $S$ and constructing the effective Hamiltonian $\tilde{H}$ via Eq. (5).
5. Diagonalizing $\tilde{H}$ in the reduced (STEOM) space to obtain excitation energies (Eq. (6)).

The decoupling achieved by the $S$ transformation dramatically reduces the dimension of the eigenvalue problem while still retaining most of the correlation effects included in the full EOM-CCSD.

---

# 2. IH-FSMR-CCSD

## 2.1 Overview

Intermediate Hamiltonian Fock Space Multi-Reference Coupled Cluster (IH-FSMR-CCSD) methods are designed for systems where a single-reference description is inadequate—typically those with near-degeneracies or strong static correlation. In Fock-space coupled cluster theory, the Hilbert space is partitioned into sectors corresponding to different numbers of electrons (or holes). The “intermediate Hamiltonian” (IH) formulation introduces an effective Hamiltonian that decouples a chosen model space from the remaining (external) space. This facilitates a balanced description of states that require a multi-reference treatment.

---

## 2.2 Fock-Space Partitioning and the Cluster Ansatz

In Fock-space CC theory, one defines a reference state $\vert \Phi_0 \rangle$ (for instance, corresponding to the N-electron system) and considers sectors such as (0,1) or (1,0) for electron attachment or removal, respectively. For excited states in a multi-reference framework, the wavefunction in a given Fock-space sector is written as:

$$
\vert \Psi_k (m,n) \rangle = e^{\hat{T}(m,n)} R_k (m,n) \vert \Phi_0 \rangle, (7)
$$


where the superscript $(m,n)$ indicates the number of electrons removed ($m$) and attached ($n$), and $R_k (m,n)$ is an excitation operator within that sector.

---

## 2.3 The Intermediate Hamiltonian (IH) Approach

The key idea of the intermediate Hamiltonian approach is to define a projection operator $P$ that selects the model (or active) space. The full Hilbert space is then partitioned as:

$$
P + Q = I, (8)
$$

where $Q$ is the projector onto the external (complement) space. The goal is to define an effective Hamiltonian $H_{\text{eff}}$ such that:

$$
H_{\text{eff}} \vert \Psi_k^{\text{model}} \rangle = E_k \vert \Psi_k^{\text{model}} \rangle, (9)
$$

with:

$$
H_{\text{eff}} = P \bar{H} P. (10)
$$

Here, $\bar{H} = e^{-\hat{T}} H e^{\hat{T}}$ is the similarity-transformed Hamiltonian in the Fock space.

---

# 5. Concluding Remarks

In this blog, we have presented a detailed discussion of four advanced methods for computing excited states in molecular systems:

- **STEOM-CCSD** uses a two-step similarity transformation to reduce the dimensionality of the excitation space by folding in ionization potential and electron affinity information.
- **IH-FSMR-CCSD** is tailored for multi-reference problems, where an intermediate Hamiltonian in Fock space is constructed to decouple the model space from intruding external states.
- **PNO-based Coupled Cluster Methods** exploit the compactness of pair natural orbitals to reformulate CC equations in a dramatically reduced virtual space.
- **DLPNO-STEOM-CCSD** further combines local correlation (via orbital localization and domain partitioning) with STEOM-CCSD to achieve near-canonical accuracy at a lower computational cost.

Each method involves a series of transformations and approximations designed to balance computational efficiency with the accurate treatment of electron correlation.

Below is a summary table that compares the key excited state methods—STEOM-CCSD, IH-FSMR-CCSD, PNO-based Coupled Cluster, and DLPNO-STEOM-CCSD—in terms of their advantages, disadvantages, and the types of systems for which they are best suited.

| **Method**               | **Pros**                                                                                                                                              | **Cons**                                                                                                                                               | **Suitable Molecules/Systems** |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| **STEOM-CCSD**          | - **Efficiency via Folding**: The additional similarity transformation reduces the dimensionality of the eigenvalue problem relative to full EOM-CCSD.  <br> - **Accuracy for Valence Excitations**: Provides high accuracy for many valence excitations in single-reference systems.  <br> - **Cost-Effective**: Lower computational cost compared to canonical EOM-CCSD.  | - **Dependence on IP/EA Quality**: The method’s performance is linked to the accuracy of the computed ionization potential (IP) and electron affinity (EA) states.  <br> - **Single-Reference Limitation**: Not optimal for systems with strong multi-reference or static correlation effects. | Medium-sized, predominantly single-reference molecules (e.g., many organic compounds and closed-shell systems) that exhibit well-defined valence excitations and moderate electron correlation. |
| **IH-FSMR-CCSD**        | - **Multi-Reference Capability**: Designed to treat systems with near-degeneracies and strong static correlation by partitioning Fock space and decoupling the model space via an intermediate Hamiltonian.  <br> - **Balanced Treatment**: Effectively balances static and dynamic correlation in challenging cases.  | - **Computational Complexity**: More demanding than single-reference methods due to the multi-reference formulation and Fock-space partitioning.  <br> - **Intruder States**: May require careful parameter tuning to avoid intruder state problems and ensure proper decoupling. | Systems exhibiting significant multi-reference character such as transition metal complexes, diradicals, and molecules with near-degenerate electronic states where a single-reference treatment would fail. |
| **PNO-Based Coupled Cluster** | - **Reduced Virtual Space**: Employs pair natural orbitals (PNOs) to compress the virtual orbital space, significantly reducing computational cost.  <br> - **Scalability**: Enhances scalability, making it feasible to study larger systems while retaining high accuracy.  <br> - **Adaptability**: Accuracy can be tuned via truncation thresholds.  | - **Threshold Sensitivity**: The method’s accuracy is sensitive to the chosen truncation thresholds; overly aggressive truncation can lead to loss of accuracy.  <br> - **Additional Overhead**: Requires extra steps for generating and managing PNOs compared to canonical approaches. | Large molecules or systems with a dominant single-reference character, where full canonical CC methods become too expensive, such as sizable organic or inorganic molecules with moderate correlation effects. |
| **DLPNO-STEOM-CCSD**    | - **Local Correlation Efficiency**: Combines domain-based local correlation (via orbital localization and domain partitioning) with the STEOM approach to further reduce computational cost.  <br> - **High Scalability**: Highly efficient for very large systems while still retaining near-canonical accuracy when properly tuned.  | - **Parameter Sensitivity**: Requires careful localization, domain selection, and threshold setting; may necessitate reparameterization for different types of systems.  <br> - **Delocalization Limitations**: Potentially less accurate for systems with highly delocalized electronic states. | Very large molecular systems (e.g., biomolecules, polymers, extended organic/inorganic systems) or cases where excitations are largely localized and computational resources are a limiting factor. |


# When and Where Double Excitations Become Important
Despite the dipole selection rules, double excitation character can become significant in organic molecules for several reasons:

Configuration Mixing:
Even though a pure double excitation is dipole-forbidden, the true excited state of a molecule is often a mixture of configurations. In many cases—especially in conjugated systems or molecules with near-degeneracies—the excited state may be described as a combination of predominantly single excitations plus a significant contribution from double excitations. This mixing occurs through electron correlation effects. Methods that only account for single excitations (like standard Time-Dependent Density Functional Theory, TDDFT, or CIS) may then miss critical parts of the state’s character, leading to inaccurate excitation energies or incorrect ordering of states.

Strongly Correlated Systems:
In systems where static (or nondynamic) electron correlation is strong—such as in polyenes, conjugated organic molecules, or molecules at a conical intersection—the ground and low-lying excited states can have multi-reference character. In these cases, the wavefunction of the excited state can have large contributions from configurations that involve the simultaneous promotion of two electrons. This is not typically reached via a single direct dipole transition but rather emerges from the strong mixing in the correlated wavefunction.

