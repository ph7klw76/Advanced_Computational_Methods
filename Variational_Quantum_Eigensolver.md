# The Variational Quantum Eigensolver (VQE)

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm that uses the variational principle to approximate the ground state energy of a quantum system described by a Hamiltonian $H$. VQE is particularly useful in quantum chemistry and materials science, where accurately determining ground states of molecules or materials is crucial yet computationally challenging.

This guide explores the mathematical framework of VQE, detailing each step to show how the algorithm approximates the ground state energy.

## Mathematical Framework of VQE

VQE is based on the variational principle of quantum mechanics, which states that for a Hamiltonian $H$, the ground state energy $E_0$ is the lowest possible expectation value of $H$ over all quantum states $|\psi\rangle$:

$$
E_0 \leq \langle \psi | H | \psi \rangle.
$$

For any trial state $|\psi(\theta)\rangle$, where $\theta$ represents a set of adjustable parameters, the variational principle ensures:

$$
E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle \geq E_0.
$$

By adjusting $\theta$, we aim to minimize $E(\theta)$ to approximate $E_0$ as closely as possible.

### Step-by-Step Breakdown of VQE

1. **Defining the Hamiltonian $H$**

   The system's Hamiltonian $H$ represents the total energy, including kinetic and potential energy, of the particles in the system. In VQE, $H$ is typically expressed in terms of Pauli operators. For example, in quantum chemistry, we represent the Hamiltonian in terms of fermionic operators, mapped to qubit operators (like Pauli-X, Pauli-Y, and Pauli-Z) using transformations such as the Jordan-Wigner or Bravyi-Kitaev mappings.

   For a molecular system, $H$ might look like:

$$
   H = \sum_{i,j} h_{ij} a_i^\dagger a_j + \sum_{i,j,k,l} h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l,
  $$

   where $a_i^\dagger$ and $a_i$ are fermionic creation and annihilation operators, and $h_{ij}$ and $h_{ijkl}$ are constants based on molecular structure. Once mapped to qubit operators, this Hamiltonian becomes a sum of products of Pauli operators:

$$
   H = \sum_k c_k P_k,
$$

   where $P_k$ are tensor products of Pauli operators (e.g., $I \otimes \sigma_z \otimes \sigma_x$), and $c_k$ are coefficients.

2. **Preparing the Parameterized Quantum State $|\psi(\theta)\rangle$**

   To approximate the ground state, we use a parameterized quantum circuit (ansatz) to create a trial state $|\psi(\theta)\rangle$. This state depends on a set of parameters $\theta = (\theta_1, \theta_2, \dots, \theta_m)$ that can be adjusted to minimize the energy expectation value.

   Mathematically, the ansatz is
   
$$
   |\psi(\theta)\rangle = U(\theta) |0\rangle^{\otimes n},
$$

   where $U(\theta)$ is a sequence of quantum gates applied to an initial state $|0\rangle^{\otimes n}$, with each gate depending on the parameters $\theta$.

   **Common Ansatz Choices**:
   - **Hardware-Efficient Ansatz**: Uses gates native to the quantum hardware to reduce circuit depth and noise.
   - **Chemistry-Specific Ansatz (e.g., UCCSD)**: For quantum chemistry, the Unitary Coupled Cluster with Single and Double excitations (UCCSD) ansatz models electron correlation accurately.

4. **Calculating the Energy Expectation Value $E(\theta)$**

   After preparing the ansatz state $|\psi(\theta)\rangle$, we calculate the expectation value of the Hamiltonian $H$:

$$
   E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle.
$$

   Given $H = \sum_k c_k P_k$, we can expand $E(\theta)$ as:

$$
   E(\theta) = \sum_k c_k \langle \psi(\theta) | P_k | \psi(\theta) \rangle.
$$

   Each term $\langle \psi(\theta) | P_k | \psi(\theta) \rangle$ is the expectation value of a Pauli string $P_k$, which can be measured on a quantum computer. This measurement involves repeatedly preparing the state $|\psi(\theta)\rangle$, measuring in the basis of $P_k$, and averaging the results.

5. **Classical Optimization**

   After calculating $E(\theta)$, a classical optimizer adjusts the parameters $\theta$ to minimize $E(\theta)$. This process involves alternating between quantum measurements and classical optimization steps to update $\theta$ based on the energy measurements.

   The objective is:

$$
   \theta_{\text{opt}} = \arg \min_\theta E(\theta),
$$

   where $E(\theta_{\text{opt}})$ approximates the ground state energy $E_0$.

   Various optimization algorithms can be used, including gradient descent, Nelder-Mead, or COBYLA. The optimizer iteratively updates $\theta$ until convergence, aiming to find parameter values that yield the minimum energy.

6. **Convergence to the Ground State Energy**

   VQE repeats Steps 2 through 4, adjusting $\theta$ with each iteration. When $E(\theta)$ converges to a stable minimum, we have found an approximation to the ground state energy, and the corresponding state $|\psi(\theta_{\text{opt}})\rangle$ is an approximation to the ground state.

## Mathematical Example of VQE

Consider a simple two-qubit Hamiltonian:

$$
H = \sigma_z \otimes I + I \otimes \sigma_z + \sigma_x \otimes \sigma_x.
$$

### Define the Ansatz

Use a hardware-efficient ansatz with rotation gates:

$$
|\psi(\theta)\rangle = R_y(\theta_1) \otimes R_y(\theta_2) |00\rangle,
$$

where $R_y(\theta) = e^{-i \theta \sigma_y / 2}$ is a rotation about the $y$-axis.

### Calculate Expectation Value $E(\theta)$

For each Pauli term in $H$, calculate the expectation value:

$$
\langle \psi(\theta) | \sigma_z \otimes I | \psi(\theta) \rangle, \quad \langle \psi(\theta) | I \otimes \sigma_z | \psi(\theta) \rangle, \quad \langle \psi(\theta) | \sigma_x \otimes \sigma_x | \psi(\theta) \rangle.
$$

Combine these measurements weighted by their coefficients to get $E(\theta)$.

### Classical Optimization

A classical optimizer adjusts $\theta_1$ and $\theta_2$ iteratively to minimize $E(\theta)$.

# The Role of Double Excitations in UCCSD

In the Unitary Coupled Cluster with Single and Double excitations (UCCSD) ansatz, **double excitations** are essential for accurately capturing electron correlation effects in molecular systems. These effects significantly influence a molecule's ground state energy and other physical properties by accounting for interactions between electrons. Without double excitations, the ansatz would fail to accurately represent many quantum states, especially for molecules with complex electron distributions.

## Why Electron Correlation Matters

In quantum chemistry, **electron correlation** describes how electrons interact and avoid each other due to mutual repulsion, which is critical for calculating accurate molecular properties and energies. Electron correlation effects can be divided into:

- **Static (or Strong) Correlation**: Dominant in systems where multiple electronic configurations contribute to the wavefunction, such as in bond dissociation.
- **Dynamic Correlation**: Arises from the instantaneous repulsion between electrons and requires capturing correlated motion, especially when one electron’s movement affects another.

The **Hartree-Fock (HF)** method, commonly the starting point for electronic structure calculations, includes an approximate mean-field correlation but misses detailed correlations between individual electrons. The **coupled cluster (CC) theory** improves upon HF by incorporating excitations (e.g., single and double excitations), leading to a more accurate description of electron correlation.

## Role of Double Excitations in UCCSD

In the UCCSD ansatz, double excitations enable the algorithm to represent electron pairs moving in a correlated manner. Here’s why they are necessary:

- **Single Excitations Alone Are Insufficient**: Single excitations involve moving one electron from an occupied to a virtual orbital. While this can adjust the electron distribution, it cannot describe situations where two electrons interact or move together to avoid each other due to repulsive forces. Double excitations, which involve moving two electrons simultaneously, allow the algorithm to capture this correlation.

- **Capturing Electron-Electron Correlation**: Double excitations are vital for modeling dynamic correlation. Moving two electrons simultaneously captures their correlated motion, reflecting each electron’s influence on the others. This correlated movement is especially relevant in regions where electrons are close or need to avoid each other.

- **Improving Accuracy in Molecular Calculations**: UCCSD (with both single and double excitations) strikes a balance between accuracy and computational complexity. It captures much of the correlation energy necessary for chemical accuracy without requiring higher-order excitations, which are computationally prohibitive.

In practice, single excitations alone would yield an incomplete wavefunction, failing to reach "chemical accuracy" (typically around 1 kcal/mol). Including double excitations enhances the UCCSD ansatz's expressiveness, enabling reliable calculations of molecular properties such as bond lengths, reaction energies, and activation energies.

- **Essential for Describing Bonding in Complex Molecules**: Double excitations are critical for accurate bonding descriptions in larger or complex molecules. In cases where bonds are stretched or broken, electrons become more correlated to avoid each other, and double excitations capture these changes in electron distribution.

## Mathematical Form of Double Excitations in UCCSD

In UCCSD, the excitation operator $T$ includes both single excitations $T_1$ and double excitations $T_2$:

$$
T = T_1 + T_2.
$$

- **Single Excitations ($T_1$)**: Represented as 

$$
  T_1 = \sum_{ia} t_{ia} a_a^\dagger a_i,
$$

  where $t_{ia}$ is a parameter for exciting an electron from an occupied orbital $i$ to a virtual orbital $a$.

- **Double Excitations ($T_2$)**: Represented as 

$$
  T_2 = \sum_{ijab} t_{ijab} a_a^\dagger a_b^\dagger a_j a_i,
$$

  where $t_{ijab}$ is a parameter for exciting two electrons from occupied orbitals $i$ and $j$ to virtual orbitals $a$ and $b$.

The UCCSD ansatz wavefunction is then:

$$
|\psi_{UCCSD}\rangle = e^{(T_1 + T_2) - (T_1^\dagger + T_2^\dagger)} |\psi_0\rangle,
$$

where $|\psi_0\rangle$ is the initial Hartree-Fock state. The inclusion of $T_2$ terms enables the ansatz to capture both single-electron movements and correlated two-electron movements, providing a more accurate description of the molecule's ground state.


# Limitations of VQE on Fault-Tolerant Quantum Computers

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm, meaning it relies on both quantum and classical components. Although VQE is promising on current noisy intermediate-scale quantum (NISQ) devices with limited coherence times and qubit counts, its dependence on classical optimization makes it less ideal for fully fault-tolerant quantum computers capable of running fully quantum algorithms.

Even with a fully scalable quantum computer, here are key reasons why VQE would be less efficient than fully quantum algorithms:

### 1. Efficiency and Scaling of Fully Quantum Algorithms

On a fault-tolerant quantum computer, we would have access to more powerful fully quantum algorithms that solve eigenvalue problems without needing classical optimization. Examples include:

- **Quantum Phase Estimation (QPE)**: A fully quantum algorithm that directly estimates the eigenvalues of a Hamiltonian, such as the ground state energy, with exponential speedup compared to VQE. QPE leverages phase estimation and controlled unitaries, which require long coherence times and error correction.
  
- **Quantum Signal Processing and Block-Encoding Methods**: Techniques in quantum signal processing, like the quantum singular value transformation and block-encoding, enable efficient Hamiltonian simulation and eigenvalue estimation without iterative classical optimization.

These fully quantum algorithms generally scale better than VQE as they avoid the costly back-and-forth between quantum and classical computation, which becomes inefficient for large systems.

### 2. Inefficiency of Classical Optimization on Large Systems

VQE relies on a classical optimizer to minimize the energy expectation value by adjusting parameters in the quantum circuit’s ansatz. This process involves challenges such as:

- **High Parameter Count**: As problem size grows, so does the number of parameters in the ansatz, leading to a complex optimization landscape with many local minima (known as the “barren plateau” problem). Classical optimizers struggle in this high-dimensional landscape.
  
- **Measurement Overhead**: Each VQE iteration requires multiple quantum measurements to evaluate the energy expectation value, creating a significant overhead for larger systems. Fully quantum algorithms like QPE avoid this iterative measurement step and output eigenvalues directly with higher accuracy.

In a large, fault-tolerant quantum system, fully quantum algorithms would bypass classical optimization and avoid VQE’s measurement overhead, making them faster and more scalable.

### 3. Ansatz Limitations and Barren Plateaus in Quantum Chemistry

VQE uses a parameterized quantum circuit, or ansatz, to represent the target quantum state. Designing an effective ansatz for complex quantum systems is challenging, especially for larger molecules or materials. Specific challenges include:

- **Expressivity Limits**: Common ansatz choices, like Unitary Coupled Cluster (UCC), may not capture all electron correlation effects in larger molecules, limiting accuracy.
  
- **Barren Plateau Problem**: As system size grows, VQE encounters “barren plateaus,” where the gradient of the cost function vanishes, making optimization difficult. Fully quantum algorithms avoid this issue as they do not rely on a parameterized ansatz or iterative classical optimization.

With a fault-tolerant quantum computer, algorithms like QPE, which do not require an ansatz or parameter optimization, would overcome these challenges.

### 4. Resource Inefficiency

On a full-scale quantum computer, VQE’s resource inefficiency would become apparent. VQE requires:

- **Repeated Quantum State Preparation**: Each VQE iteration involves preparing a parameterized quantum state, which can be costly for larger systems.
  
- **Frequent Quantum-to-Classical Switching**: VQE’s reliance on switching between quantum (for state preparation and measurement) and classical (for optimization) components adds latency. Fully quantum algorithms like QPE perform eigenvalue estimation directly, eliminating this back-and-forth and making better use of quantum resources.

### Summary

While VQE is effective on today’s NISQ devices, it is not ideal for a fully fault-tolerant quantum computer with long coherence times. A fully scalable quantum computer would benefit from fully quantum algorithms like Quantum Phase Estimation and block-encoding techniques, which avoid classical optimization and repeated measurements. These algorithms are more efficient, scalable, and capable of handling larger, complex systems without the limitations of VQE’s hybrid structure. Thus, VQE is an excellent interim solution for NISQ-era devices but will likely be superseded by more efficient quantum algorithms as hardware advances.


# The Variational Quantum Eigensolver for the Hydrogen Molecule with PennyLane

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the ground state energy of a quantum system. It is particularly useful for quantum chemistry problems, where exact solutions are computationally infeasible for large molecules. In this post, we’ll delve into a VQE implementation for the hydrogen molecule (H₂) using PennyLane, a quantum machine learning library. We’ll explore the quantum circuit, discuss the quantum gates used, and provide the mathematical foundations necessary to understand the algorithm.

## Introduction to VQE and Quantum Chemistry

Quantum chemistry aims to solve the Schrödinger equation for molecular systems to determine properties like energy levels and molecular orbitals. For simple molecules, analytical solutions are possible, but as the system size grows, the computational cost becomes prohibitive due to the exponential scaling of the Hilbert space.

The VQE algorithm addresses this challenge by using a parameterized quantum circuit (ansatz) to prepare trial wavefunctions and a classical optimizer to minimize the expected energy. The goal is to find the parameter set that yields the lowest possible energy, approximating the ground state of the system.

## Setting Up the Molecular Hamiltonian

We begin by defining the molecular structure of H₂ and generating its Hamiltonian.

```python
import pennylane as qml
from pennylane import numpy as np

# Define the molecular structure and Hamiltonian for H₂
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, 0.0,  # Coordinates of the first hydrogen atom
                        0.0, 0.0, 0.74])  # Coordinates of the second hydrogen atom (0.74 Å apart)

# Create the molecule
H2 = qml.qchem.Molecule(symbols, coordinates)

# Generate the Hamiltonian
H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
```
# Understanding the Molecular Hamiltonian

The molecular Hamiltonian $\hat{H}$ describes the energy of the electrons in the molecule, considering electron-electron interactions and interactions with the nuclei. In second quantization, it is expressed in terms of creation and annihilation operators, which can be mapped to qubit operators using techniques like the Jordan-Wigner transformation.

The Hamiltonian for H₂ can be written as:

![image](https://github.com/user-attachments/assets/cf1e381e-801c-4730-98db-949dac02ec0f)

where $h_{pq}$ and $h_{pqrs}$ are one- and two-electron integrals, and $\hat{a}^\dagger$ and $\hat{a}$ are creation and annihilation operators, respectively.

# Defining the Quantum Circuit

The quantum circuit (ansatz) prepares the trial wavefunction. It includes parameterized gates whose parameters are optimized to minimize the energy expectation value.

```python
# Initialize a quantum device
dev = qml.device("default.qubit", wires=qubits)

# Define a more complex quantum circuit with additional parameterized gates
def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.RY(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.RY(params[3], wires=2)
    qml.RY(params[4], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.RZ(params[5], wires=0)
    qml.RZ(params[6], wires=1)
    qml.RZ(params[7], wires=2)
    qml.RZ(params[8], wires=3)
```
# Preparing the Initial State with `BasisState`

To prepare the initial state for the quantum circuit, we use the `BasisState` function.

```python
qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
```

This gate initializes the qubits to a specific computational basis state. In our case, we prepare the state $\vert 1100 \rangle$, where the first two qubits are in the $\vert 1 \rangle$ state (occupied orbitals), and the last two are in the $\vert 0 \rangle$ state (unoccupied orbitals).

# Double Excitation with `DoubleExcitation`

To apply a double excitation, we use the `DoubleExcitation` gate.

```python
qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
```

The `DoubleExcitation` gate represents a double excitation from two occupied orbitals to two unoccupied orbitals. Mathematically, it implements the unitary operator:

$$
U_{\text{DE}}(\theta) = \exp \left( -i \theta \left( \hat{c}_2^\dagger \hat{c}_3^\dagger \hat{c}_0 \hat{c}_1 + \text{H.c.} \right) \right)
$$

where $\hat{c}^\dagger$ and $\hat{c}$ are fermionic creation and annihilation operators, and H.c. denotes the Hermitian conjugate.

This gate allows us to explore configurations where two electrons are excited simultaneously, which is essential for capturing electron correlation in quantum chemistry.

# Parameterized Single-Qubit Rotations with `RY` and `RZ`

To add parameterized single-qubit rotations, we apply the `RY` and `RZ` gates as follows:

```python
qml.RY(params[1], wires=0)
qml.RY(params[2], wires=1)
qml.RY(params[3], wires=2)
qml.RY(params[4], wires=3)
# ...
qml.RZ(params[5], wires=0)
qml.RZ(params[6], wires=1)
qml.RZ(params[7], wires=2)
qml.RZ(params[8], wires=3)
```

The `RY` and `RZ` gates perform rotations around the Y and Z axes of the Bloch sphere, respectively. They are defined as:

$$
RY(\theta) = \exp \left( -i \frac{\theta}{2} Y \right), \quad RZ(\theta) = \exp \left( -i \frac{\theta}{2} Z \right)
$$

where $Y$ and $Z$ are the Pauli matrices.

These gates introduce variational parameters that adjust the amplitudes and phases of the qubits, allowing the circuit to explore different quantum states within the Hilbert space.

# Entanglement with `CNOT` Gates

To introduce entanglement between qubits, we use the `CNOT` gate as follows:

```python
qml.CNOT(wires=[0, 1])
qml.CNOT(wires=[2, 3])
```

The `CNOT` (Controlled-NOT) gates create entanglement between qubits, which is crucial for representing correlated electronic states. The `CNOT` gate flips the target qubit if the control qubit is in the $\vert 1 \rangle$ state.

Mathematically, the `CNOT` gate is represented as:

$$
\text{CNOT}_{c \rightarrow t} = \vert 0 \rangle \langle 0 \vert_c \otimes I_t + \vert 1 \rangle \langle 1 \vert_c \otimes X_t
$$

where $c$ is the control qubit, $t$ is the target qubit, $I$ is the identity operator, and $X$ is the Pauli-X matrix.

By applying `CNOT` gates, we entangle pairs of qubits, enabling the circuit to capture electron correlations more effectively.

# Understanding Each Step of the Quantum Circuit for VQE on the Hydrogen Molecule

Let's delve deeper into each step of the quantum circuit to understand why they are necessary and how they contribute to finding the ground state energy of the hydrogen molecule using the Variational Quantum Eigensolver (VQE).

## 1. Preparing the Initial State with `BasisState`

### What is the `BasisState`?

The `BasisState` gate initializes the qubits to a specific computational basis state, represented by a binary string of zeros and ones. In quantum computing, qubits are typically initialized to the $\vert 0 \rangle$ state. However, for quantum chemistry simulations, we often need to represent specific electronic configurations corresponding to molecular orbitals being occupied or unoccupied by electrons.

### Why is it Necessary?

In the context of the H₂ molecule, we need to represent the reference electronic configuration, which serves as the starting point for our variational ansatz. This reference state corresponds to the **Hartree-Fock ground state**, where electrons occupy the lowest energy orbitals according to the Pauli exclusion principle.

For H₂, considering a minimal basis set, we have:

- **Occupied Orbitals:** The lowest energy orbitals where the electrons reside.
- **Unoccupied (Virtual) Orbitals:** Higher energy orbitals that are empty in the Hartree-Fock ground state but can be occupied in excited configurations.

By preparing the initial state $\vert 1100 \rangle$, we represent the two electrons of H₂ occupying the first two orbitals (qubits 0 and 1), with the remaining orbitals (qubits 2 and 3) unoccupied.

### How is it Used in the Code?

```python
qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
```

`np.array([1, 1, 0, 0])`: This array specifies the state of each qubit.

- **Qubit 0:** $\vert 1 \rangle$ (occupied)
- **Qubit 1:** $\vert 1 \rangle$ (occupied)
- **Qubit 2:** $\vert 0 \rangle$ (unoccupied)
- **Qubit 3:** $\vert 0 \rangle$ (unoccupied)

`wires=wires`: Applies the `BasisState` to all qubits in the device.

### Rationale

Starting from the Hartree-Fock state ensures that our variational algorithm begins from a physically meaningful configuration. It represents the best single-particle approximation to the ground state and provides a solid foundation for introducing electron correlation through subsequent gates.

## 2. Double Excitation with `DoubleExcitation`

### What is the `DoubleExcitation` Gate?

The `DoubleExcitation` gate simulates the excitation of two electrons from occupied orbitals to unoccupied orbitals simultaneously. It is a parameterized gate that depends on an angle $\theta$, which controls the amplitude of the excitation.

Mathematically, it implements the unitary operator:

$$
U_{\text{DE}}(\theta) = \exp \left( -i \theta \left( \hat{T} + \hat{T}^\dagger \right) \right)
$$

where $\hat{T}$ is the double excitation operator:

$$
\hat{T} = \hat{a}_a^\dagger \hat{a}_b^\dagger \hat{a}_j \hat{a}_i
$$

with $\hat{a}^\dagger$ and $\hat{a}$ being the fermionic creation and annihilation operators.

### Why is it Necessary?

Electron correlation is a critical aspect of accurately modeling molecular systems. The Hartree-Fock method neglects electron correlation by considering only a single electronic configuration. To capture the effects of electron correlation, we need to consider excited configurations where electrons are promoted to higher energy orbitals.

The double excitation operator allows us to include configurations where two electrons are excited simultaneously. This is essential for capturing correlated electron movements that cannot be represented by single excitations alone.

### How is it Used in the Code?

```python
qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
```

`params[0]`: The variational parameter $\theta$ controlling the amplitude of the double excitation.

`wires=[0, 1, 2, 3]`: Specifies the qubits involved in the excitation.
- **Qubits 0 and 1:** Occupied orbitals (initially in state $\vert 1 \rangle$)
- **Qubits 2 and 3:** Unoccupied orbitals (initially in state $\vert 0 \rangle$)

### Rationale

Including the `DoubleExcitation` gate in our ansatz allows the circuit to explore configurations where both electrons are simultaneously excited to unoccupied orbitals. This provides a more expressive ansatz capable of approximating the true ground state, which may include contributions from these doubly excited states.

By adjusting the parameter $\theta$, the optimizer can control the extent to which these configurations contribute to the trial wavefunction, thus minimizing the energy.

## 3. Parameterized Single-Qubit Rotations with `RY` and `RZ`

### What are the `RY` and `RZ` Gates?

- **`RY(θ)` Gate:** Performs a rotation around the Y-axis of the Bloch sphere by an angle $\theta$.
  
  $$
  R_Y(\theta) = \begin{bmatrix} \cos \left( \frac{\theta}{2} \right) & -\sin \left( \frac{\theta}{2} \right) \\ \sin \left( \frac{\theta}{2} \right) & \cos \left( \frac{\theta}{2} \right) \end{bmatrix}
  $$

- **`RZ(θ)` Gate:** Performs a rotation around the Z-axis of the Bloch sphere by an angle $\theta$.
  
  $$
  R_Z(\theta) = \begin{bmatrix} e^{-i \theta / 2} & 0 \\ 0 & e^{i \theta / 2} \end{bmatrix}
  $$

### Why are They Necessary?

Parameterized single-qubit rotations allow us to adjust the amplitudes and phases of the qubit states in a continuous manner. They introduce additional variational parameters that the optimizer can tune to minimize the energy.

- **Amplitude Control:** `RY` gates can change the probability amplitudes of the qubit being in $\vert 0 \rangle$ or $\vert 1 \rangle$, effectively creating superpositions.
- **Phase Control:** `RZ` gates adjust the relative phase between the basis states $\vert 0 \rangle$ and $\vert 1 \rangle$, which is crucial for interference effects in quantum algorithms.

### How are They Used in the Code?

```python
# RY rotations
qml.RY(params[1], wires=0)
qml.RY(params[2], wires=1)
qml.RY(params[3], wires=2)
qml.RY(params[4], wires=3)

# RZ rotations
qml.RZ(params[5], wires=0)
qml.RZ(params[6], wires=1)
qml.RZ(params[7], wires=2)
qml.RZ(params[8], wires=3)
```

- **`params[1]` to `params[8]`:** The variational parameters for the rotations.
- **Applied to Each Qubit Individually:** This allows independent control over each qubit's state.

### Rationale

By applying parameterized rotations to each qubit, we enrich the ansatz, enabling it to represent a wider variety of quantum states. This flexibility is essential for the optimizer to find the parameter set that yields the lowest energy.

- **Adjusting Superpositions:** The `RY` gates create superpositions between the $\vert 0 \rangle$ and $\vert 1 \rangle$ states, which can represent partial occupancies of orbitals in the quantum chemistry context.
- **Controlling Phases:** The `RZ` gates adjust the phases, allowing the ansatz to construct states with the correct interference patterns necessary for representing the ground state.

The combination of amplitude and phase adjustments is crucial for accurately modeling the quantum state of the electrons in the molecule.

## 4. Entanglement with `CNOT` Gates

### What are `CNOT` Gates?

The `CNOT` (Controlled-NOT) gate is a two-qubit gate that flips the state of the target qubit if the control qubit is in the $\vert 1 \rangle$ state.

Mathematically, the `CNOT` gate is represented by the following matrix:

$$
\text{CNOT} = \begin{bmatrix} 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \end{bmatrix}
$$

### Why are They Necessary?

Entanglement is a fundamental feature of quantum mechanics that allows qubits to exhibit correlations that cannot be explained classically. In quantum chemistry, electron correlation is a crucial aspect, and entanglement between qubits is necessary to model these correlations accurately.

- **Modeling Electron Correlation:** Electrons are indistinguishable particles that interact with each other. To capture these interactions, our quantum states need to exhibit entanglement.
- **Enhancing Ansätze Expressibility:** Entangling gates increase the expressiveness of the ansatz, enabling it to represent more complex quantum states that cannot be factorized into single-qubit states.

### How are They Used in the Code?

```python
qml.CNOT(wires=[0, 1])
qml.CNOT(wires=[2, 3])
```
- `wires=[0, 1]`: The first `CNOT` gate uses qubit 0 as the control and qubit 1 as the target.
- `wires=[2, 3]`: The second `CNOT` gate uses qubit 2 as the control and qubit 3 as the target.

### Rationale

By applying `CNOT` gates between specific pairs of qubits, we introduce entanglement that captures the correlation between electrons in different orbitals.

- **Qubits 0 and 1:** Representing occupied orbitals, entangling them can model correlation effects between electrons occupying these orbitals.
- **Qubits 2 and 3:** Representing unoccupied orbitals, entangling them with each other (and indirectly with occupied orbitals through subsequent gates) allows the ansatz to capture the possibility of simultaneous excitations and their correlations.

**Note:** The choice of which qubits to entangle depends on the specific electron correlation effects we aim to model. In this case, entangling qubits within the occupied and unoccupied subspaces adds necessary correlations without overly complicating the circuit.

## Putting It All Together

### The Overall Circuit

1. **Initialize the Hartree-Fock State:** Start from a meaningful reference state that represents the ground state in the absence of electron correlation.
2. **Introduce Double Excitations:** Allow the ansatz to include configurations where electrons are excited to higher orbitals, which is essential for capturing electron correlation.
3. **Parameterized Rotations:** Provide the optimizer with adjustable parameters to fine-tune the quantum state, enhancing the ansatz's flexibility to approximate the true ground state.
4. **Create Entanglement:** Introduce correlations between qubits to model electron-electron interactions accurately.

### The Necessity of Each Step

- **Initialization:** Without starting from a reasonable reference state, the optimizer might explore irrelevant parts of the Hilbert space, making convergence slower or leading to incorrect results.
- **Double Excitations:** Single excitations might not be sufficient to capture all the necessary electron correlations, especially in molecules where simultaneous electron movements are significant.
- **Parameterized Rotations:** Fixed gates without parameters limit the ansatz to a discrete set of states. Parameterization allows continuous exploration of the state space.
- **Entanglement:** Without entanglement, the ansatz would be restricted to separable states, which cannot represent the correlated nature of the true ground state.

## Optimization and Variational Principle

The VQE algorithm leverages the variational principle, which states that the expectation value of the Hamiltonian with any trial wavefunction $\vert \psi(\theta) \rangle$ is an upper bound to the ground state energy $E_0$:

$$
E(\theta) = \langle \psi(\theta) \vert \hat{H} \vert \psi(\theta) \rangle \geq E_0
$$

By constructing an ansatz with sufficient expressibility (through the steps outlined above), we enable the optimizer to adjust the parameters $\theta$ to find the lowest possible energy, thus approximating the ground state.

## Defining the Cost Function

The cost function calculates the expectation value of the Hamiltonian with respect to the quantum state prepared by the circuit.

```python
# Define the cost function (expectation value of the Hamiltonian)
@qml.qnode(dev)
def cost_fn(params):
    circuit(params, wires=range(qubits))
    return qml.expval(H)
```

This function uses the `qml.qnode` decorator to convert it into a quantum node that can be executed on a quantum device (simulator in this case). The expectation value $\langle \psi(\theta) \vert \hat{H} \vert \psi(\theta) \rangle$ is then minimized by adjusting the parameters $\theta$.

## Optimization Process

We use the Nesterov Momentum Optimizer to minimize the cost function.

```python
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

# Initialize parameters
params = np.random.random(9)  # 9 parameters corresponding to the gates in the circuit

# Choose an optimizer
opt = NesterovMomentumOptimizer(stepsize=0.4)

# Run the optimization
max_iterations = 100
conv_tol = 1e-06
energy_values = []

for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    energy = cost_fn(params)
    energy_values.append(energy)

    if np.abs(energy - prev_energy) < conv_tol:
        break

    print(f"Iteration: {n}, Energy: {energy:.8f} Ha")
```

## Understanding the Optimizer

The Nesterov Momentum Optimizer is a gradient-based optimization algorithm that accelerates convergence by incorporating momentum from previous steps. It updates the parameters according to:

$$
\theta_{n+1} = \theta_n - \eta \nabla_{\theta} E(\theta_n) + \gamma (\theta_n - \theta_{n-1})
$$

where:
- $\eta$ is the learning rate (stepsize),
- $\gamma$ is the momentum coefficient,
- $\nabla_{\theta} E(\theta_n)$ is the gradient of the energy with respect to the parameters.

This combination of gradient descent and momentum helps the optimizer avoid local minima and converge more quickly toward the global minimum.

## Results and Discussion

After running the optimization, we obtain the optimized parameters and the estimated ground state energy.

The optimized parameters represent the values of $\theta$ that minimize the expectation value of the Hamiltonian, effectively approximating the ground state of the hydrogen molecule.

The final energy value represents our best estimate for the ground state energy, based on the variational principle. This energy serves as an upper bound to the true ground state energy of the system and can be compared to the exact value for validation.

The energy convergence plot shows how the energy decreases over iterations, approaching the ground state energy. The optimized parameters define the quantum state that best approximates the ground state of H₂ within the chosen ansatz.

### Accuracy of the VQE

The accuracy of the VQE depends on the expressibility of the ansatz. Our circuit includes a `DoubleExcitation` gate and parameterized single-qubit rotations, which allow us to capture essential electron correlations. These components enhance the ansatz’s ability to approximate the true ground state by:

- **Capturing Electron Correlations**: The `DoubleExcitation` gate enables the circuit to model configurations where electrons are simultaneously excited, which is important for accurate electron correlation.
- **Adjusting Amplitudes and Phases**: The parameterized single-qubit `RY` and `RZ` rotations add flexibility, allowing the circuit to explore different quantum states and better approximate the ground state.

This combination provides an expressive ansatz that balances computational feasibility with accuracy in estimating the ground state energy of the hydrogen molecule.

Full code
```python
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

# Define the molecular structure and Hamiltonian for H₂
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])  # Bond length of 0.74 Å

# Create the molecule
H2 = qml.qchem.Molecule(symbols, coordinates)

# Generate the Hamiltonian
H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

# Initialize a quantum device
dev = qml.device("default.qubit", wires=qubits)

# Define a more complex quantum circuit with additional parameterized gates
def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.RY(params[1], wires=0)
    qml.RY(params[2], wires=1)
    qml.RY(params[3], wires=2)
    qml.RY(params[4], wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    qml.RZ(params[5], wires=0)
    qml.RZ(params[6], wires=1)
    qml.RZ(params[7], wires=2)
    qml.RZ(params[8], wires=3)

# Define the cost function (expectation value of the Hamiltonian)
@qml.qnode(dev)
def cost_fn(params):
    circuit(params, wires=range(qubits))
    return qml.expval(H)

# Initialize parameters
params = np.random.random(9)  # Updated to match the number of parameters in the circuit

# Choose an optimizer
opt = NesterovMomentumOptimizer(stepsize=0.4)

# Run the optimization
max_iterations = 100
conv_tol = 1e-06
energy_values = []

for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    energy = cost_fn(params)
    energy_values.append(energy)

    if np.abs(energy - prev_energy) < conv_tol:
        break

    print(f"Iteration: {n}, Energy: {energy:.8f} Ha")

# Output the results
print(f"\nOptimal parameters: {params}")
print(f"Ground state energy: {energy:.8f} Ha")

# Plot the energy convergence
plt.plot(energy_values, label="Energy")
plt.xlabel("Iteration")
plt.ylabel("Energy (Ha)")
plt.title("VQE Energy Convergence")
plt.legend()
plt.show()
```













