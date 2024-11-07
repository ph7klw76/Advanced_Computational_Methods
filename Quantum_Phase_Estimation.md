# Quantum Phase Estimation

Quantum Phase Estimation (QPE) is one of the most fundamental algorithms in quantum computing. It serves as a critical subroutine in many other quantum algorithms, such as Shor's algorithm for integer factorization and quantum algorithms for solving linear systems of equations. Understanding QPE is essential for grasping the power and potential of quantum computation.

## Introduction

In this blog post, we will delve into the mathematical foundations of QPE, providing a rigorous and detailed explanation of the algorithm. We'll explore how QPE works, the mathematical principles behind it, and its significance in the broader context of quantum computing.

### Why Quantum Phase Estimation Matters

QPE plays a pivotal role in many quantum algorithms by estimating the eigenvalues (phases) associated with unitary operators. It forms a core building block for algorithms that promise exponential speedups over classical counterparts, making it crucial for researchers and practitioners alike.

### The Problem Statement

At its core, the Quantum Phase Estimation (QPE) algorithm aims to solve the following problem:

**Given:**

- A unitary operator $U$ acting on a quantum state.
- An eigenvector $\vert \psi \rangle$ of $U$, such that $U \vert \psi \rangle = e^{2 \pi i \phi} \vert \psi \rangle$, where $\phi$ is an unknown real number between 0 and 1.

**Goal:**

Estimate the phase $\phi$ with high precision.

In essence, QPE estimates the eigenvalues (phases) of a unitary operator corresponding to its eigenvectors.

## Mathematical Foundations

### Unitary Operators and Eigenvalues

A unitary operator $U$ is a linear operator that satisfies $U^{\dagger} U = UU^{\dagger} = I$, where $U^{\dagger}$ is the Hermitian adjoint of $U$, and $I$ is the identity operator. Unitary operators are significant in quantum mechanics because they represent evolution operators that preserve the norm of quantum states.

Eigenvalues of unitary operators have a special form. For a unitary $U$ and eigenvector $\vert \psi \rangle$, we have: 

$$
U \vert \psi \rangle = e^{2 \pi i \phi} \vert \psi \rangle,
$$

where $\phi$ is a real number in the range $[0, 1)$.

### Quantum Fourier Transform (QFT)

The Quantum Fourier Transform is the quantum analogue of the discrete Fourier transform. It plays a crucial role in QPE by transforming quantum states into a different basis where phase information becomes accessible.

For an $n$-qubit system, the QFT is defined as:

$$
\text{QFT} \vert k \rangle = \frac{1}{\sqrt{2^n}} \sum_{j=0}^{2^n - 1} e^{2 \pi i kj / 2^n} \vert j \rangle.
$$

## The Quantum Phase Estimation Algorithm: Detailed Steps

### Overview

The Quantum Phase Estimation algorithm estimates the phase $\phi$ in the eigenvalue equation:

$$
U \vert \psi \rangle = e^{2 \pi i \phi} \vert \psi \rangle,
$$

where:

- $U$ is a known unitary operator.
- $\vert \psi \rangle$ is an eigenstate of $U$.
- $\phi$ is an unknown real number in the range $[0, 1)$.

The algorithm uses two quantum registers:

- **Control Register**: An $n$-qubit register initialized to $\vert 0 \rangle^{\otimes n}$.
- **Target Register**: Initialized to the eigenstate $\vert \psi \rangle$.

The goal is to estimate $\phi$ to $n$ bits of precision.

### Step-by-Step Explanation

#### Step 1: Initialize the Registers

The initial state of the system is:

$$
\vert \Psi_0 \rangle = \vert 0 \rangle^{\otimes n} \otimes \vert \psi \rangle.
$$

- **Control Register**: All qubits are in the $\vert 0 \rangle$ state.
- **Target Register**: Contains the eigenstate $\vert \psi \rangle$ of $U$.

This state represents the tensor product of the control and target registers.

#### Step 2: Apply Hadamard Gates to the Control Register

Apply the Hadamard gate $H$ to each qubit in the control register:

$$
H^{\otimes n} \vert 0 \rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} \vert k \rangle.
$$

The Hadamard gate transforms $\vert 0 \rangle$ into $\frac{1}{\sqrt{2}} (\vert 0 \rangle + \vert 1 \rangle)$. Applying $H$ to each qubit creates a superposition of all possible $n$-bit binary numbers $\vert k \rangle$, where $k$ ranges from 0 to $2^n - 1$.

The combined state after applying the Hadamard gates is:

$$
\vert \Psi_1 \rangle = \left( \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} \vert k \rangle \right) \otimes \vert \psi \rangle.
$$

This state is a superposition over all possible control register states, each tensor-multiplied by the target register state $\vert \psi \rangle$.

#### Step 3: Apply Controlled Unitary Operations

We apply a series of controlled-unitary operations to entangle the control register with the phase information from the unitary operator $U$.

For each qubit $j$ in the control register (from most significant bit to least significant bit), we perform a controlled-$U^{2^{n-j}}$ operation.

**Mathematical Formulation:**

For each $k$ in the sum, the binary representation of $k$ determines which controlled-unitaries are applied. The number $k$ can be represented in binary as:

$$
k = k_{n-1}k_{n-2} \ldots k_0,
$$

where $k_j \in \{0, 1\}$ is the state of the $j$-th qubit.

**Applying Controlled Unitaries:**

Each qubit $k_j$ controls the application of $U^{2^j}$. The cumulative operation for a given $k$ is $U_k$, where:

$$
U_k = U^{k_{n-1} 2^{n-1} + k_{n-2} 2^{n-2} + \ldots + k_0 2^0}.
$$

**State After Controlled Unitaries:**

$$
\vert \Psi_2 \rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} \vert k \rangle \otimes U_k \vert \psi \rangle.
$$

Each term in the sum now has the target register transformed by $U_k$.

**Understanding $U_k \vert \psi \rangle$:**

Since $\vert \psi \rangle$ is an eigenstate of $U$, we have:

$$
U_k \vert \psi \rangle = (e^{2 \pi i \phi})^k \vert \psi \rangle = e^{2 \pi i k \phi} \vert \psi \rangle.
$$

Therefore, the state becomes:

$$
\vert \Psi_2 \rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} e^{2 \pi i k \phi} \vert k \rangle \otimes \vert \psi \rangle.
$$

The phase $e^{2 \pi i k \phi}$ is now associated with each basis state $\vert k \rangle$.

#### Step 4: Apply the Inverse Quantum Fourier Transform to the Control Register

The Quantum Fourier Transform (QFT) is a linear transformation that maps computational basis states $\vert k \rangle$ to a new set of basis states. The inverse QFT reverses this transformation.

**Definition of the QFT$^{\dagger}$:**

For an $n$-qubit register, the inverse QFT is defined as:

$$
\text{QFT}^{\dagger} \vert y \rangle = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n - 1} e^{-2 \pi i x y / 2^n} \vert x \rangle.
$$

**Applying QFT$^{\dagger}$ to the Control Register:**

We apply $\text{QFT}^{\dagger}$ to the control register in state $\vert \Psi_2 \rangle$:

$$
\vert \Psi_3 \rangle = (\text{QFT}^{\dagger} \otimes I) \vert \Psi_2 \rangle,
$$

where $I$ is the identity operator on the target register.

Substituting $\vert \Psi_2 \rangle$:

$$
\vert \Psi_3 \rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} e^{2 \pi i k \phi} (\text{QFT}^{\dagger} \vert k \rangle) \otimes \vert \psi \rangle.
$$

#### Step 5: Measure the Control Register

After applying the inverse QFT, we measure the control register. The measurement collapses the superposition to a specific state $\vert x \rangle$.

The value of $x$ provides an $n$-bit approximation of $\phi$:

$$
\tilde{\phi} = \frac{x}{2^n}.
$$

The accuracy of $\tilde{\phi}$ depends on the number of qubits $n$.

### Putting It All Together: The Complete Algorithm

1. **Initialize Registers**:
   - Control register: $\vert 0 \rangle^{\otimes n}$.
   - Target register: $\vert \psi \rangle$.
2. **Apply Hadamard Gates to Control Register**:
   - Create a uniform superposition over all possible $n$-bit states.
3. **Apply Controlled-$U^{2^j}$ Operations**:
   - For each qubit $j$ in the control register, apply $U^{2^j}$ controlled by that qubit.
   - This entangles the control register with the phase information of $U$.
4. **Apply the Inverse QFT to the Control Register**:
   - Transforms the phase-encoded state into one where the probability amplitudes are concentrated around the binary fractions approximating $\phi$.
5. **Measure the Control Register**:
   - Obtain an $n$-bit approximation $\tilde{\phi}$ of $\phi$.

### An Intuitive Understanding

The algorithm encodes the unknown phase $\phi$ into the amplitudes of the control register. The inverse QFT effectively performs a quantum Fourier analysis, transforming the phase information into a measurable probability distribution. Measurement of the control register reveals the approximate value of $\phi$.

## Detailed Example: Estimating $\phi = 0.625$ with $n = 3$ Qubits

Let's walk through an explicit example to solidify our understanding.

### Given:
- $\phi = 0.625 = \frac{5}{8}$.
- Binary representation of $\phi$: $0.101$.
- Number of qubits in the control register: $n = 3$.

### Step-by-Step Execution

#### Step 1: Initialize the Registers
$$
\vert \Psi_0 \rangle = \vert 0 \rangle^{\otimes 3} \otimes \vert \psi \rangle.
$$

#### Step 2: Apply Hadamard Gates to the Control Register
$$
\vert \Psi_1 \rangle = \left( \frac{1}{\sqrt{8}} \sum_{k=0}^{7} \vert k \rangle \right) \otimes \vert \psi \rangle.
$$

The control register is now in an equal superposition of all 3-bit states $\vert k \rangle$.

#### Step 3: Apply Controlled Unitary Operations

We need to apply controlled-$U^{2^j}$ operations for $j = 0, 1, 2$.

**Calculating $U_k \vert \psi \rangle$:**

For each $k$ from 0 to 7, compute $U_k \vert \psi \rangle$:

$$
U_k \vert \psi \rangle = e^{2 \pi i k \phi} \vert \psi \rangle = e^{2 \pi i k \times 0.625} \vert \psi \rangle.
$$

**Compute the Phases:**

- $k = 0$: $e^{2 \pi i \times 0 \times 0.625} = 1$.
- $k = 1$: $e^{2 \pi i \times 1 \times 0.625} = e^{2 \pi i \times 0.625}$.
- $k = 2$: $e^{2 \pi i \times 2 \times 0.625} = e^{2 \pi i \times 1.25} = e^{2 \pi i \times 0.25}$ (since $e^{2 \pi i n} = 1$ for integer $n$).
- $k = 3$: $e^{2 \pi i \times 1.875} = e^{2 \pi i \times 0.875}$.
- $k = 4$: $e^{2 \pi i \times 2.5} = e^{2 \pi i \times 0.5}$.
- $k = 5$: $e^{2 \pi i \times 3.125} = e^{2 \pi i \times 0.125}$.
- $k = 6$: $e^{2 \pi i \times 3.75} = e^{2 \pi i \times 0.75}$.
- $k = 7$: $e^{2 \pi i \times 4.375} = e^{2 \pi i \times 0.375}$.

**Resulting State After Controlled Unitaries:**

$$
\vert \Psi_2 \rangle = \frac{1}{\sqrt{8}} \sum_{k=0}^{7} e^{2 \pi i k \times 0.625} \vert k \rangle \otimes \vert \psi \rangle.
$$

#### Step 4: Apply the Inverse QFT to the Control Register

We need to apply $\text{QFT}^\dagger$ to the control register.

**Expressing the State Before $\text{QFT}^\dagger$:**

Let’s write down the phases explicitly:

$$
\vert \Psi_2 \rangle = \frac{1}{\sqrt{8}} \left( \vert 0 \rangle + e^{2 \pi i \times 0.625} \vert 1 \rangle + e^{2 \pi i \times 0.25} \vert 2 \rangle + e^{2 \pi i \times 0.875} \vert 3 \rangle + e^{2 \pi i \times 0.5} \vert 4 \rangle + e^{2 \pi i \times 0.125} \vert 5 \rangle + e^{2 \pi i \times 0.75} \vert 6 \rangle + e^{2 \pi i \times 0.375} \vert 7 \rangle \right) \otimes \vert \psi \rangle.
$$

**Computing the Exponentials:**

Use $e^{2 \pi i \theta} = \cos(2 \pi \theta) + i \sin(2 \pi \theta)$.

- $e^{2 \pi i \times 0.625} = \cos(2 \pi \times 0.625) + i \sin(2 \pi \times 0.625) = \cos(1.25 \pi) + i \sin(1.25 \pi) = -\frac{1}{2} - i \frac{1}{2}$.

Similarly, compute for other $k$.

**Applying the Inverse QFT:**

The inverse QFT transforms the state into one where the amplitudes are concentrated around $\vert x \rangle$ such that $x / 2^n$ is closest to $\phi$.

Since $\phi = 0.625$ and $2^n = 8$:

$$
2^n \phi = 8 \times 0.625 = 5.
$$

We expect the measurement to yield $x = 5$.

**After Applying $\text{QFT}^\dagger$, the State Becomes:**

$$
\vert \Psi_3 \rangle = \vert 5 \rangle \otimes \vert \psi \rangle.
$$

This is because the inverse QFT transforms the phase-encoded superposition into a computational basis state corresponding to the best estimate of $\phi$.

#### Step 5: Measure the Control Register

Measuring the control register, we obtain:

$$
x = 5.
$$

**Estimate of $\phi$:**

$$
\tilde{\phi} = \frac{x}{2^n} = \frac{5}{8} = 0.625.
$$

In this example, the algorithm perfectly recovers the value of $\phi$ due to the choice of $\phi$ being exactly representable with $n$ bits.

### Notes on Precision and Errors

- If $\phi$ cannot be exactly represented with $n$ bits, the algorithm will still provide the closest $n$-bit approximation.
- The probability distribution over $x$ will have its maximum at the integer closest to $2^n \phi$.
- Increasing $n$ increases the resolution and reduces the estimation error.

### Conclusion

The Quantum Phase Estimation algorithm elegantly combines quantum superposition, entanglement, and the Quantum Fourier Transform to estimate the phase $\phi$ associated with the eigenvalues of a unitary operator $U$. By carefully constructing a quantum circuit that encodes $\phi$ into the phases of a quantum state and then extracting that information via the inverse QFT and measurement, the algorithm demonstrates the power of quantum computation in solving problems that are challenging for classical computers.

Through this detailed mathematical exploration, we've unpacked each step of the QPE algorithm, providing clarity on how and why it works. The key takeaways include:

- **Superposition and Entanglement**: Utilized to encode and manipulate phase information across multiple qubits.
- **Quantum Fourier Transform**: Acts as a bridge between the phase-encoded state and a measurable probability distribution.
- **Measurement and Estimation**: Collapses the quantum state to provide an approximate value of $\ph

# Quantum Phase Estimation (QPE) Implementation with PennyLane

This code below implements the Quantum Phase Estimation (QPE) algorithm using the PennyLane library. Below is a detailed summary of what the code does:

## 1. **Define the Hamiltonian**
- The Hamiltonian `H` is defined as a simple Pauli-Z operator for a single qubit.

## 2. **Initialize the Quantum Device**
- A quantum device is initialized with the required number of qubits, including both system and ancilla qubits.

## 3. **Define the Unitary Operator**
- The `unitary_operator` function constructs the unitary matrix $U = \exp(-i \cdot H \cdot t)$ using the matrix exponential.

## 4. **Define the Quantum Circuit**
- The `qpe_circuit` function implements the QPE algorithm:
  - Applies Hadamard gates to the ancilla qubits to create a superposition.
  - Applies controlled-$U$ operations, where $U = \exp(-i \cdot H \cdot t)$.
  - Applies the inverse Quantum Fourier Transform (QFT) to the ancilla qubits.

## 5. **Run the QPE Algorithm**
- The QPE algorithm is executed using the `qpe` QNode.
- The phase (eigenvalue) is extracted from the probability distribution of the ancilla qubits.

## 6. **Plot the Probability Distribution**
- The probability distribution of the ancilla qubits is plotted to visualize the results of the QPE algorithm.

The QPE algorithm estimates the eigenvalue of the Hamiltonian by measuring the phase, which is then used to calculate the eigenvalue. The results are displayed as a probability distribution plot.

## Code Explanation

### Importing Necessary Libraries

```python
import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import expm
```
The code imports `pennylane` for building and simulating quantum circuits, `numpy` for numerical operations, and `scipy.linalg.expm` for matrix exponentiation, which is used to create the unitary operator $U$.

### Defining the Hamiltonian:

```python
H = 0.5 * np.array([[1, 0], [0, -1]])
```

The Hamiltonian $H$ is defined as $0.5 \times \sigma_z$, where $\sigma_z$ is the Pauli-Z matrix. This Hamiltonian is simple and has eigenvalues $\pm 0.5$.

### Setting Up the Quantum Device:

```python
n_system_qubits = 1
n_ancilla_qubits = 3
dev = qml.device("default.qubit", wires=n_system_qubits + n_ancilla_qubits)
```
One system qubit represents the state on which we apply the unitary operator $U$. Three ancilla qubits are used to perform phase estimation, determining the precision of the estimated phase. A PennyLane quantum device (`default.qubit`) is initialized with a total of four wires (qubits).

The use of three ancilla qubits in the Quantum Phase Estimation (QPE) algorithm determines the precision of the estimated phase for the following reasons:

### 1. Binary Representation of the Phase
The goal of QPE is to estimate a phase $\theta$ in the range $[0, 1)$ associated with an eigenvalue of a unitary operator $U$. This phase is typically expressed in binary form as:

$$
\theta = 0.\theta_1 \theta_2 \theta_3 \ldots
$$

where $\theta_1, \theta_2, \theta_3, \ldots$ are binary digits (bits) of the fractional part. Using three ancilla qubits allows us to represent the phase up to three bits of precision:

$$
\theta \approx 0.\theta_1 \theta_2 \theta_3.
$$

This means that the phase estimation is accurate up to $2^{-3} = 0.125$ (or 1/8) in terms of precision.

### 2. Ancilla Qubits Determine the Resolution
Each additional ancilla qubit doubles the number of possible states (or "bins") that can represent the phase. For $n$ ancilla qubits, there are $2^n$ possible states that can be measured, providing a resolution of $1 / 2^n$. Specifically:

- With three ancilla qubits, we have $2^3 = 8$ possible states, meaning the phase can be estimated with a resolution of $1/8$.
- If we increased the number of ancilla qubits, say to four, we would have a resolution of $1/16$.

### 3. Trade-Off Between Precision and Circuit Complexity
Using more ancilla qubits increases the precision of the phase estimation but also makes the quantum circuit more complex. More ancilla qubits require:

- Additional Hadamard gates to initialize the qubits into a superposition state.
- More controlled-unitary operations, where each ancilla qubit applies a controlled unitary with increasing powers.

The complexity of these operations grows with the number of ancilla qubits, leading to more computational resources and potential noise in real quantum hardware. Three ancilla qubits offer a balance between precision and circuit complexity, providing sufficient resolution for many applications while keeping the circuit manageable.

### Defining the Unitary Operator:

```python
def unitary_operator(t):
    return expm(-1j * H * t)
```
The function `unitary_operator` computes $U = e^{-iHt}$ using matrix exponentiation. This operator evolves the state according to the Hamiltonian $H$ over a time $t$.

### Constructing the QPE Circuit:
 Applying Hadamard Gates to the Ancilla Qubits
```python
def qpe_circuit():
    # Apply Hadamard gates to the ancilla qubits
    for i in range(n_ancilla_qubits):
        qml.Hadamard(wires=i)
    
    # Apply controlled-U operations
    for i in range(n_ancilla_qubits):
        qml.ControlledQubitUnitary(unitary_operator(2 ** i), control_wires=i, wires=n_ancilla_qubits)
    
    # Apply inverse Quantum Fourier Transform (QFT)
    qml.adjoint(qml.QFT(wires=range(n_ancilla_qubits)))
```

### Hadamard Gates
The Hadamard gates are applied to create a superposition of all possible states in the ancilla qubits.
What It Does
This part of the code applies Hadamard gates to each of the ancilla qubits. The purpose of applying the Hadamard gate is to create an equal superposition state across all possible basis states for the ancilla qubits.

### Mathematical Explanation
The Hadamard gate transforms the basis state $\vert 0 \rangle$ into:

$$
H \vert 0 \rangle = \frac{1}{\sqrt{2}} (\vert 0 \rangle + \vert 1 \rangle).
$$

For $n$ ancilla qubits, applying the Hadamard gate to each qubit results in a superposition of all $2^n$ states:

$$
\frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n - 1} \vert k \rangle.
$$

This superposition allows the QPE algorithm to encode phase information for all possible outcomes simultaneously through quantum parallelism.


### Controlled Unitaries
Controlled applications of the unitary operator $U$ are performed. The unitary is applied with powers of 2 (i.e., $U, U^2, U^4, \ldots$), controlled by each ancilla qubit.

What It Does
The code applies controlled-unitary operations on the target qubit (which represents the eigenstate $\vert \psi \rangle$) using each ancilla qubit as the control qubit. The unitary operator $U = \exp(-iHt)$ is applied with increasing powers $2^i$ as $i$ ranges over the number of ancilla qubits.

### Mathematical Explanation
For a given ancilla qubit $i$, the controlled-unitary operation applies $U^{2^i}$ to the target state if the control qubit is in the state $\vert 1 \rangle$. Mathematically, the evolution looks like:

$$
\vert k \rangle \otimes \vert \psi \rangle \rightarrow \vert k \rangle \otimes U^k \vert \psi \rangle,
$$

where $k$ is the value represented by the ancilla qubits. Since $\vert \psi \rangle$ is an eigenstate of $U$ with eigenvalue $e^{2 \pi i \theta}$, we have:

$$
U^k \vert \psi \rangle = e^{2 \pi i k \theta} \vert \psi \rangle.
$$

This step effectively encodes the phase $\theta$ into the amplitude of the quantum state, preparing it for extraction using the Quantum Fourier Transform.


### Inverse Quantum Fourier Transform
Finally, an inverse QFT is applied to extract the phase encoded in the amplitudes of the ancilla qubits.
What It Does
This line applies the inverse Quantum Fourier Transform (QFT) to the ancilla qubits. The inverse QFT transforms the state from a superposition that encodes the phase information to a state where measuring the ancilla qubits gives a binary representation of the phase.

### Mathematical Explanation
The QFT maps a computational basis state $\vert k \rangle$ to a superposition state:

$$
\text{QFT} \vert k \rangle = \frac{1}{\sqrt{2^n}} \sum_{m=0}^{2^n - 1} e^{2 \pi i k m / 2^n} \vert m \rangle.
$$

Applying the inverse QFT (denoted by $\text{QFT}^{-1}$) transforms the superposition of phases encoded in the amplitudes of the ancilla qubits into a state where the probability distribution is peaked at values representing the estimated phase $\theta$. This is crucial for extracting the phase information accurately.

### Creating the QNode

```python
@qml.qnode(dev)
def qpe():
    qpe_circuit()
    return qml.probs(wires=range(n_ancilla_qubits))
```
What It Does
This function creates a Quantum Node (QNode) using the PennyLane library. A QNode is a hybrid quantum/classical computational node that can be executed and differentiated. Here, the QNode runs the QPE circuit and returns the measurement probabilities of the ancilla qubits.

Explanation
The `qpe_circuit` function (defined earlier) is called within the QNode. After executing the circuit, the QNode measures the probability distribution of each possible state of the ancilla qubits, providing insight into the estimated phase.

### Running the QPE Algorithm:

```python
probs = qpe()
phase = np.argmax(probs) / (2 ** n_ancilla_qubits)
eigenvalue = 2 * np.pi * phase
```

The probabilities are calculated, and the phase $\theta$ is extracted by finding the index of the state with the highest probability and dividing by $2^{\text{number of ancilla qubits}}$.

The estimated eigenvalue is then computed as $2\pi \times \text{phase}$.



The peak in the plot represents the most likely phase value of the eigenvalue of the Hamiltonian, as estimated by the QPE algorithm.

- The QPE algorithm efficiently estimates the phase associated with the eigenvalue of the unitary operator generated by the Hamiltonian $H$.
- The precision of the estimation depends on the number of ancilla qubits used. In this case, three ancilla qubits provide a resolution of $1/8$ in the phase estimation.















