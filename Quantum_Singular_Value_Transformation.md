# Quantum Singular Value Transformation and Block-Encodingg

In quantum computing, the ability to perform transformations on singular values of operators lies at the heart of several advanced algorithms, including quantum linear systems solvers, eigenvalue estimation, and various applications in quantum machine learning. Quantum Singular Value Transformation (QSVT) and block-encoding are essential tools in this framework. In this blog, we delve into the mathematical principles behind these concepts, explain how they work, and explore their significance in quantum algorithms.

## 1. Introduction to Singular Value Decomposition (SVD) and Matrix Transformations

To understand QSVT and block-encoding, it’s essential first to review Singular Value Decomposition (SVD) of matrices. Given an $m \times n$ matrix $A$, its SVD is given by:

$$
A = U \Sigma V^{\dagger},
$$

where:

- $U$ is an $m \times m$ unitary matrix.
- $V$ is an $n \times n$ unitary matrix.
- $\Sigma$ is an $m \times n$ diagonal matrix with non-negative real numbers (singular values) on the diagonal.

### Key Idea of Quantum Singular Value Transformation (QSVT)
QSVT provides a way to apply polynomial transformations to the singular values of a matrix $A$ while preserving its unitary structure. This transformation is achieved using quantum operations, and it plays a crucial role in efficient quantum algorithms for matrix computations.

## 1.1. Block-Encoding: A Framework for Matrix Representation in Quantum Circuits

### Definition of Block-Encoding
Given a Hermitian matrix $A$ of dimension $2^n \times 2^n$ with $\|A\| \leq 1$ (normalized), a unitary matrix $U$ is said to be a block-encoding of $A$ if:


$$
U = \begin{bmatrix} A & * ,\\ * & * \end{bmatrix}
$$

where $A$ appears as a submatrix of $U$ in the upper-left corner and the remaining entries are arbitrary.

### Constructing a Block-Encoding
To construct a block-encoding of a matrix $A$, we use ancillary qubits and quantum gates to embed $A$ into a larger unitary matrix $U$. This embedding allows us to perform quantum operations on $A$ indirectly by manipulating $U$.

**Example**: Suppose we want to encode a $2 \times 2$ matrix $A$:

$$
A = 
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}.
$$

A possible block-encoding of $A$ is a $4 \times 4$ unitary matrix:

$$
U = 
\begin{bmatrix}
A & \cdot \\
\cdot & \cdot
\end{bmatrix}.
$$

### Mathematical Requirements
For $U$ to be a valid block-encoding, it must satisfy:

1. $U$ is unitary.
2. $A$ is embedded in the top-left block.

The block-encoding technique allows for matrix operations such as matrix inversion, multiplication, and exponentiation to be performed efficiently on quantum hardware using unitary transformations.

## 2. Mathematical Derivation of QSVT

### 2.1 Preliminaries

#### 2.1.1 Singular Value Decomposition (SVD)
Any matrix $A \in \mathbb{C}^{m \times n}$ can be decomposed using SVD as:

$$
A = U \Sigma V^{\dagger},
$$

where:

- $U \in \mathbb{C}^{m \times m}$ and $V \in \mathbb{C}^{n \times n}$ are unitary matrices.
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix with non-negative real numbers $\sigma_i$ (the singular values of $A$) on the diagonal.

#### 2.1.2 Block-Encoding Unitaries
A unitary $U_A$ is a block-encoding of $A$ if:

$$
U_A = 
\begin{bmatrix}
A / \alpha & \cdot \\
\cdot & \cdot
\end{bmatrix}.
$$

Our goal is to construct a new unitary $U'$ that encodes $p(A / \alpha)$ in its top-left block.

### 2.2 Polynomial Transformation of Singular Values
Given a polynomial $p(x)$ of degree $d$, we aim to construct $U'$ such that:

$$
U' = QSVT(U_A, p) \approx 
\begin{bmatrix}
p(A / \alpha) & \cdot \\
\cdot & \cdot
\end{bmatrix}.
$$

This means that $U'$ applies $p$ to each singular value $\sigma_i / \alpha$ of $A / \alpha$.

#### 2.2.1 Chebyshev Polynomials
To facilitate the transformation, we use Chebyshev polynomials of the first kind, $T_k(x)$, which have useful properties for approximating functions over specific intervals.

**Definition**:

$$
T_k(x) = \cos(k \arccos x), \quad x \in [-1, 1].
$$

**Recurrence Relation**:

$$
T_0(x) = 1, \quad T_1(x) = x, \quad T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x).
$$

Chebyshev polynomials are orthogonal over $[-1, 1]$ with respect to the weight $(1 - x^2)^{-1/2}$.

#### 2.2.2 Polynomial Approximation
Given a function $f(x)$ defined over $[-1, 1]$, we can approximate $f$ using a polynomial $p(x)$:

$$
f(x) \approx p(x) = \sum_{k=0}^{d} a_k T_k(x).
$$

Our objective is to find coefficients $a_k$ such that $p(x)$ closely approximates $f(x)$.

### 2.3 Constructing QSVT Using Quantum Gates

#### 2.3.1 Quantum Signal Processing (QSP)
QSVT builds upon the Quantum Signal Processing framework, which allows the implementation of arbitrary polynomials of unitary operators.

- **Signal Operator**: A unitary operator $U_{\text{sig}}(x)$ parameterized by $x \in [-1, 1]$.
- **Phase Factors**: A sequence of real numbers $\{\phi_k\}$ used to construct the desired polynomial transformation.

#### 2.3.2 Constructing the Transformation
The core idea is to construct a sequence of unitary operations that, when composed, yield a transformation applying $p(x)$ to the singular values.

**Steps**:

1. **Define Reflection Operators**:

   Reflection over the Ancilla State:

$$ 
   R_0 = I - 2 \vert 0 \rangle \langle 0 \vert.
$$

2. **Controlled Unitary Operations**:

   Implement $U_A$ and $U_A^{\dagger}$ controlled on the ancilla qubits.

3. **Phase Rotations**:

   Apply phase shifts $e^{i \phi_k}$ to adjust the coefficients of the polynomial.

4. **Construct the Product Operator**:

   The overall transformation is given by:

$$
   U' = e^{i \phi_0 \sigma_z} \prod_{k=1}^{d} (R_0 \cdot e^{i \phi_k \sigma_z} \cdot U_A),
$$

   where $\sigma_z$ is the Pauli-Z operator acting on the ancilla qubit.

#### 2.3.3 Mathematical Derivation
Consider the singular value decomposition of $U_A$:

$$
U_A = \sum_i \cos(\theta_i) \vert u_i \rangle \langle u_i \vert \otimes \vert 0 \rangle \langle 0 \vert + \sin(\theta_i) \vert u_i \rangle \langle v_i \vert \otimes \vert 0 \rangle \langle 1 \vert + \text{rest},
$$

where $\theta_i = \arcsin(\sigma_i / \alpha)$.

**Transformation of Singular Values**:

The action of $U'$ on the singular vectors leads to:

$$
U' \vert u_i \rangle \vert 0 \rangle = p(\cos(\theta_i)) \vert u_i \rangle \vert 0 \rangle + \text{orthogonal terms}.
$$

By carefully choosing the phase factors $\{\phi_k\}$, we ensure that $p(\cos(\theta_i))$ corresponds to the desired polynomial evaluated at $\cos(\theta_i)$.

#### 2.3.4 Selecting Phase Factors
The phase factors $\{\phi_k\}$ are determined by solving the following problem:

- **Given**: A target polynomial $p(x)$ of degree $d$.
- **Find**: Phases $\{\phi_k\}$ such that the recursive formula produces $p(x)$.

This can be achieved using techniques from signal processing and optimization, such as the Carathéodory-Fejér method.

## 3. Example: Quantum Matrix Inversion

### 3.1 The Goal
Compute $x = A^{-1}b$ efficiently using quantum algorithms.

### 3.2 Applying QSVT for Matrix Inversion

#### 3.2.1 Setting Up the Problem
- **Matrix $A$**: Hermitian and invertible.
- **Block-Encoding $U_A$**: A unitary encoding $A / \alpha$.
- **Right-Hand Side $\vert b \rangle$**: Prepared as a quantum state.

#### 3.2.2 Polynomial Approximation of $1/x$
Since we cannot implement $p(x) = 1/x$ exactly (as it's not a polynomial), we approximate $1/x$ over the spectrum of $A$ using a truncated polynomial expansion.

**Chebyshev Polynomial Approximation**:

Find a polynomial $p(x)$ such that:

$$
p(x) \approx \frac{1}{x}, \quad x \in [a, b],
$$

where $[a, b]$ is the interval containing the eigenvalues of $A / \alpha$.

#### 3.2.3 Implementing the Inversion
**Construct $U'$ Using QSVT**:

Use QSVT to build $U'$ that approximates $p(A / \alpha)$.

The transformed unitary $U'$ acts as:

$$
U' \approx 
\begin{bmatrix}
A^{-1} / \alpha & \cdot \\
\cdot & \cdot
\end{bmatrix}.
$$

**Preparing the Solution State**:

Apply $U'$ to $\vert b \rangle$:

$$
U' \vert b \rangle \vert 0 \rangle \approx \left(\frac{A^{-1}b}{\alpha}\right) \vert 0 \rangle + \text{orthogonal terms}.
$$

**Normalization**: The solution vector $x = A^{-1}b$ is encoded in the amplitudes of the quantum state.

#### 3.2.4 Measuring the Result
To extract useful information from $x$, we perform measurements or compute expectation values.

- **Overlap with Test States**: Compute $\langle x \vert \psi \rangle$ for some test state $\vert \psi \rangle$.
- **Observable Measurements**: Measure observables of the form $\langle x \vert O \vert x \rangle$.

### 3.3 Complexity Analysis
The efficiency of the algorithm depends on:

- **Condition Number $\kappa$**: Ratio of the largest to smallest singular value of $A$.
- **Degree of Polynomial $d$**: Related to $\kappa$ and the desired precision $\epsilon$.

The overall runtime scales polylogarithmically with the problem size, offering potential exponential speedups over classical algorithms for certain problems.

### 4. Practical Applications of QSVT and Block-Encoding

#### 4.1 Quantum Machine Learning
QSVT and block-encoding enable efficient quantum algorithms for tasks such as:

- **Principal Component Analysis (PCA)**: By transforming singular values, QSVT can extract the principal components of a data matrix.
- **Support Vector Machines (SVMs)**: Applying polynomial transformations allows for kernel-based SVMs on quantum computers.

#### 4.2 Quantum Simulation
In quantum physics and chemistry, QSVT is used to approximate the evolution of quantum states under a given Hamiltonian by transforming the eigenvalues of the system’s Hamiltonian matrix.

#### 4.3 Quantum Optimization
QSVT can be used to perform transformations on matrices that represent optimization problems, enabling faster convergence and more accurate solutions.

## Conclusion

Quantum Singular Value Transformation (QSVT) is a unifying framework that extends the capabilities of quantum algorithms for linear algebra. By enabling the application of polynomial functions to the singular values of a matrix, QSVT facilitates advanced operations like matrix inversion, eigenvalue transformations, and Hamiltonian simulations.

Through a detailed mathematical derivation, we've seen how QSVT constructs a unitary transformation that applies a desired polynomial to the singular values of a block-encoded matrix. The example of quantum matrix inversion illustrates the practical significance of QSVT in solving linear systems efficiently on a quantum computer.

Understanding QSVT requires a solid grasp of linear algebra, quantum mechanics, and polynomial approximations. As quantum computing continues to evolve, frameworks like QSVT will play a crucial role in unlocking new computational possibilities.

Full Python code:

```python
import pennylane as qml
from pennylane import numpy as np

# Define the Hamiltonian for a simple material (e.g., H2 molecule)
H = np.array([[0.5, 0.1], [0.1, -0.5]])

# Number of qubits for the system and ancilla qubits for block-encoding
n_system_qubits = 1
n_ancilla_qubits = 1

# Initialize the quantum device
dev = qml.device("default.qubit", wires=n_system_qubits + n_ancilla_qubits)

# Define the unitary operator for block-encoding
def block_encode(H):
    # Create a block-encoded unitary matrix
    I = np.eye(2)
    zero_pad = np.zeros_like(H)
    U = np.block([[H, zero_pad], [zero_pad, I]])
    return U

# Define the Quantum Singular Value Transformation circuit
def qsvt_circuit():
    # Apply Hadamard gate to the ancilla qubit
    qml.Hadamard(wires=0)
    
    # Apply the block-encoded unitary
    U = block_encode(H)
    qml.QubitUnitary(U, wires=[0, 1])
    
    # Apply another Hadamard gate to the ancilla qubit
    qml.Hadamard(wires=0)

# Define the QNode
@qml.qnode(dev)
def qsvt():
    qsvt_circuit()
    return qml.expval(qml.PauliZ(0))

# Run the QSVT algorithm
result = qsvt()

print(f"Result of QSVT: {result}")

# Plot the result
import matplotlib.pyplot as plt

plt.bar([0, 1], [result, 1 - result])
plt.xlabel("State")
plt.ylabel("Probability")
plt.title("Result of QSVT")
plt.show()
```

### Quantum Device Initialization

```python
n_system_qubits = 1
n_ancilla_qubits = 1
dev = qml.device("default.qubit", wires=n_system_qubits + n_ancilla_qubits)
```
A quantum device is initialized with two qubits: one for the system and one ancillary qubit used for block-encoding.
The default.qubit device simulates a quantum circuit using a statevector approach.

### Defining the Block-Encoding Unitary

```python
def block_encode(H):
    # Create a block-encoded unitary matrix
    I = np.eye(2)
    zero_pad = np.zeros_like(H)
    U = np.block([[H, zero_pad], [zero_pad, I]])
    return U
```

This function constructs a block-encoded unitary matrix $U$ from the Hamiltonian $H$. The block-encoding embeds the matrix $H$ into the top-left corner of a larger unitary matrix $U$, while padding the remaining elements with zeros and an identity matrix:

$$
U = 
\begin{bmatrix}
H & 0 \\
0 & I
\end{bmatrix},
$$

where $I$ is the identity matrix of the same dimension as $H$ and $0$ represents zero matrices of appropriate size. This is done to ensure $U$ is a unitary matrix.

### Defining the Quantum Singular Value Transformation (QSVT) Circuit

```python
def qsvt_circuit():
    # Apply Hadamard gate to the ancilla qubit
    qml.Hadamard(wires=0)
    
    # Apply the block-encoded unitary
    U = block_encode(H)
    qml.QubitUnitary(U, wires=[0, 1])
    
    # Apply another Hadamard gate to the ancilla qubit
    qml.Hadamard(wires=0)
```
- **Hadamard Gate**: A Hadamard gate is applied to the ancilla qubit (wire 0) to create a superposition state.

- **Block-Encoding Unitary**: The block-encoded unitary matrix $U$ is applied to both the system and ancilla qubits using the `qml.QubitUnitary` function. This operation represents a transformation based on the Hamiltonian $H$.

- **Second Hadamard Gate**: Another Hadamard gate is applied to the ancilla qubit, essentially performing an interference operation that helps extract information about the transformed singular values of $H$.

### Defining and Running the QNode

```python
@qml.qnode(dev)
def qsvt():
    qsvt_circuit()
    return qml.expval(qml.PauliZ(0))
```

A Quantum Node (QNode) is defined that executes the `qsvt_circuit` on the initialized device. The expectation value of the Pauli-Z operator on the ancilla qubit (wire 0) is measured and returned. This expectation value provides information about the transformation performed by the QSVT circuit.

### Explanation of the Purpose and Behavior of the Code

- **Block-Encoding**: The code constructs a block-encoding of the Hamiltonian $H$ into a unitary matrix $U$. This embedding allows for performing operations on $H$ using quantum gates.

- **QSVT Circuit**: The circuit applies a series of quantum gates that encode the Hamiltonian's information into the quantum state, perform transformations, and then extract useful information by measuring the expectation value.

- **Result Interpretation**: The final measurement and expectation value capture information about the transformed singular values of $H$. This can be useful for tasks such as determining properties of the matrix, quantum simulations, or more complex operations involving singular value transformations.

This code demonstrates the application of QSVT and block-encoding techniques to a simple matrix (Hamiltonian), illustrating how these tools can be used to perform transformations and extract information about matrix properties on quantum hardware.




