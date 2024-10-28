# Adiabatic Quantum Computing and Quantum Annealing:

## Introduction:

Adiabatic Quantum Computing (AQC) and Quantum Annealing (QA) are computation methods in quantum mechanics that solve optimization problems by evolving a quantum system adiabatically. This evolution process allows the system to stay in its ground state, gradually transforming from an initial Hamiltonian, where the ground state is easy to prepare, to a final Hamiltonian, which encodes the solution to a specific problem. These methods offer a promising approach to solving combinatorial optimization problems that are intractable for classical algorithms.

This article will provide an in-depth look at the mathematical foundation underlying AQC and QA, exploring their derivations, Hamiltonians, and the physics principles guiding their processes.

## 1. Foundations of Quantum Adiabatic Theorem

In physics, an adiabatic process is one in which there is no exchange of heat or energy between the system and its environment. In quantum mechanics, however, adiabaticity refers to a process where a system undergoes a slow evolution in such a way that it stays in its instantaneous eigenstate (its ground state, for example) despite changes in the Hamiltonian governing it.

### Adiabatic Theorem in Quantum Computing

The adiabatic theorem underpins Adiabatic Quantum Computing (AQC), stating that a quantum system remains in its instantaneous ground state if a Hamiltonian changes slowly enough over time. This can be formulated as follows:

#### 1.1 Adiabatic Theorem
If we have a time-dependent Hamiltonian $H(t)$ that evolves from an initial Hamiltonian $H(0)$ to a final Hamiltonian $H(T)$ over a time interval $T$, then if the evolution is slow, the system remains in its ground state $|\psi_0(t)\rangle$ of $H(t)$.

#### Mathematical Condition for Adiabatic Evolution
The condition for adiabatic evolution can be expressed as:

$$
\frac{|\langle \psi_1(t) | \dot{H}(t) | \psi_0(t) \rangle|}{(E_1(t) - E_0(t))^2} \ll 1
$$

where:

- $|\psi_0(t)\rangle$ and $|\psi_1(t)\rangle$ are the instantaneous ground and first excited states, respectively.
- $E_0(t)$ and $E_1(t)$ are their corresponding eigenvalues.

The rate of change of $H(t)$ must be small relative to the energy gap $E_1(t) - E_0(t)$ to ensure that the system remains in the ground state.

### Adiabatic Quantum Computing

In AQC, two Hamiltonians are defined:

- **Initial Hamiltonian** $H_0$: This Hamiltonian has an easily prepared ground state. It is typically chosen so that its ground state is simple to compute or prepare.
- **Problem Hamiltonian** $H_P$: This Hamiltonian encodes the solution to the problem in its ground state. The ground state of $H_P$ corresponds to the optimal solution for a given optimization problem.

The time-dependent interpolating Hamiltonian $H(t)$ is defined as a linear combination of these two Hamiltonians:

$$
H(t) = (1 - s(t)) H_0 + s(t) H_P,
$$

where $s(t)$ is a continuous function of time that monotonically increases from $s(0) = 0$ to $s(T) = 1$. Here:

- $t = 0$: The system begins with $H(0) = H_0$, where the ground state is easily prepared.
- $t = T$: The evolution ends with $H(T) = H_P$, and the system's ground state at this point corresponds to the solution of the problem.

## 3. Quantum Annealing Process

Quantum Annealing leverages this adiabatic process to solve optimization problems. Instead of a fully controlled adiabatic evolution, Quantum Annealing often involves a finite rate of change, which can lead to non-zero probabilities of exiting the ground state. Quantum Annealing can thus be thought of as a probabilistic optimization technique rather than a fully adiabatic one, balancing computational speed with solution fidelity.

### 3.1 Quantum Annealing Hamiltonian

For QA, the Hamiltonian typically takes the form:

$$
H(t) = \Gamma(t) H_0 + (1 - \Gamma(t)) H_P,
$$

where $\Gamma(t)$ is a control parameter that decreases from $\Gamma(0) = 1$ to $\Gamma(T) = 0$.

## 4. Derivation of Ground State Evolution in AQC and QA

To solve a problem using AQC or QA, we compute the ground state of the final Hamiltonian $H_P$ after adiabatic evolution. The adiabatic theorem guarantees that if $T$ (total evolution time) is large enough, the probability of remaining in the ground state is high. We derive this as follows:

### 4.1 Schrödinger Equation for Time Evolution
The time evolution of the state $|\psi(t)\rangle$ is governed by:

$$
i \hbar \frac{d}{dt} |\psi(t)\rangle = H(t) |\psi(t)\rangle.
$$

### 4.2 Transition Probability Between States
The probability of transitioning to a higher energy state due to non-adiabatic evolution is minimized if the system satisfies the adiabatic condition. Given $T$, the probability of remaining in the ground state $P_{gs}(T)$ depends inversely on the square of the minimum energy gap $\Delta_{min}$ between the ground and first excited state:

$$
P_{gs}(T) \approx 1 - \frac{C}{T \Delta_{min}^2},
$$

where $C$ is a constant related to the rate of change of $H(t)$.

# The Ising Model with Two Spins

Consider an Ising model with two spins ($\sigma_1$ and $\sigma_2$), each of which can be in the spin-up ($|\uparrow\rangle$) or spin-down ($|\downarrow\rangle$) state.

The goal is to find the ground state (minimum energy configuration) of the problem Hamiltonian:

$$
H_P = -J \sigma_{1z} \sigma_{2z}
$$

where:

- $J > 0$ is the coupling constant favoring aligned spins.
- $\sigma_{iz}$ is the Pauli-Z matrix acting on spin $i$.

## Constructing the Hamiltonians

### Pauli Matrices
First, recall the Pauli matrices:

- **Pauli-X ($\sigma_x$):**
  
$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$$
  
- **Pauli-Z ($\sigma_z$):**
  
$$
\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

### Initial Hamiltonian ($H_0$)
The initial Hamiltonian introduces quantum fluctuations:

$$
H_0 = -\Delta (\sigma_{1x} + \sigma_{2x}),
$$

where $\Delta > 0$ controls the strength of the transverse field.

### Problem Hamiltonian ($H_P$)
The problem Hamiltonian encodes the optimization problem:

$$
H_P = -J \sigma_{1z} \sigma_{2z}.
$$

## Time-Dependent Hamiltonian ($H(t)$)
We interpolate between $H_0$ and $H_P$:

$$
H(t) = A(t) H_0 + B(t) H_P,
$$

with boundary conditions:
- $A(0) = 1$, $B(0) = 0$ at $t = 0$,
- $A(T) = 0$, $B(T) = 1$ at $t = T$.

For simplicity, let $A(t) = 1 - s(t)$ and $B(t) = s(t)$, where $s(t)$ varies smoothly from 0 to 1.

## Matrix Representation of the Hamiltonians

### Basis States
We will use the computational basis:

- $|\uparrow\uparrow\rangle = |0\rangle \otimes |0\rangle$
- $|\uparrow\downarrow\rangle = |0\rangle \otimes |1\rangle$
- $|\downarrow\uparrow\rangle = |1\rangle \otimes |0\rangle$
- $|\downarrow\downarrow\rangle = |1\rangle \otimes |1\rangle$

In vector form:

$$
|\uparrow\uparrow\rangle = \begin{pmatrix} 1 \\
0 \\
0 \\
0 \end{pmatrix}, \quad
|\uparrow\downarrow\rangle = \begin{pmatrix} 0 \\
1 \\
0 \\
0 \end{pmatrix}, \quad
|\downarrow\uparrow\rangle = \begin{pmatrix} 0 \\
0 \\
1 \\
0 \end{pmatrix}, \quad
|\downarrow\downarrow\rangle = \begin{pmatrix} 0 \\
0 \\
0 \\
1 \end{pmatrix}
$$

### Operators Acting on Two Spins
The Pauli matrices act on individual spins. For two spins, we use the Kronecker product:

- $\sigma_{1x} = \sigma_x \otimes I$
- $\sigma_{2x} = I \otimes \sigma_x$
- $\sigma_{1z} = \sigma_z \otimes I$
- $\sigma_{2z} = I \otimes \sigma_z$

where $I$ is the 2x2 identity matrix:

$$
I = \begin{pmatrix} 1 & 0 \\
0 & 1 \end{pmatrix}
$$

# Calculating $H_0$

## Compute $\sigma_{1x}$ and $\sigma_{2x}$

1. **For $\sigma_{1x}$**:
   $$
   \sigma_{1x} = \sigma_x \otimes I
   $$

   Calculating $\sigma_x \otimes I$:

   $$
   \sigma_x = \begin{pmatrix} 0 & 1 \\
    1 & 0 \end{pmatrix}, \quad I = \begin{pmatrix} 1 & 0 \\
   0 & 1 \end{pmatrix}
   $$

   Therefore:

   $$
   \sigma_x \otimes I = \begin{pmatrix} 0 & 1 \\
    1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\
    0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \end{pmatrix}
   $$

3. **For $\sigma_{2x}$**:
   $$
   \sigma_{2x} = I \otimes \sigma_x
   $$

   Calculating $I \otimes \sigma_x$:

   $$
   I = \begin{pmatrix} 1 & 0 \\
    0 & 1 \end{pmatrix}, \quad \sigma_x = \begin{pmatrix} 0 & 1 \\
    1 & 0 \end{pmatrix}
   $$

   Therefore:

   $$
   I \otimes \sigma_x = \begin{pmatrix} 1 & 0 & 0 & 1 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \end{pmatrix}
   $$

### Total $H_0$
The initial Hamiltonian $H_0$ is given by:

$$
H_0 = -\Delta (\sigma_{1x} + \sigma_{2x})
$$

Expanding:

$$
H_0 = -\Delta \left( \begin{pmatrix} 0 & 1 & 1 & 0 \\
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \end{pmatrix} \right)
$$

# Calculating $H_P$

## Compute $\sigma_{1z} \sigma_{2z}$

1. **For $\sigma_{1z} \sigma_{2z}$**:
   $$
   \sigma_{1z} \sigma_{2z} = (\sigma_z \otimes I)(I \otimes \sigma_z) = \sigma_z \otimes \sigma_z
   $$

   Calculate $\sigma_z \otimes \sigma_z$:

   $$
   \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
   $$

   Therefore:

   $$
   \sigma_z \otimes \sigma_z = \begin{pmatrix} 1 & 0 & 0 & -1 \\
    0 & -1 & 0 & 0 \\
    0 & 0 & -1 & 0 \\
    -1 & 0 & 0 & 1 \end{pmatrix}
   $$

### Total $H_P$

Thus, the problem Hamiltonian $H_P$ is:

$$
H_P = -J \sigma_{1z} \sigma_{2z} = -J (\sigma_z \otimes \sigma_z) = -J \begin{pmatrix} 1 & 0 & 0 & -1 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
-1 & 0 & 0 & 1 \end{pmatrix}
$$

# Eigenvalues and Eigenvectors

## For $H_P$
To find the eigenvalues and eigenvectors of $H_P$, observe:

$$
H_P = -J \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} -J & 0 & 0 & 0 \\
0 & J & 0 & 0 \\
0 & 0 & J & 0 \\
0 & 0 & 0 & -J \end{pmatrix}
$$

### Eigenvalues and Eigenvectors
The eigenvalues are the diagonal elements:

- Eigenvalue $-J$ with eigenvectors $|\uparrow\uparrow\rangle$ and $|\downarrow\downarrow\rangle$.
- Eigenvalue $J$ with eigenvectors $|\uparrow\downarrow\rangle$ and $|\downarrow\uparrow\rangle$.

#### Ground States of $H_P$
The ground states correspond to the lowest eigenvalue ($-J$):

$$
|\psi_{\text{ground}}\rangle = |\uparrow\uparrow\rangle, |\downarrow\downarrow\rangle
$$

These states have aligned spins, as expected.

# For $H_0$

## Matrix of $H_0$

The matrix of $H_0$ is given by:

$$
H_0 = -\Delta \begin{pmatrix} 0 & 1 & 1 & 0 \\
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0 \end{pmatrix}
$$

### Eigenvalues and Eigenvectors

We find the eigenvalues of this matrix by solving:

$$
\det(H_0 - \lambda I) = 0
$$

The eigenvalues of $H_0$ are:

- $\lambda_1 = -2\Delta$ with eigenvector $\frac{1}{2}(|\uparrow\uparrow\rangle + |\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle + |\downarrow\downarrow\rangle)$.
- $\lambda_2 = 0$ (degenerate)
- $\lambda_3 = 0$ (degenerate)
- $\lambda_4 = 2\Delta$ with eigenvector $\frac{1}{2}(|\uparrow\uparrow\rangle - |\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle + |\downarrow\downarrow\rangle)$

#### Ground State of $H_0$
The ground state corresponds to the lowest eigenvalue ($-2\Delta$):

$$
|\psi(0)\rangle = \frac{1}{2}(|\uparrow\uparrow\rangle + |\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle + |\downarrow\downarrow\rangle)
$$

This is an equal superposition of all basis states.
# Time Evolution of the System

## Time-Dependent Schrödinger Equation
The system evolves according to:

$$
i \hbar \frac{d}{dt} |\psi(t)\rangle = H(t) |\psi(t)\rangle
$$

For simplicity, set $\hbar = 1$.

## Discretizing Time Evolution
For a small time step $\delta t$, the evolution can be approximated using the time evolution operator:

$$
|\psi(t + \delta t)\rangle = e^{-i H(t) \delta t} |\psi(t)\rangle
$$

Since $H(t)$ changes with time, we need to update $H(t)$ at each time step.

## Numerical Simulation
Let's perform a numerical simulation with the following parameters:

- Total time $T$
- Number of time steps $N$
- Time step $\delta t = T / N$
- Vary $s(t) = t / T$ linearly from 0 to 1

### Parameters:
- $\Delta = 1$
- $J = 1$
- $T = 10$
- $N = 1000$
- $\delta t = 0.01$

### Implementing the Simulation

1. **Initialize the State:**

$$
|\psi(0)\rangle = \frac{1}{2}(|\uparrow\uparrow\rangle + |\uparrow\downarrow\rangle + |\downarrow\uparrow\rangle + |\downarrow\downarrow\rangle)
$$

In vector form:

$$
|\psi(0)\rangle = \frac{1}{2} \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}
$$

2. **Loop Over Time Steps:**

   For each time step $t_i = i \delta t$:
   - Compute $s(t_i) = t_i / T$
   - Compute $H(t_i) = (1 - s(t_i)) H_0 + s(t_i) H_P$
   - Compute $U(t_i) = e^{-i H(t_i) \delta t}$ (matrix exponential)
   - Update the state: $|\psi(t_{i+1})\rangle = U(t_i) |\psi(t_i)\rangle$

### Observables
At each step, we can compute observables such as the probability of finding the system in each basis state.

## Calculating $U(t_i)$
The exponential of a matrix $M$ can be calculated using:

$$
e^{-i M \delta t} = \sum_{n=0}^{\infty} \frac{(-i M \delta t)^n}{n!}
$$

For small $\delta t$, we can approximate:

$$
U(t_i) \approx I - i H(t_i) \delta t
$$

For better accuracy, use a method like eigendecomposition or `numpy.linalg.expm` if programming.

## Final State
After evolving the system, the final state $|\psi(T)\rangle$ should approximate the ground state of $H_P$:

$$
|\psi(T)\rangle \approx \frac{1}{\sqrt{2}}(|\uparrow\uparrow\rangle + |\downarrow\downarrow\rangle)
$$

## Example Calculation for a Single Time Step
Let's perform a single time step to illustrate the calculation.

1. **At $t = 0$:**
   - $s(0) = 0$
   - $H(0) = H_0$

2. **Compute $U(0)$:**

   Using matrix exponential:

$$
U(0) = e^{-i H_0 \delta t}
$$

   Given $H_0$ and $\delta t$, compute $U(0)$ numerically.

3. **Update State:**

$$
|\psi(\delta t)\rangle = U(0) |\psi(0)\rangle
$$

   Compute $|\psi(\delta t)\rangle$ by multiplying $U(0)$ with $|\psi(0)\rangle$.

## Interpretation of Results
After performing the simulation, we can analyze:

- **Probability Distribution**: The probabilities $P_\alpha = |\langle \alpha | \psi(t) \rangle|^2$ for each basis state $|\alpha\rangle$.
- **Convergence to Ground State**: As $t \to T$, the probabilities for $|\uparrow\uparrow\rangle$ and $|\downarrow\downarrow\rangle$ increase, while others decrease.
- **Energy Expectation Value**: Compute $\langle \psi(t) | H_P | \psi(t) \rangle$ to see the system's energy decreasing toward the ground state energy.

## Analytical Solution for Small Systems
For a system as small as two spins, we can, in fact, solve the time evolution analytically.

### Diagonalization of $H(t)$
At each time $t$, diagonalize $H(t)$:

1. Find eigenvalues $E_n(t)$ and eigenvectors $|\phi_n(t)\rangle$.
2. Express the initial state $|\psi(0)\rangle$ in terms of $|\phi_n(0)\rangle$.
3. Evolve each component:

$$
|\psi(t)\rangle = \sum_n c_n e^{-i \int_0^t E_n(t') \, dt'} |\phi_n(t)\rangle
$$

Since $H(t)$ changes with $t$, this method requires solving differential equations for $E_n(t)$ and $|\phi_n(t)\rangle$, which can be complex.


