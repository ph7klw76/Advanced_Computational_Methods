# Quantum Approximate Optimization Algorithm (QAOA)

One of the most exciting applications of quantum computing is optimization, where the goal is to find the best solution from a set of possible solutions. The Quantum Approximate Optimization Algorithm (QAOA) is a quantum algorithm designed for such tasks.

## Max-Cut Problem

The Max-Cut Problem is a classic challenge in computer science and mathematics that can be understood as an exercise in dividing a group of people into two teams to maximize the number of friendships that are split between the teams. Imagine a network or graph where each node represents a person, and each edge (or line) between two nodes represents a friendship connecting those two people. The goal of the Max-Cut Problem is to split this group of people into two teams so that as many friendships as possible are "cut" across the teams—that is, friends end up on opposite teams.

For example, consider a small group of four friends: Alice, Bob, Carol, and Dave. Alice is friends with Bob and Carol, while Bob and Carol are both friends with Dave. Representing this situation as a graph, each person is a node, and each friendship is an edge connecting two nodes. The objective now is to divide Alice, Bob, Carol, and Dave into two teams so that the maximum number of friendships are split between the two teams, maximizing the "cut." In graph terms, a "cut" divides the nodes into two groups and counts the edges (friendships) that go between them. The "Max-Cut" is simply the division that yields the maximum number of these cross-group connections.

While this problem is relatively easy to solve by trial and error for small groups, it becomes increasingly complex for larger groups with more connections, making it a popular and challenging problem in optimization. Beyond theory, the Max-Cut Problem has practical applications in various fields, such as designing efficient communication networks by dividing a network into sub-networks to reduce interference, analyzing social networks by identifying tightly connected communities, and solving clustering problems in machine learning. Essentially, the Max-Cut Problem is about splitting a group into two parts to maximize cross-connections, and its simplicity belies the complex optimization challenges it presents, especially as network sizes increase.

# Quantum Approximate Optimization Algorithm (QAOA)

QAOA is designed to find approximate solutions to combinatorial optimization problems using quantum computers.

## Cost and Mixer Hamiltonians

In quantum mechanics, a Hamiltonian represents the total energy of a system.

### Cost Hamiltonian ($H_C$)

- **Purpose**: Encodes the optimization problem into a quantum system.
- **Construction**: For a given problem, $H_C$ is defined such that its ground state (lowest energy state) corresponds to the optimal solution.

### Mixer Hamiltonian ($H_M$)

- **Purpose**: Introduces transitions between different states to explore the solution space.
- **Common Choice**: $H_M = \sum_i X_i$, where $X_i$ is the Pauli-X operator acting on qubit $i$.

## The Variational Approach

QAOA uses a variational method:

1. **Initialize**: Start with an equal superposition of all possible states.
2. **Apply Unitaries**: Alternately apply unitaries derived from $H_C$ and $H_M$.
3. **Measure**: After several layers, measure the qubits to obtain a solution.
4. **Optimize Parameters**: Adjust the parameters in the unitaries to maximize the probability of finding the optimal solution.

## QAOA Circuit

The quantum circuit for QAOA with $p$ layers consists of:

1. **Initialization**: Apply Hadamard gates to all qubits to create a superposition.
2. **Alternating Operators**:
   - **Cost Operator**: $U_C(\gamma) = e^{-i \gamma H_C}$
   - **Mixer Operator**: $U_M(\beta) = e^{-i \beta H_M}$
3. **Repeat**: Apply $U_C$ and $U_M$ alternately $p$ times with parameters $\gamma_k$ and $\beta_k$.
4. **Measurement**: Measure the final state to obtain a candidate solution.

## Mathematical Explanation

### Step-by-Step Derivation

#### Initialization

Start with the state:

$$
|\psi_0\rangle = \bigotimes_{i=1}^n \frac{|0\rangle_i + |1\rangle_i}{\sqrt{2}}
$$

This represents an equal superposition of all $2^n$ possible states.

#### Applying the Cost Operator

For layer $k$, apply the cost operator:

$$
U_C(\gamma_k) = e^{-i \gamma_k H_C}
$$

This operation adds a phase to each computational basis state based on its cost.

#### Applying the Mixer Operator

Next, apply the mixer operator:

$$
U_M(\beta_k) = e^{-i \beta_k H_M}
$$

This mixes the states, allowing exploration of the solution space.

### Complete State After $p$ Layers

After $p$ layers, the state is:

$$
|\psi_p(\gamma, \beta)\rangle = U_M(\beta_p) U_C(\gamma_p) \dots U_M(\beta_1) U_C(\gamma_1) |\psi_0\rangle
$$

where $\gamma = (\gamma_1, \gamma_2, \dots, \gamma_p)$ and $\beta = (\beta_1, \beta_2, \dots, \beta_p)$.

### Expected Value

The goal is to maximize the expected value of the cost Hamiltonian:

$$
\langle C \rangle = \langle \psi_p | H_C | \psi_p \rangle
$$

By adjusting the parameters $\gamma$ and $\beta$, we aim to maximize $\langle C \rangle$, increasing the probability of measuring the optimal solution.

## Parameter Optimization

The optimization of parameters is performed classically:

1. **Define Objective Function**: $f(\gamma, \beta) = \langle \psi_p | H_C | \psi_p \rangle$
2. **Optimization Algorithm**: Use a classical optimizer (e.g., gradient descent, Nelder-Mead) to find $\gamma$*
   and $\beta$* that maximize $f$.
4. **Iterative Process**: The quantum circuit is executed multiple times to evaluate $f$ at different parameter values.

# Examples

Let's solidify our understanding with some examples.

## Two-Qubit System

Consider a simple system with two qubits.

### Problem Definition

Suppose we want to maximize the function:

$$
C(z_1, z_2) = z_1 z_2
$$

where $z_i \in \{-1, 1\}$.

### Constructing the Cost Hamiltonian

Express $z_i$ in terms of Pauli-Z operators:

$$
Z_i |z_i\rangle = z_i |z_i\rangle
$$

So the cost Hamiltonian is:

$$
H_C = Z_1 Z_2
$$

### Mixer Hamiltonian

Using Pauli-X operators:

$$
H_M = X_1 + X_2
$$

### Initialization

Starting state:

$$
|\psi_0\rangle = \frac{1}{2} (|00\rangle + |01\rangle + |10\rangle + |11\rangle)
$$

### Applying QAOA with $p=1$

- **Apply Cost Operator**:
  
$$
U_C(\gamma) = e^{-i \gamma Z_1 Z_2}
$$

**Apply Mixer Operator**:

$$
U_M(\beta) = e^{-i \beta (X_1 + X_2)}
$$

### Final State

Compute the final state:

$$
|\psi_1(\gamma, \beta)\rangle = U_M(\beta) U_C(\gamma) |\psi_0\rangle
$$

### Expected Value

Compute:

$$
\langle C \rangle = \langle \psi_1 | Z_1 Z_2 | \psi_1 \rangle
$$

### Parameter Optimization

Adjust $\gamma$ and $\beta$ to maximize $\langle C \rangle$. For this simple case, we can plot $\langle C \rangle$ as a function of $\gamma$ and $\beta$ to find the optimal values.

---

## Max-Cut Problem on a Simple Graph

Consider a graph with three vertices connected in a triangle.

### Graph Definition

- **Vertices**: $V = \{1, 2, 3\}$
- **Edges**: $E = \{\{1, 2\}, \{2, 3\}, \{1, 3\}\}$

### Objective

Partition $V$ into two sets to maximize the number of edges between them.

### Cost Hamiltonian

Define:

$$
H_C = \sum_{\{i, j\} \in E} \frac{1}{2} (1 - Z_i Z_j)
$$

This Hamiltonian assigns energy to cuts between different partitions.

### Mixer Hamiltonian

$$
H_M = \sum_{i=1}^3 X_i
$$

### QAOA Circuit for $p=1$

- **Initialization**:
  
$$
|\psi_0\rangle = \frac{1}{\sqrt{8}} \sum_{z \in \{0,1\}^3} |z\rangle
$$

- **Apply Cost Operator**:

$$
U_C(\gamma) = e^{-i \gamma H_C}
$$

- **Apply Mixer Operator**:

$$
U_M(\beta) = e^{-i \beta H_M}
$$

### Final State

$$
|\psi_1(\gamma, \beta)\rangle = U_M(\beta) U_C(\gamma) |\psi_0\rangle
$$

### Expected Value

Compute:

$$
\langle C \rangle = \langle \psi_1 | H_C | \psi_1 \rangle
$$

### Parameter Optimization

- **Visualization**: Plot $\langle C \rangle$ over a range of $\gamma$ and $\beta$.
- **Optimal Parameters**: Find $(\gamma^*, \beta^*)$ that maximize $\langle C \rangle$.

---

## Measuring and Interpreting Results

- **Measurement Outcomes**: After running the circuit with optimal parameters, measure the qubits multiple times.
- **Interpretation**: The bit strings correspond to partitions. The most frequent ones are the approximate solutions to the Max-Cut problem.


