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
- **Optimal Parameters**: Find $\gamma\$*, $\beta$ * that maximize $\langle C \rangle$.

---

## Measuring and Interpreting Results

- **Measurement Outcomes**: After running the circuit with optimal parameters, measure the qubits multiple times.
- **Interpretation**: The bit strings correspond to partitions. The most frequent ones are the approximate solutions to the Max-Cut problem.

### QAOA Implementation with PennyLane


# Max-Cut Problem Solution Using QAOA

The code is designed to solve the Max-Cut problem on a simple graph using the Quantum Approximate Optimization Algorithm (QAOA). The Max-Cut problem involves partitioning the nodes of a graph into two sets such that the number of edges between the sets is maximized.

## Key Components

- **Graph Definition**: A simple triangle graph is defined.
  
- **Quantum Device Setup**: A quantum device is initialized using PennyLane.

- **Hamiltonians Construction**: Cost and mixer Hamiltonians are defined to encode the problem and facilitate state transitions.

- **QAOA Circuit Implementation**: The QAOA algorithm is implemented using quantum gates corresponding to the Hamiltonians.

- **Parameter Optimization**: Classical optimization is used to find the best parameters for the quantum circuit.

- **Sampling and Interpretation**: The optimized circuit is run to obtain samples, which are interpreted to find solutions to the Max-Cut problem.


# Defining the Graph for the Max-Cut Problem

The full code is below:

```python
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the graph for the Max-Cut problem
# We'll use a simple triangle graph (3 nodes connected in a cycle)
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (0, 2)])

# Visualize the graph
nx.draw(G, with_labels=True)
plt.show()

# Number of nodes (qubits)
n_qubits = G.number_of_nodes()

# Define the device
dev = qml.device('default.qubit', wires=n_qubits, shots=1000)

# Helper function to create the cost Hamiltonian
def cost_hamiltonian(graph):
    """Creates the cost Hamiltonian for the Max-Cut problem."""
    coeffs = []
    observables = []
    for (i, j) in graph.edges():
        coeffs.append(0.5)
        observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, observables)

# Helper function to create the mixer Hamiltonian
def mixer_hamiltonian(n_qubits):
    """Creates the mixer Hamiltonian."""
    coeffs = [-1.0] * n_qubits
    observables = [qml.PauliX(i) for i in range(n_qubits)]
    return qml.Hamiltonian(coeffs, observables)

# Define the QAOA cost function
def qaoa_cost(params):
    gamma, beta = params
    @qml.qnode(dev)
    def circuit():
        # Apply Hadamard gates to all qubits
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Cost Hamiltonian evolution
        qml.ApproxTimeEvolution(cost_h, gamma, 1)
        
        # Mixer Hamiltonian evolution
        qml.ApproxTimeEvolution(mixer_h, beta, 1)
        
        # Measurement (expectation value of the cost Hamiltonian)
        return qml.expval(cost_h)
    return circuit()

# Create the cost and mixer Hamiltonians
cost_h = cost_hamiltonian(G)
mixer_h = mixer_hamiltonian(n_qubits)

# Initialize parameters
np.random.seed(42)
params = np.random.uniform(0, np.pi, 2, requires_grad=True)

# Optimize the parameters to minimize the cost function
opt = qml.GradientDescentOptimizer(stepsize=0.1)
max_iterations = 100
for i in range(max_iterations):
    params, cost = opt.step_and_cost(qaoa_cost, params)
    if i % 10 == 0:
        print(f"Iteration {i}: Cost = {cost}")

print(f"Optimized parameters: gamma = {params[0]}, beta = {params[1]}")

# Run the circuit with optimized parameters and sample results
@qml.qnode(dev)
def optimized_circuit():
    # Apply Hadamard gates to all qubits
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Cost Hamiltonian evolution
    qml.ApproxTimeEvolution(cost_h, params[0], 1)
    
    # Mixer Hamiltonian evolution
    qml.ApproxTimeEvolution(mixer_h, params[1], 1)
    
    # Measurement
    return qml.sample(wires=range(n_qubits))

# Sample the circuit multiple times to get probable solutions
samples = optimized_circuit()
print("Sampled bitstrings:")
print(samples)

# Interpret the results
def bitstring_to_cut(bitstring):
    """Converts a bitstring to a cut value."""
    cut_value = 0
    for (i, j) in G.edges():
        if bitstring[i] != bitstring[j]:
            cut_value += 1
    return cut_value

# Evaluate the cut values of the sampled bitstrings
cut_values = [bitstring_to_cut(sample) for sample in samples]
print("Cut values of the sampled bitstrings:")
print(cut_values)
```


```python
# Define the graph for the Max-Cut problem
# We'll use a simple triangle graph (3 nodes connected in a cycle)
import networkx as nx

G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (0, 2)])
```

# Explanation of the codes

# Creating the Cost Hamiltonian

```python
# Helper function to create the cost Hamiltonian
def cost_hamiltonian(graph):
    """Creates the cost Hamiltonian for the Max-Cut problem."""
    coeffs = []
    observables = []
    for (i, j) in graph.edges():
        coeffs.append(0.5)
        observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, observables)
```
### Purpose

The cost Hamiltonian encodes the Max-Cut problem into the quantum circuit.

### Hamiltonian Terms

- **Coefficients (`coeffs`)**: Each term has a coefficient of $0.5$.
- **Observables (`observables`)**: For each edge $(i, j)$, we include the term $Z_i Z_j$, where $Z_i$ is the Pauli-Z operator acting on qubit $i$.

### Explanation

- **Pauli-Z Operator (`qml.PauliZ`)**: Measures the spin along the Z-axis.
- **Tensor Product (`@`)**: Combines two operators acting on different qubits.

### Hamiltonian Construction

The cost Hamiltonian is:

$$
H_C = \sum_{\langle i, j \rangle} \frac{1}{2} Z_i Z_j
$$

where the sum is over all edges in the graph.

### Creating the Mixer Hamiltonian

# Helper function to create the mixer Hamiltonian

```python
def mixer_hamiltonian(n_qubits):
    """Creates the mixer Hamiltonian."""
    coeffs = [-1.0] * n_qubits
    observables = [qml.PauliX(i) for i in range(n_qubits)]
    return qml.Hamiltonian(coeffs, observables)
```

### Purpose

The mixer Hamiltonian facilitates transitions between different computational basis states.

### Hamiltonian Terms

- **Coefficients (`coeffs`)**: Each term has a coefficient of $-1.0$.
- **Observables (`observables`)**: Includes the Pauli-X operator for each qubit.

### Explanation

- **Pauli-X Operator (`qml.PauliX`)**: Acts as a bit-flip gate, flipping the state of a qubit from $|0\rangle$ to $|1\rangle$ and vice versa.

### Defining the QAOA Cost Function

```python
# Define the QAOA cost function
def qaoa_cost(params):
    gamma, beta = params
    @qml.qnode(dev)
    def circuit():
        # Apply Hadamard gates to all qubits
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Cost Hamiltonian evolution
        qml.ApproxTimeEvolution(cost_h, gamma, 1)
        
        # Mixer Hamiltonian evolution
        qml.ApproxTimeEvolution(mixer_h, beta, 1)
        
        # Measurement (expectation value of the cost Hamiltonian)
        return qml.expval(cost_h)
    return circuit()
```

### Parameters

- **Parameters (`params`)**: A tuple containing $\gamma$ and $\beta$, the angles (parameters) for the unitaries derived from the Hamiltonians.

### Quantum Node

- **Quantum Node (`@qml.qnode`)**: Decorator that turns the circuit function into a quantum node compatible with PennyLane's autodifferentiation.

### Circuit Steps

#### Initialization

- **Hadamard Gates (`qml.Hadamard`)**: Applied to each qubit to create an equal superposition of all possible states.
  - **Explanation**: The Hadamard gate transforms $|0\rangle$ into $(|0\rangle + |1\rangle) / \sqrt{2}$, setting up the initial state for QAOA.

#### Cost Hamiltonian Evolution

- **Time Evolution (`qml.ApproxTimeEvolution`)**: Simulates the unitary evolution under the cost Hamiltonian $H_C$ for time $\gamma$.
- **Unitary Operator**:

$$
  U_C(\gamma) = e^{-i \gamma H_C}
$$

  - **Explanation**: This operation adds phases to the computational basis states based on their cost, which interferes constructively or destructively to amplify the probability of measuring optimal solutions.

#### Mixer Hamiltonian Evolution

- **Time Evolution**: Simulates the unitary evolution under the mixer Hamiltonian $H_M$ for time $\beta$.
- **Unitary Operator**:

$$
  U_M(\beta) = e^{-i \beta H_M}
$$

  - **Explanation**: This operation mixes the amplitudes between different states, allowing the algorithm to explore the solution space.

#### Measurement

- **Expectation Value (`qml.expval(cost_h)`)**: Measures the expectation value of the cost Hamiltonian, which serves as the cost function to be minimized during optimization.
- **Explanation**: By minimizing the expectation value, we aim to find parameters that maximize the probability of measuring states corresponding to high-quality solutions.


### Creating the Cost and Mixer Hamiltonians

```python
# Create the cost and mixer Hamiltonians
cost_h = cost_hamiltonian(G)
mixer_h = mixer_hamiltonian(n_qubits)
```

- **`cost_h`**: Instance of the cost Hamiltonian for the given graph.
- **`mixer_h`**: Instance of the mixer Hamiltonian for the specified number of qubits.

### Initializing Parameters

```python
# Initialize parameters
np.random.seed(42)
params = np.random.uniform(0, np.pi, 2, requires_grad=True)
```

- **Random Initialization**: Parameters $\gamma$ and $\beta$ are initialized randomly between $0$ and $\pi$.
- **`requires_grad=True`**: Indicates that these parameters should be tracked for automatic differentiation.

### Optimizing the Parameters

```python
# Optimize the parameters to minimize the cost function
opt = qml.GradientDescentOptimizer(stepsize=0.1)
max_iterations = 100
for i in range(max_iterations):
    params, cost = opt.step_and_cost(qaoa_cost, params)
    if i % 10 == 0:
        print(f"Iteration {i}: Cost = {cost}")

print(f"Optimized parameters: gamma = {params[0]}, beta = {params[1]}")
```

- **Optimizer (`opt`)**: Uses gradient descent with a specified step size to minimize the cost function.

### Optimization Loop

- **`opt.step_and_cost`**: Performs one optimization step and returns the updated parameters and the cost value.
- **Iteration Logging**: Every 10 iterations, the current cost is printed.

### Result

- **Optimized Parameters**: The optimized parameters $\gamma^*$ and $\beta^*$ are obtained after the optimization loop.

### Running the Optimized Circuit and Sampling Results


```python
# Run the circuit with optimized parameters and sample results
@qml.qnode(dev)
def optimized_circuit():
    # Apply Hadamard gates to all qubits
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Cost Hamiltonian evolution
    qml.ApproxTimeEvolution(cost_h, params[0], 1)
    
    # Mixer Hamiltonian evolution
    qml.ApproxTimeEvolution(mixer_h, params[1], 1)
    
    # Measurement
    return qml.sample(wires=range(n_qubits))

# Sample the circuit multiple times to get probable solutions
samples = optimized_circuit()
print("Sampled bitstrings:")
print(samples)
```

- **Optimized Circuit**: Uses the optimized parameters to construct the final quantum circuit for sampling.

### Sampling

- **`qml.sample`**: Measures the qubits, returning samples (bitstrings) representing possible solutions.
- **Shots**: Since the device was initialized with `shots=1000`, this circuit execution will return 1000 samples.

### Note on Measurement

- **Measurement Basis**: By default, `qml.sample` measures in the computational basis ($|0\rangle$ and $|1\rangle$).
- **Result Interpretation**: Each bitstring corresponds to a possible partitioning of the graph's nodes.

### Interpreting the Results

```python
# Interpret the results
def bitstring_to_cut(bitstring):
    """Converts a bitstring to a cut value."""
    cut_value = 0
    for (i, j) in G.edges():
        if bitstring[i] != bitstring[j]:
            cut_value += 1
    return cut_value

# Evaluate the cut values of the sampled bitstrings
cut_values = [bitstring_to_cut(sample) for sample in samples]
print("Cut values of the sampled bitstrings:")
print(cut_values)
```
- **Function `bitstring_to_cut`**: Calculates the cut value (the number of edges crossing the partition) for a given bitstring.

### Evaluating Samples

- **Cut Value Calculation**: The cut values for all sampled bitstrings are calculated.
- **Interpretation**: Bitstrings with higher cut values represent better solutions to the Max-Cut problem.

### Quantum Gates Used in the Code

- **Hadamard Gates**: Used to create an equal superposition of all possible states, initializing the quantum circuit for QAOA.

```python
for i in range(n_qubits):
    qml.Hadamard(wires=i)
```

- **Purpose**: Initializes each qubit in a superposition state, creating an equal probability of being in $|0\rangle$ or $|1\rangle$.

### Mathematical Operation

$$
H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad H|1\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}
$$

### Cost Hamiltonian Evolution

```python
qml.ApproxTimeEvolution(cost_h, gamma, 1)
```

- **Purpose**: Simulates the unitary evolution under the cost Hamiltonian $H_C$ for time $\gamma$.

### Underlying Operation

Applies the operator:

$$
U_C(\gamma) = e^{-i \gamma H_C}
$$

### Implementation

- **Approximation**: Since simulating arbitrary Hamiltonian evolution can be complex, `ApproxTimeEvolution` approximates the evolution using techniques like Trotterization.
- **Decomposition into Gates**: The evolution under each term in the Hamiltonian is implemented using quantum gates.

### Decomposition

For each term $Z_i Z_j$ in $H_C$:

- **Controlled-Z Gates**: Implemented using controlled-phase gates.
- **Rotation Gates**: Single-qubit rotations may be used when the Hamiltonian includes single-qubit terms.

### Mixer Hamiltonian Evolution

```python
qml.ApproxTimeEvolution(mixer_h, beta, 1)
```

- **Purpose**: Simulates the unitary evolution under the mixer Hamiltonian $H_M$ for time $\beta$.

### Underlying Operation

Applies the operator:

$$
U_M(\beta) = e^{-i \beta H_M}
$$

### Implementation

- **Pauli-X Rotations**: Each term $-X_i$ in the mixer Hamiltonian corresponds to a rotation around the X-axis.

### Quantum Gates Used

```python
qml.RX(2 * beta, wires=i)
```
***Explanation***: The evolution under $-X_i$ for time $\beta$ is equivalent to applying an $RX$ rotation of angle $2\beta$.


### Measurement

```python
return qml.expval(cost_h)
```

- **Expectation Value Measurement**: Measures the expected value of the cost Hamiltonian in the final state.

- **Purpose**: Provides a scalar cost value used in the optimization process.


### Sampling

```python
return qml.sample(wires=range(n_qubits))
```

- **Sample Measurement**: Measures the qubits in the computational basis to obtain bitstrings.

- **Purpose**: Provides actual candidate solutions to the Max-Cut problem.


# Summary

- ### Hadamard Gates (`qml.Hadamard`)

- **Used for**: Initializing qubits in superposition.
- **Effect**: Enables the exploration of all possible states simultaneously.

### Evolution under Cost Hamiltonian (`qml.ApproxTimeEvolution(cost_h, gamma, 1)`)

- **Implemented via**: Controlled-phase gates and single-qubit rotations, depending on the Hamiltonian terms.
- **Effect**: Encodes the problem's cost function into the quantum state via phase accumulation.

### Evolution under Mixer Hamiltonian (`qml.ApproxTimeEvolution(mixer_h, beta, 1)`)

- **Implemented via**: Pauli-X rotations.
- **Effect**: Mixes the amplitudes of different states, allowing the algorithm to explore the solution space.

### Measurement (`qml.expval`, `qml.sample`)

- **Expectation Value Measurement**: Provides a cost value for optimization.
- **Sample Measurement**: Yields bitstrings representing possible solutions.













