# Quadratic Unconstrained Binary Optimization (QUBO) and Its Role in Quantum Computing Algorithms

Quadratic Unconstrained Binary Optimization, or **QUBO**, is a mathematical framework that formulates a range of combinatorial optimization problems. QUBO has recently gained significant attention, particularly in quantum computing and hybrid quantum-classical algorithms, because it elegantly represents complex optimization challenges as binary, quadratic problems.

In this discussion, we will explore:

- The definition and structure of QUBO.
- The mathematical formulation of QUBO problems.
- How QUBO maps to quantum computing and optimization techniques.
- Applications of QUBO, especially in fields such as machine learning, logistics, finance, and quantum computing.

## 1. Defining Quadratic Unconstrained Binary Optimization (QUBO)

Quadratic Unconstrained Binary Optimization is an optimization framework that tackles problems where:

- The variables are binary (i.e., take values of either 0 or 1).
- The objective function is quadratic, consisting of linear and pairwise interaction terms.
- There are no explicit constraints on the variables (although constraints can be modeled within the objective function).

A QUBO problem is typically defined to minimize a quadratic function of binary variables:

$$
\min_{x \in \{0,1\}^n} x^T Q x
$$

where:

- \( x \) is a binary vector of length \( n \) with each element \( x_i \in \{0, 1\} \).
- \( Q \) is a symmetric matrix of size \( n \times n \), representing weights for both individual terms and pairwise interactions.

This formulation is "unconstrained" because there are no explicit equality or inequality constraints on \( x \). However, in practice, some constraints may be encoded within \( Q \) by adding penalty terms to the objective function.

## 2. Mathematical Formulation of QUBO

The core goal of QUBO is to find a binary vector \( x \) that minimizes a quadratic objective function. The mathematical formulation involves expressing the problem as follows:

### Objective Function

The QUBO objective function can be expressed as:

$$
f(x) = \sum_{i=1}^{n} Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j
$$

where:

- The diagonal elements \( Q_{ii} \) represent the linear contributions for each \( x_i \).
- The off-diagonal elements \( Q_{ij} \) represent pairwise interactions between binary variables \( x_i \) and \( x_j \).

This formulation is computationally efficient because it allows a compact representation of a quadratic problem using binary variables. The QUBO model translates complex interactions and dependencies in the system into entries in the matrix \( Q \), making it a suitable structure for algorithms that handle binary optimization.

### Encoding Constraints in QUBO

In many real-world problems, there are constraints, such as requiring certain variables to sum to a specific value. To incorporate these, we add penalty terms to the objective function. For example, if we want the sum of binary variables to equal a constant \( C \), we can add a penalty term like:

$$
P \left( \sum_{i=1}^n x_i - C \right)^2
$$

where \( P \) is a large positive constant. This penalty ensures that any deviation from the constraint results in a high value of the objective function, effectively "enforcing" the constraint.

## 3. QUBO in Quantum Computing: Mapping Problems to Quantum Annealers

QUBO problems are inherently suitable for quantum computing because of their binary, quadratic structure, which maps efficiently onto quantum annealers and quantum-inspired optimization algorithms. Quantum annealers, such as D-Wave systems, leverage QUBO formulations to solve complex combinatorial optimization problems using quantum mechanics principles.

### Why QUBO Works Well with Quantum Computing

Quantum computing algorithms, specifically quantum annealing, work by finding the ground state (minimum energy state) of a problem's Hamiltonian—a function that describes the energy landscape. Since QUBO is naturally represented as a binary system with quadratic interactions, it directly translates to a Hamiltonian that can be processed by quantum annealers.

In this mapping:

- The QUBO matrix \( Q \) corresponds to an energy Hamiltonian.
- Each binary variable \( x_i \) is represented by a quantum bit (qubit).
- Pairwise interactions \( Q_{ij} x_i x_j \) are encoded as qubit interactions in a quantum processor.

### Quantum Annealing for QUBO

Quantum annealing solves QUBO problems by gradually lowering the energy landscape of the Hamiltonian, encouraging the system to settle into the lowest-energy configuration. This process is akin to a "quantum optimization" technique, where qubits explore various configurations to find the optimal solution.

## 4. Applications of QUBO in Quantum and Classical Computing

The QUBO framework's versatility makes it applicable across various industries and problems, including:

### a) Machine Learning and Data Clustering

QUBO can reformulate clustering problems by representing data points as binary variables, minimizing a cost function that groups similar data points. This approach has applications in image segmentation, document clustering, and recommendation systems.

### b) Portfolio Optimization in Finance

In finance, portfolio optimization requires selecting a subset of assets that maximizes return and minimizes risk. This problem is naturally suited for QUBO since binary variables can represent asset inclusion or exclusion, and interaction terms capture correlations between assets.

### c) Logistics and Supply Chain Optimization

QUBO is also applicable in logistics, where it can solve complex routing and scheduling problems. For instance, the Traveling Salesperson Problem (TSP), which seeks the shortest route visiting multiple cities, can be formulated as a QUBO problem. Quantum annealers and hybrid quantum-classical methods have shown promising results for such logistics problems.

### d) Quantum-Inspired Optimization Techniques

Even on classical computers, quantum-inspired optimization methods like simulated annealing or digital annealing can solve QUBO problems efficiently. This trend enables companies to benefit from quantum algorithms on classical hardware, addressing combinatorial problems at scale.

## 5. Example: Formulating a QUBO Problem for Portfolio Optimization

Let’s consider a simple example of formulating a portfolio optimization problem as a QUBO:

Suppose we have a set of assets, each represented by a binary variable \( x_i \) where \( x_i = 1 \) indicates that the asset is selected, and \( x_i = 0 \) indicates that it is not. We aim to maximize returns while minimizing risk, encoded as a QUBO problem.

### Objective Function

Define:

- \( r_i \): Expected return of asset \( i \).
- \( c_{ij} \): Covariance of returns between assets \( i \) and \( j \).
- A constant \( \lambda \) to balance return and risk.

Then, the QUBO formulation is:

$$
\text{maximize } \sum_{i} r_i x_i - \lambda \sum_{i < j} c_{ij} x_i x_j
$$

Here:

- The linear term \( \sum_{i} r_i x_i \) represents returns.
- The quadratic term \( \sum_{i < j} c_{ij} x_i x_j \) models the risk, which depends on covariances.

### Example 1: A Basic QUBO Problem (Minimizing a Simple Binary Quadratic Function)

Consider a basic QUBO problem where we want to find a binary vector \( x \in \{0,1\}^n \) that minimizes a simple quadratic function. Suppose we have two binary variables, \( x_1 \) and \( x_2 \), and the QUBO matrix \( Q \) is given by:

$$
Q = \begin{bmatrix} 3 & -2 \\
-2 & 4 \end{bmatrix}
$$

The objective function for QUBO is:

$$
f(x) = x^T Q x = 3x_1^2 + 4x_2^2 - 2x_1 x_2
$$

Since \( x_i \in \{0,1\} \), \( x_i^2 = x_i \), simplifying the objective function to:

$$
f(x) = 3x_1 + 4x_2 - 2x_1 x_2
$$

To minimize \( f(x) \):

- Substitute all combinations of \( x_1 \) and \( x_2 \):
  - For \( x = [0, 0] \): \( f(x) = 0 \)
  - For \( x = [1, 0] \): \( f(x) = 3 \)
  - For \( x = [0, 1] \): \( f(x) = 4 \)
  - For \( x = [1, 1] \): \( f(x) = 5 \)

The minimum occurs at \( x = [0, 0] \) with \( f(x) = 0 \).

This basic example demonstrates how a QUBO problem seeks to minimize a quadratic binary function. Here, the result \( x = [0, 0] \) is the configuration with the lowest energy in a quantum annealing or classical optimization approach.

### Example 2: Adding Constraints to QUBO (Constrained Optimization via Penalty Term)

In many real-world problems, constraints need to be incorporated into QUBO formulations. Let’s consider a situation where we want to enforce a constraint that exactly one of the binary variables should be 1, often termed a "one-hot" constraint.

Given three variables \( x_1, x_2, x_3 \), we want to minimize:

$$
f(x) = -x_1 - 2x_2 - 3x_3
$$

subject to the constraint \( x_1 + x_2 + x_3 = 1 \).

To enforce this constraint, we introduce a penalty term:

$$
P(x) = \lambda (x_1 + x_2 + x_3 - 1)^2
$$

where \( \lambda \) is a large positive constant to penalize deviations from the constraint.

Expanding \( P(x) \):

$$
P(x) = \lambda (x_1^2 + x_2^2 + x_3^2 + 2x_1 x_2 + 2x_1 x_3 + 2x_2 x_3 - 2x_1 - 2x_2 - 2x_3 + 1)
$$

Then, the QUBO objective function becomes:

$$
f(x) + P(x) = -x_1 - 2x_2 - 3x_3 + \lambda (x_1 + x_2 + x_3 - 1)^2
$$

To minimize this function:

- Substitute all feasible combinations satisfying the constraint (e.g., only one variable should be 1).
- With a sufficiently large \( \lambda \), only configurations meeting the constraint will have a low function value.
  
For instance, with \( x = [0, 1, 0] \), \( f(x) + P(x) = -2 \), which might be the minimal achievable value, depending on \( \lambda \).

This approach allows us to encode constraints directly into the QUBO formulation by using penalty terms.

### Example 3: Portfolio Optimization Problem (Application of QUBO in Finance)

Consider a portfolio optimization problem where we have four assets, each with binary decision variables \( x_1, x_2, x_3, x_4 \), representing whether each asset is included (1) or excluded (0) in the portfolio.

Suppose we have:

- Expected returns for each asset: \( r = [5, 6, 7, 8] \).
- Risk correlations between assets, represented as a covariance matrix \( C \):

$$
C = \begin{bmatrix} 1 & 0.2 & 0.3 & 0.1 \\
0.2 & 1 & 0.4 & 0.5 \\
0.3 & 0.4 & 1 & 0.2 \\ 
0.1 & 0.5 & 0.2 & 1 \end{bmatrix}
$$

We aim to maximize returns while minimizing risk, represented as a QUBO problem:

$$
f(x) = - \sum_{i=1}^{4} r_i x_i + \lambda \sum_{i < j} C_{ij} x_i x_j
$$

where \( \lambda \) balances between maximizing returns and minimizing risk.

### Objective Function Expansion

- Substitute \( r_i \) values:

$$
\sum_{i=1}^{4} r_i x_i = -5x_1 - 6x_2 - 7x_3 - 8x_4
$$

- Substitute pairwise terms with covariances:

$$
\lambda \sum_{i < j} C_{ij} x_i x_j = \lambda (0.2 x_1 x_2 + 0.3 x_1 x_3 + 0.1 x_1 x_4 + 0.4 x_2 x_3 + 0.5 x_2 x_4 + 0.2 x_3 x_4)
$$

The final QUBO formulation is:

$$
f(x) = -5x_1 - 6x_2 - 7x_3 - 8x_4 + \lambda (0.2 x_1 x_2 + 0.3 x_1 x_3 + 0.1 x_1 x_4 + 0.4 x_2 x_3 + 0.5 x_2 x_4 + 0.2 x_3 x_4)
$$

### To solve:

A quantum annealer or classical QUBO solver can minimize this function over all possible values of \( x \in \{0,1\}^4 \). The solution will select assets that optimize the trade-off between high returns and low risk, depending on the chosen \( \lambda \).

### Example 4: QUBO Application in Logistics and Supply Chain Optimization (Traveling Salesperson Problem - TSP)

In the Traveling Salesperson Problem (TSP), suppose we have four cities (A, B, C, and D) with the following distance matrix \( D \):

$$
D = \begin{bmatrix} 0 & 10 & 15 & 20 \\
10 & 0 & 35 & 25 \\
15 & 35 & 0 & 30 \\ 
20 & 25 & 30 & 0 \end{bmatrix}
$$

Here, \( D_{ij} \) represents the distance between city \( i \) and city \( j \). For simplicity, let's represent each path between cities as a binary variable \( x_{ij} \), where:

- \( x_{ij} = 1 \) if the path from city \( i \) to city \( j \) is taken.
- \( x_{ij} = 0 \) otherwise.

#### Step 1: Objective Function (Minimize Distance Traveled)

The QUBO objective function aims to minimize the total distance traveled, formulated as:

$$
\text{Minimize } f(x) = \sum_{i=1}^4 \sum_{j=1}^4 D_{ij} x_{ij}
$$

#### Step 2: Adding Constraints

1. **Each City is Visited Once**: For each city, add a constraint ensuring that each city is entered exactly once:

$$
   \sum_{j=1}^4 x_{ij} = 1, \quad \forall i
  $$

3. **Each City is Left Once**: Similarly, each city must be exited exactly once:

$$
   \sum_{i=1}^4 x_{ij} = 1, \quad \forall j
$$

5. **No Subtours**: We’ll address subtour elimination by adding penalties for any deviations from a single loop structure.

To enforce these constraints in the QUBO model, we add penalty terms using a large constant \( P \). For example, for city A, the penalty term would look like:

$$
P \left( \sum_{j=1}^4 x_{1j} - 1 \right)^2 + P \left( \sum_{i=1}^4 x_{i1} - 1 \right)^2
$$

#### Step 3: QUBO Formulation

Combining the objective function with penalty terms, we get:

$$
\text{Minimize } f(x) = \sum_{i=1}^4 \sum_{j=1}^4 D_{ij} x_{ij} + P \sum_{i=1}^4 \left( \sum_{j=1}^4 x_{ij} - 1 \right)^2 + P \sum_{j=1}^4 \left( \sum_{i=1}^4 x_{ij} - 1 \right)^2
$$

#### Calculation Example:

Suppose we assign:

- \( P = 100 \), a sufficiently large penalty constant.

Let’s evaluate the function for an example configuration \( x = [x_{12} = 1, x_{23} = 1, x_{34} = 1, x_{41} = 1] \), representing a possible tour \( A \to B \to C \to D \to A \).

- **Distance Contribution**:

$$
\sum_{i,j} D_{ij} x_{ij} = D_{12} + D_{23} + D_{34} + D_{41} = 10 + 35 + 30 + 20 = 95
$$

- **Penalty Contribution** (for each city): Each city constraint \( \sum x_{ij} = 1 \) is satisfied, so the penalty term contributes \( 0 \).

Thus, the total cost function for this configuration is:

$$
f(x) = 95 + 0 = 95
$$

This is the minimized value if all constraints are met, providing the optimal route with the shortest total distance.

### Example 5: QUBO Application in Machine Learning and Data Clustering

In this example, we’ll apply QUBO to a clustering problem, where we aim to group data points into clusters. Let’s consider four data points (labeled 1, 2, 3, and 4) with the following distance matrix \( D \), where \( D_{ij} \) represents the dissimilarity between points \( i \) and \( j \):

$$
D = \begin{bmatrix} 0 & 2 & 6 & 10 \\
2 & 0 & 7 & 12 \\
6 & 7 & 0 & 3 \\
10 & 12 & 3 & 0 \end{bmatrix}
$$

We want to partition these points into two clusters, \( C_1 \) and \( C_2 \).

#### Step 1: Define Binary Variables

Define binary variables \( x_{ic} \):

- \( x_{i1} = 1 \) if data point \( i \) is assigned to cluster \( C_1 \), otherwise \( x_{i1} = 0 \).
- \( x_{i2} = 1 \) if data point \( i \) is assigned to cluster \( C_2 \), otherwise \( x_{i2} = 0 \).

#### Step 2: Objective Function (Minimize Intra-cluster Dissimilarity)

The objective is to minimize the total intra-cluster dissimilarity:

$$
\text{Minimize } f(x) = \sum_{c=1}^2 \sum_{i=1}^4 \sum_{j=1}^4 D_{ij} x_{ic} x_{jc}
$$

#### Step 3: Constraint (Single Cluster Assignment)

Each point must belong to exactly one cluster, which we enforce with penalty terms:

$$
P \sum_{i=1}^4 \left( \sum_{c=1}^2 x_{ic} - 1 \right)^2
$$

### QUBO Formulation

Combining the objective and penalty terms, the QUBO objective becomes:

$$
\text{Minimize } f(x) = \sum_{c=1}^2 \sum_{i=1}^4 \sum_{j=1}^4 D_{ij} x_{ic} x_{jc} + P \sum_{i=1}^4 \left( \sum_{c=1}^2 x_{ic} - 1 \right)^2
$$

#### Calculation Example

Assume \( P = 100 \). Let’s evaluate the cost for an example clustering configuration:

- Points 1 and 2 assigned to \( C_1 \): \( x_{11} = 1 \), \( x_{21} = 1 \).
- Points 3 and 4 assigned to \( C_2 \): \( x_{33} = 1 \), \( x_{44} = 1 \).

- **Intra-cluster Dissimilarity Contribution**:
  - For \(C_1\) (points 1 and 2): \( D_{12} x_{11} x_{21} = 2 \times 1 \times 1 = 2 \)
  - For \(C_2\) (points 3 and 4): \( D_{34} x_{33} x_{43} = 3 \times 1 \times 1 = 3 \)

  Total intra-cluster dissimilarity:

$$
\sum_{c=1}^2 \sum_{i,j} D_{ij} x_{ic} x_{jc} = 2 + 3 = 5
$$

- **Penalty Contribution**:

Each point is correctly assigned to one cluster, so the penalty contribution is \( 0 \).

Thus, the total cost function for this configuration is:

$$
f(x) = 5 + 0 = 5
$$

This minimized cost reflects a configuration where points are grouped to minimize intra-cluster dissimilarity, yielding an optimal clustering solution.



# Formulation of the Traveling Salesperson Problem (TSP) as a QUBO

In the TSP, we are given:

- A set of \( n \) cities, each of which must be visited exactly once.
- A distance matrix \( D \), where \( D_{ij} \) represents the distance between city \( i \) and city \( j \).

Our goal is to find a tour that visits each city once and minimizes the total travel distance.

## Step 1: Define Binary Variables

Define binary variables \( x_{ij} \):

$$
x_{ij} = \begin{cases} 
1 & \text{if the tour visits city } i \text{ at position } j, \\ 
0 & \text{otherwise.} 
\end{cases}
$$

This binary encoding requires \( n \times n \) variables, where \( x_{ij} \) tells us whether city \( i \) is in the \( j \)-th position in the tour.

## Step 2: Objective Function for Minimizing Distance

The objective is to minimize the total distance traveled in the tour. This objective can be written in terms of the distance matrix \( D \) and the binary variables \( x_{ij} \):

$$
\text{Minimize } \sum_{i=1}^n \sum_{j=1}^n D_{ij} x_{i,j} x_{i, (j+1) \, \text{mod} \, n}
$$

### Implementation

In code, this can be implemented as follows:

```python
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            for k in range(num_cities):
                Q[i * num_cities + k, j * num_cities + (k + 1) % num_cities] += distance_matrix[i, j]
```

Each entry in the QUBO matrix \( Q \) is updated with the corresponding distance \( D_{ij} \) for transitions between adjacent cities \( i \) and \( j \) in the tour.

## Step 3: Constraints to Ensure a Valid Tour

We need to add constraints to make sure that:

1. **Each city is visited exactly once.**
2. **Each position in the tour is occupied by exactly one city.**

These constraints are encoded by adding large penalty terms to the QUBO matrix \( Q \), ensuring that any invalid tour configuration (such as missing or repeating cities) incurs a high cost.

### Constraint 1: Each City is Visited Exactly Once

For each city \( i \), we want:

$$
\sum_{j=1}^n x_{ij} = 1
$$

This ensures that each city appears once in the tour. To enforce this, we add a penalty term to \( Q \):

$$
P \sum_{i=1}^n \left( \sum_{j=1}^n x_{ij} - 1 \right)^2
$$

### Code Implementation

This constraint can be implemented as follows:

```python
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
            for k in range(num_cities):
                Q[i * num_cities + k, j * num_cities + k] += 2
                Q[i * num_cities + k, i * num_cities + k] -= 1
                Q[j * num_cities + k, j * num_cities + k] -= 1
```

By adding these penalties, the QUBO matrix \( Q \) is constructed to minimize the distance while enforcing a valid tour configuration.

## Constructing the QUBO Matrix

The `create_qubo` function returns the fully constructed QUBO matrix \( Q \), which encapsulates both the objective function and the constraints.

### Step 4: Implement the Quantum Circuit

We use **QAOA (Quantum Approximate Optimization Algorithm)** to solve the TSP as a QUBO problem. Let’s dive into the mathematical reasoning behind each gate and why it’s used.

#### QAOA Overview

QAOA is a hybrid quantum-classical algorithm that alternates between applying two types of operations:

1. **Phase Separation (Problem Hamiltonian)**: Encodes the problem’s objective and constraints into the quantum state by modifying the phases of certain qubits.
2. **Mixer (Driver Hamiltonian)**: Encourages exploration of different solutions by spreading the probability distribution across possible states.

The QAOA algorithm requires two parameters for each layer:

- \( \gamma \): Controls the influence of the problem Hamiltonian (penalty and objective encoded in the QUBO matrix).
- \( \beta \): Controls the influence of the mixer Hamiltonian (spread over possible states).

These layers are repeated multiple times, with each layer tuned by \( \gamma \) and \( \beta \) parameters to refine the solution.

#### Quantum Circuit for TSP with QAOA

The goal of our quantum circuit is to prepare a quantum state that represents an optimal solution to the TSP by evolving the quantum state according to these QAOA layers.

The quantum circuit is constructed as follows:

```python
@qml.qnode(dev)
def circuit(params):
    for i in range(num_cities**2):
        qml.Hadamard(wires=i)
    for gamma, beta in zip(params[:len(params)//2], params[len(params)//2:]):
        qaoa_layer(gamma, beta)
    return qml.probs(wires=range(num_cities**2))
```
## Step 1: Hadamard Gates for Superposition

```python
for i in range(num_cities**2):
    qml.Hadamard(wires=i)
```

The Hadamard gate is used to create an equal superposition state for each qubit. Mathematically, applying a Hadamard gate to a qubit in the \( |0\rangle \) state produces:

$$
H|0\rangle = \frac{1}{\sqrt{2}} \left( |0\rangle + |1\rangle \right)
$$

Applying Hadamard gates to all \( n \times n \) qubits (representing all possible city-position pairs) results in a superposition of all possible routes through the cities, where each bit in the binary solution represents whether a city is visited at a particular position.

For \( N \) qubits, the overall superposition state is:

$$
|\psi\rangle = \frac{1}{\sqrt{2^N}} \sum_{x=0}^{2^N - 1} |x\rangle
$$

This superposition enables QAOA to explore all possible configurations simultaneously, leveraging quantum parallelism to test all routes through interference.

### Step 2: Define the QAOA Layer (Mixer and Phase Separation)

The QAOA layer consists of two main operations:

1. **Rotation around the X-axis (Mixer Hamiltonian)**
2. **Controlled Rotation around the Z-axis (Phase Separation/Problem Hamiltonian)**

#### Code Implementation

```python
def qaoa_layer(gamma, beta):
    # Mixer: RX rotations on all qubits
    for i in range(num_cities**2):
        qml.RX(2 * beta, wires=i)

    # Phase Separation: RZ rotations based on QUBO matrix
    for i in range(num_cities**2):
        for j in range(i + 1, num_cities**2):
            if Q[i, j] != 0:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * Q[i, j], wires=j)
                qml.CNOT(wires=[i, j])
```

The **RX gate** (rotation around the X-axis) acts as the mixer Hamiltonian, responsible for distributing amplitude across states. Mathematically, an RX gate with angle \( \theta \) on a qubit \( i \) is represented by the unitary:

$$
RX(\theta) = e^{-i \frac{\theta}{2} X} = \cos\left(\frac{\theta}{2}\right) I - i \sin\left(\frac{\theta}{2}\right) X
$$

For \( \theta = 2\beta \), the RX gate applies:

$$
RX(2\beta) |0\rangle = \cos(\beta) |0\rangle - i \sin(\beta) |1\rangle
$$

This rotation acts on each qubit, redistributing probability across possible states, encouraging exploration and preventing the algorithm from getting stuck in local minima.

The **mixer Hamiltonian** for all qubits is effectively:

$$
H_M(\beta) = \sum_{i=1}^{n^2} RX(2\beta)_i
$$

Applying this Hamiltonian spreads the probability mass over different states by creating a "mixing" effect.

### Phase Separation Hamiltonian (Controlled RZ Gates)

The Controlled-RZ gate sequence encodes the QUBO problem’s penalties and objectives (distance and constraints) into the quantum state by adjusting qubit phases.

The **RZ gate**, \( RZ(\theta) \), performs a rotation around the Z-axis:

$$
RZ(\theta) = e^{-i \frac{\theta}{2} Z} = \cos\left(\frac{\theta}{2}\right) I - i \sin\left(\frac{\theta}{2}\right) Z
$$

In the QAOA layer, the RZ rotation is applied conditionally, controlled by CNOT gates, with an angle proportional to \( 2\gamma Q_{ij} \), where \( Q_{ij} \) represents the pairwise penalty and cost between qubits \( i \) and \( j \) in the QUBO matrix.

Mathematically, the **phase separation Hamiltonian** is:

$$
H_P(\gamma) = \sum_{i,j} Q_{ij} Z_i Z_j
$$

where \( Z_i \) and \( Z_j \) are Pauli-Z operators on qubits \( i \) and \( j \), and \( Q_{ij} \) represents the QUBO cost associated with that pair.

### Implementation of Controlled-RZ Gates in the Code

```python
for i in range(num_cities**2):
    for j in range(i+1, num_cities**2):
        if Q[i, j] != 0:
            qml.CNOT(wires=[i, j])
            qml.RZ(2 * gamma * Q[i, j], wires=j)
            qml.CNOT(wires=[i, j])
```
### CNOT Gate

The **CNOT gate** is used to set up a control condition, where an **RZ rotation** on qubit \( j \) is applied only if qubit \( i \) is in the state \( |1\rangle \).

### RZ Gate

The **RZ rotation** $RZ(2\gamma Q_{ij})$ applies a phase shift based on the QUBO matrix value $Q_{ij}$ and the parameter $\gamma$, encoding both the objective function and constraints. This **phase separation operation** "marks" each possible configuration based on its cost (distance or penalty). Configurations that don’t satisfy the TSP constraints or have longer paths receive phase penalties, leading to lower probabilities in the final measurement.

### Final Measurement and Optimization

After repeating the QAOA layers, the state of the qubits represents a superposition where the probability of each basis state corresponds to the likelihood of that route being optimal. By measuring the qubits and using classical optimization to adjust \( \gamma \) and \( \beta \), QAOA iteratively improves the solution.

The final circuit returns a probability distribution over all configurations, from which we select the most probable configuration as the optimal solution to the TSP.

### Summary of the QAOA Circuit Components

1. **Hadamard Gates**: Initialize a superposition of all possible city-position configurations.
2. **Mixer (RX) Gates**: Spread the probability distribution over all configurations, allowing exploration of solution space.
3. **Phase Separation (Controlled-RZ) Gates**: Encode the QUBO matrix, imposing penalties for invalid routes and minimizing distance.

Through iterative optimization, **QAOA** gradually focuses the quantum state on configurations with lower penalties and distances, converging on an optimal solution to the TSP.

### Cost Function and Optimization

The **cost function** calculates the cost for the current set of parameters using the QAOA circuit output. The **GradientDescentOptimizer** minimizes this cost to find the optimal parameters.


```python
def cost(params):
    return circuit(params)[0]

params = np.random.rand(2 * num_cities)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

for i in range(100):
    params = opt.step(cost, params)
```

### Step 5: Interpret the Results

After optimization, we interpret the results by finding the **optimal tour** based on the final probability distribution of the QAOA circuit output.

```python
result = circuit(params)
solution = np.argmax(result)
tour = [solution // num_cities**i % num_cities for i in range(num_cities)]
print("Optimal tour:", tour)
```

- `np.argmax(result)` retrieves the binary solution with the **highest probability**, representing the most likely **optimal tour configuration**.
- The `tour` list decodes the solution into a readable **tour sequence** by iterating through the binary solution.



Full code
```python
import pennylane as qml
from pennylane import numpy as np


# Define the distance matrix for the cities
distance_matrix = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

num_cities = len(distance_matrix)

# Define the QUBO matrix for the TSP
def create_qubo(distance_matrix):
    num_cities = len(distance_matrix)
    Q = np.zeros((num_cities**2, num_cities**2))

    # Constraint: Each city must be visited exactly once
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                for k in range(num_cities):
                    Q[i*num_cities + k, j*num_cities + k] += 2
                    Q[i*num_cities + k, i*num_cities + k] -= 1
                    Q[j*num_cities + k, j*num_cities + k] -= 1

    # Constraint: Each position in the tour must be occupied by exactly one city
    for k in range(num_cities):
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    Q[k*num_cities + i, k*num_cities + j] += 2
                    Q[k*num_cities + i, k*num_cities + i] -= 1
                    Q[k*num_cities + j, k*num_cities + j] -= 1

    # Objective: Minimize the total distance
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                for k in range(num_cities):
                    Q[i*num_cities + k, j*num_cities + (k+1) % num_cities] += distance_matrix[i, j]

    return Q

Q = create_qubo(distance_matrix)

# Define the quantum device
dev = qml.device('default.qubit', wires=num_cities**2)

# Define the QAOA circuit
def qaoa_layer(gamma, beta):
    for i in range(num_cities**2):
        qml.RX(2 * beta, wires=i)
    for i in range(num_cities**2):
        for j in range(i+1, num_cities**2):  # Ensure i != j
            if Q[i, j] != 0:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * Q[i, j], wires=j)
                qml.CNOT(wires=[i, j])

@qml.qnode(dev)
def circuit(params):
    for i in range(num_cities**2):
        qml.Hadamard(wires=i)
    for gamma, beta in zip(params[:len(params)//2], params[len(params)//2:]):
        qaoa_layer(gamma, beta)
    return qml.probs(wires=range(num_cities**2))

# Optimize the QAOA parameters
def cost(params):
    return circuit(params)[0]

params = np.random.rand(2 * num_cities)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

for i in range(100):
    params = opt.step(cost, params)

# Interpret the results
result = circuit(params)
solution = np.argmax(result)
tour = [solution // num_cities**i % num_cities for i in range(num_cities)]
print("Optimal tour:", tour)
```
The answer is: Optimal tour: [3, 3, 0, 0]
Beware it takes a few minutes to calculate.
