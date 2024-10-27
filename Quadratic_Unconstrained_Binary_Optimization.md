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


```python
import pennylane as qml
from pennylane import numpy as np

# Number of cities
N = 4

# Distance matrix D
D = np.array([
    [0, 2, 9, 10],
    [2, 0, 6, 4],
    [9, 6, 0, 8],
    [10, 4, 8, 0]
])

# Penalty coefficients
A = 10
B = 10

def qubit_index(i, j):
    return i * N + j

num_vars = N * N
Q = np.zeros((num_vars, num_vars))

# Objective Function
for i in range(N):
    for j in range(N):
        for k in range(N):
            if j != k:
                idx1 = qubit_index(i, j)
                idx2 = qubit_index((i + 1) % N, k)
                Q[idx1, idx2] += D[j, k]

# Constraints
# Each city is visited exactly once
for j in range(N):
    idx = [qubit_index(i, j) for i in range(N)]
    for a in idx:
        Q[a, a] += -A * (2 * (N - 1))
        for b in idx:
            Q[a, b] += 2 * A

# One city per position
for i in range(N):
    idx = [qubit_index(i, j) for j in range(N)]
    for a in idx:
        Q[a, a] += -B * (2 * (N - 1))
        for b in idx:
            Q[a, b] += 2 * B

# Convert QUBO to Ising Hamiltonian
h = np.zeros(num_vars)
J = {}
constant = 0

for i in range(num_vars):
    h[i] += Q[i, i] * (-0.5)
    constant += Q[i, i] * 0.5

for i in range(num_vars):
    for j in range(i+1, num_vars):
        if Q[i, j] != 0:
            h[i] += Q[i, j] * (-0.25)
            h[j] += Q[i, j] * (-0.25)
            J[(i, j)] = Q[i, j] * 0.25
            constant += Q[i, j] * 0.25

# Construct the Hamiltonian
coeffs = []
obs = []

for i in range(num_vars):
    if h[i] != 0:
        coeffs.append(h[i])
        obs.append(qml.PauliZ(i))

for (i, j), value in J.items():
    coeffs.append(value)
    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

H = qml.Hamiltonian(coeffs, obs)

# Quantum Circuit with QAOA
num_qubits = num_vars
dev = qml.device('default.qubit', wires=num_qubits)

p = 2  # Number of QAOA layers

@qml.qnode(dev)
def circuit(params):
    # Initialize qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    
    gamma = params[0]
    beta = params[1]
    
    # Apply QAOA layers
    for l in range(p):
        qml.qaoa.cost_layer(gamma[l], H)
        qml.qaoa.mixer_layer(beta[l], wires=range(num_qubits))
    
    return qml.expval(H)

def cost_function(params):
    return circuit(params)

# Optimization
np.random.seed(42)
params = [np.random.uniform(0, np.pi, p), np.random.uniform(0, np.pi, p)]
opt = qml.AdamOptimizer(stepsize=0.1)
steps = 100

for _ in range(steps):
    params = opt.step(cost_function, params)

# Retrieve solution
@qml.qnode(dev)
def final_circuit(params):
    # Initialize qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    
    gamma = params[0]
    beta = params[1]
    
    # Apply QAOA layers
    for l in range(p):
        qml.qaoa.cost_layer(gamma[l], H)
        qml.qaoa.mixer_layer(beta[l], wires=range(num_qubits))
    
    # Sample all qubits
    return qml.sample(wires=range(num_qubits))

samples = final_circuit(params)
binary_solution = samples.flatten()
solution_matrix = binary_solution.reshape(N, N)

# Extract tour
tour = []
for i in range(N):
    city = np.argmax(solution_matrix[i])
    tour.append(city)

print("Optimal tour:", tour)
```
