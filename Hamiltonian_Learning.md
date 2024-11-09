# Hamiltonian Learning

## Introduction

Hamiltonian learning is a powerful technique in quantum computing for inferring the parameters of a quantum system's Hamiltonian from measurement data. The Hamiltonian $H$ governs the dynamics of quantum systems and encodes the energy and evolution of the system. Accurate Hamiltonian estimation is crucial for quantum simulation, error correction, and the development of quantum technologies. In this blog, we delve into the mathematical framework of Hamiltonian learning, exploring various approaches, algorithms, and applications.

## 1. Background on Quantum Hamiltonians

### 1.1 Definition of a Hamiltonian

In quantum mechanics, the Hamiltonian $H$ of a system is a Hermitian operator that represents the total energy of the system. It determines the time evolution of the quantum state $\lvert \psi(t) \rangle$ according to the Schrödinger equation:

$$
i\hbar \frac{\partial}{\partial t} \lvert \psi(t) \rangle = H \lvert \psi(t) \rangle,
$$

where $\hbar$ is the reduced Planck constant (set to 1 in natural units). The solution to this equation gives the time evolution of the state:

$$
\lvert \psi(t) \rangle = e^{-iHt} \lvert \psi(0) \rangle.
$$

The Hamiltonian can be decomposed as a linear combination of Pauli operators in the case of qubit systems:

$$
H = \sum_i \theta_i P_i,
$$

where $P_i$ are Pauli operators (e.g., $I, X, Y, Z$) and $\theta_i$ are the unknown parameters to be estimated.

### 1.2 Why Hamiltonian Learning?

Hamiltonian learning seeks to estimate the parameters $\theta_i$ given experimental data about the system's evolution. Applications include:

- **Characterization of Quantum Devices**: Determining the Hamiltonian of a quantum device for benchmarking and calibration.
- **Quantum Control**: Optimizing control protocols for quantum systems.
- **Quantum Error Correction**: Understanding noise processes in quantum systems.

## 2. Problem Formulation of Hamiltonian Learning

### 2.1 Mathematical Objective

Given a set of measurement data $\{(t_j, M_j)\}$, where $t_j$ denotes the time at which the measurement $M_j$ (expectation value of an observable) is taken, the goal of Hamiltonian learning is to find the parameters $\theta = (\theta_1, \theta_2, \ldots, \theta_n)$ such that:

$$
M_j \approx \langle \psi(t_j) \rvert O \lvert \psi(t_j) \rangle,
$$

where $O$ is an observable, and $\lvert \psi(t_j) \rangle = e^{-iHt_j} \lvert \psi(0) \rangle$ is the evolved state under the Hamiltonian $H$.

### 2.2 Likelihood Function

One approach to Hamiltonian learning is to maximize the likelihood of observing the measurement outcomes given the Hamiltonian parameters:

$$
L(\theta) = \prod_j P(M_j \mid \theta),
$$

where $P(M_j \mid \theta)$ is the probability of observing measurement $M_j$ given the parameters $\theta$.

## 3. Approaches to Hamiltonian Learning

### 3.1 Direct Optimization

The simplest approach is to directly optimize the parameters $\theta$ to minimize a cost function that quantifies the difference between the observed and predicted measurements. A common cost function is the mean squared error:

$$
C(\theta) = \sum_j \left(M_j - \langle \psi(t_j) \rvert O \lvert \psi(t_j) \rangle \right)^2.
$$

#### Gradient-Based Optimization

The gradients of the cost function can be computed using techniques such as the parameter-shift rule in variational quantum algorithms:

$$
\frac{\partial C(\theta)}{\partial \theta_i} = \frac{C(\theta + \frac{\pi}{2}e_i) - C(\theta - \frac{\pi}{2}e_i)}{2},
$$

where $e_i$ is the unit vector in the $i$-th direction.

### 3.2 Bayesian Hamiltonian Learning

Bayesian methods provide a probabilistic approach to Hamiltonian learning by updating a prior distribution over the Hamiltonian parameters based on observed data. Given a prior $P(\theta)$ and measurement data $\{(t_j, M_j)\}$, the posterior distribution is given by Bayes' rule:

$$
P(\theta \mid \{(t_j, M_j)\}) \propto P(\{(t_j, M_j)\} \mid \theta) P(\theta),
$$

where $P(\{(t_j, M_j)\} \mid \theta)$ is the likelihood function.

#### Sequential Bayesian Updates

Bayesian learning can be performed iteratively by updating the posterior distribution as new data is acquired. This approach is particularly useful for adaptive Hamiltonian learning, where measurements are chosen to maximize information gain.

### 3.3 Machine Learning-Based Methods

Recent approaches use machine learning techniques, such as neural networks, to approximate the relationship between the parameters $\theta$ and the observed data. A neural network can be trained to predict the parameters based on input measurement data, providing a flexible and scalable method for Hamiltonian learning.

## 4. Example: Learning a Two-Qubit Hamiltonian

Consider a two-qubit Hamiltonian of the form:

$$
H = \theta_1 Z_1 + \theta_2 X_2 + \theta_3 Z_1 Z_2,
$$

where $Z_1, X_2, Z_1Z_2$ are Pauli operators acting on the first and second qubits, and $\theta = (\theta_1, \theta_2, \theta_3)$ are the parameters to be learned.

### 4.1 Simulation of Time Evolution

Given an initial state $\lvert \psi(0) \rangle$, the evolved state at time $t$ is:

$$
\lvert \psi(t) \rangle = e^{-iHt} \lvert \psi(0) \rangle.
$$

To compute the expectation value of an observable $O$, we use:

$$
\langle O \rangle_t = \langle \psi(t) \rvert O \lvert \psi(t) \rangle.
$$

### 4.2 Measurement Data

Assume we have measurement data $\{(t_j, M_j)\}$ for different times $t_j$. Our goal is to find the parameters $\theta$ that best fit the observed data.

### 4.3 Optimization Procedure

- **Define Cost Function**:
  
$$
  C(\theta) = \sum_j \left(M_j - \langle \psi(t_j) \rvert O \lvert \psi(t_j) \rangle \right)^2.
$$

- **Optimize Parameters**: Use gradient-based optimization (e.g., gradient descent) to minimize $C(\theta)$.

## 5. Challenges and Practical Considerations

### 5.1 Noise and Imperfect Measurements

Realistic quantum systems are affected by noise, which can complicate Hamiltonian learning. Noise modeling and robust estimation techniques are crucial for accurate parameter estimation.

### 5.2 Scalability

The dimensionality of the Hamiltonian parameter space grows exponentially with the number of qubits, posing challenges for large-scale Hamiltonian learning.

### 5.3 Efficient Data Acquisition

Choosing optimal measurement times and observables can significantly improve the efficiency of Hamiltonian learning. Adaptive strategies, such as Bayesian experimental design, can be used to maximize information gain.

# Application Example: Characterizing a Qubit with a Drift Hamiltonian

## Problem Statement

Consider a single qubit with a Hamiltonian that exhibits a time-dependent drift due to noise. The true Hamiltonian is unknown and takes the form:

$$
H = \theta_1 Z + \theta_2 X,
$$

where:

- $Z$ and $X$ are Pauli matrices:
  
  $$
  Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
  $$

- $\theta_1$ and $\theta_2$ are unknown parameters representing the strength of the $Z$- and $X$-components of the Hamiltonian.

The goal of Hamiltonian learning is to estimate the parameters $\theta_1$ and $\theta_2$ using measurement data obtained from the qubit's evolution.

## Step-by-Step Solution

### Step 1: Time Evolution of the Qubit

The state of the qubit at time $t$ is given by the solution to the Schrödinger equation:

$$
\lvert \psi(t) \rangle = e^{-iHt} \lvert \psi(0) \rangle,
$$

where $\lvert \psi(0) \rangle$ is the initial state. For simplicity, let's assume the qubit starts in the state $\lvert 0 \rangle$ (the eigenstate of $Z$ with eigenvalue $+1$).

### Step 2: Expressing the Time Evolution Operator

The Hamiltonian can be written as:

$$
H = \theta_1 Z + \theta_2 X.
$$

The time evolution operator is given by:

$$
U(t) = e^{-iHt} = e^{-i(\theta_1 Z + \theta_2 X)t}.
$$

To evaluate $U(t)$, we use the fact that $Z$ and $X$ do not commute. Therefore, we use the Baker-Campbell-Hausdorff formula for a small time step or employ numerical methods to approximate the evolution.

### Step 3: Generating Measurement Data

Assume we can measure the expectation value of an observable $O = Z$ at different times $t_j$. The measurement at time $t_j$ is given by:

$$
M_j = \langle \psi(t_j) \rvert Z \lvert \psi(t_j) \rangle.
$$

For example, let’s compute the measurement for a small time $t_j$ using a first-order approximation:

$$
\lvert \psi(t_j) \rangle \approx \left(I - iHt_j\right) \lvert 0 \rangle = \left(I - i(\theta_1 Z + \theta_2 X)t_j\right) \lvert 0 \rangle.
$$

Applying this approximation:

- $Z \lvert 0 \rangle = \lvert 0 \rangle$
- $X \lvert 0 \rangle = \lvert 1 \rangle$

Thus:

$$
\lvert \psi(t_j) \rangle \approx \lvert 0 \rangle - i\theta_1 t_j \lvert 0 \rangle - i\theta_2 t_j \lvert 1 \rangle.
$$

The expectation value of $Z$ is:

$$
M_j = \langle \psi(t_j) \rvert Z \lvert \psi(t_j) \rangle \approx \langle 0 \rvert Z \lvert 0 \rangle + 2i\theta_1 t_j \langle 0 \rvert Z \lvert 0 \rangle = 1 - 2\theta_2^2 t_j^2 \text{ (to second-order)}.
$$

### Step 4: Constructing the Cost Function

Given a set of measurement data $\{(t_j, M_j)\}$, we define the cost function:

$$
C(\theta_1, \theta_2) = \sum_j \left(M_j - \langle \psi(t_j) \rvert Z \lvert \psi(t_j) \rangle_{\theta_1, \theta_2}\right)^2.
$$

Our goal is to find the parameters $\theta_1$ and $\theta_2$ that minimize this cost function.

### Step 5: Optimization

Using gradient-based optimization methods, we update the parameters:

$$
\theta_i \leftarrow \theta_i - \eta \frac{\partial C}{\partial \theta_i},
$$

where $\eta$ is the learning rate, and the gradients can be computed using finite differences or variational techniques.

## Practical Example: Estimating Parameters from Noisy Data

Assume we collect measurements $M_j$ at times $t_j = \{0.1, 0.2, 0.3, \ldots\}$ and observe:

$$
M = \{0.95, 0.87, 0.80, \ldots\}.
$$

Using Hamiltonian learning, we fit the parameters $\theta_1$ and $\theta_2$ such that the predicted measurements from the Hamiltonian's evolution match the observed data as closely as possible.

## Applications in Quantum Technology

### Quantum Device Characterization

Hamiltonian learning can be used to determine the precise Hamiltonian governing a quantum device, enabling accurate calibration and benchmarking.

### Quantum Error Correction

Understanding noise processes in a quantum system often requires learning the system's Hamiltonian, allowing for more effective error correction strategies.

### Quantum Control

By learning the Hamiltonian of a quantum system, we can design optimal control protocols to steer the system's evolution towards desired states.

```python
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the Hamiltonian for a simple quantum system (e.g., H2 molecule)
coeffs = [0.5, -0.3, 0.2]
obs = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
H = qml.Hamiltonian(coeffs, obs)

# Define the quantum device
dev = qml.device("default.qubit", wires=2)

# Define the quantum node to measure the expectation value of the Hamiltonian
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return [qml.expval(o) for o in obs]

# Generate synthetic data for training
true_params = [0.1, 0.2]
true_expectations = circuit(true_params)

# Define the cost function for Hamiltonian learning
def cost(params, target_expectations):
    predictions = circuit(params)
    return np.sum((np.array(predictions) - np.array(target_expectations)) ** 2)

# Initialize random parameters
params = np.random.random(2)

# Optimize the parameters using gradient descent
opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
costs = []

for i in range(steps):
    params, cost_val = opt.step_and_cost(lambda p: cost(p, true_expectations), params)
    costs.append(cost_val)
    if i % 10 == 0:
        print(f'Step {i}, Cost: {cost_val}')

# Plot the cost function over iterations
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.title('Cost Function during Hamiltonian Learning')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Print the learned parameters
print(f'Learned parameters: {params}')
print(f'True parameters: {true_params}')
```

## Explanation of the Components

### Defining the Hamiltonian for a Simple Quantum System:

```python
coeffs = [0.5, -0.3, 0.2]
obs = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
H = qml.Hamiltonian(coeffs, obs)
```

Hamiltonian: A Hamiltonian $H$ is a Hermitian operator that defines the energy and evolution of a quantum system. Here, we are modeling a simple system with three terms:

$H = 0.5Z_0 - 0.3Z_1 + 0.2X_0X_1,$

where:

$Z_0$ and $Z_1$ are Pauli-Z operators acting on qubits 0 and 1.
$X_0X_1$ is a product of Pauli-X operators acting on qubits 0 and 1 (representing an interaction between the two qubits).
Coefficients: The coefficients $[0.5, -0.3, 0.2]$ represent the weights of each term in the Hamiltonian, capturing the relative strength of each interaction.

### Mathematical Interpretation:

The Hamiltonian can be interpreted as a weighted sum of Pauli operators:

$H = \sum_{i} \theta_i O_i,$

where $\theta_i$ are coefficients (weights) and $O_i$ are Pauli operators.

The first term $0.5Z_0$ represents a local field acting on qubit 0, while $-0.3Z_1$ acts on qubit 1. The term $0.2X_0X_1$ represents a coupling or interaction between the two qubits.

Defining the Quantum Device:

```python
dev = qml.device("default.qubit", wires=2)
```

A quantum device is specified with two qubits, representing the two degrees of freedom needed for the Hamiltonian terms.

### Defining the Quantum Circuit:

```python
@qml.qnode(dev)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return [qml.expval(o) for o in obs]
```

Purpose: The circuit represents a quantum state that is parameterized by angles $params[0]$ and $params[1]$. It applies rotations to each qubit and then measures the expectation values of the observables defined in the Hamiltonian.

### Quantum Gates:
$qml.RX(params[0], wires=0)$: Applies a rotation around the $x$-axis to qubit 0. Mathematically, this is represented by the unitary operator:
$R_X(\theta) = \exp(-i \frac{\theta}{2} X)$.

$qml.RY(params[1], wires=1)$: Applies a rotation around the $y$-axis to qubit 1:
$R_Y(\theta) = \exp(-i \frac{\theta}{2} Y)$.

These rotation gates transform the state of each qubit, creating a parameterized quantum state $\lvert \psi(params) \rangle$.

### Measurement:
return [qml.expval(o) for o in obs]: Measures the expectation values of the observables in the Hamiltonian (i.e., $Z_0$, $Z_1$, and $X_0X_1$). Mathematically, this corresponds to:
$\langle O_i \rangle = \langle \psi(params) \lvert O_i \rvert \psi(params) \rangle$,
where $O_i$ is each observable in the Hamiltonian.

### Why This Circuit Structure?

Parameterization: By parameterizing the rotations, we create a flexible quantum state that can represent various configurations of the system. The parameters $params$ can be adjusted during learning to match the observed behavior of the Hamiltonian.

Expectation Value Measurement: Measuring the expectation values of the Hamiltonian terms allows us to compare the predicted values from the model with observed data, forming the basis of the cost function for optimization.

Synthetic Data: The "true" parameters represent a known configuration of the system. By evaluating the circuit with these parameters, we obtain synthetic measurements (expectation values of the observables).
This serves as a target for the optimization process, simulating the scenario where we have experimental data and seek to learn the parameters of the underlying Hamiltonian.
