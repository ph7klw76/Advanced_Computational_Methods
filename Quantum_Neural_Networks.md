# Quantum Neural Networks (QNNs)

## Introduction

Quantum Neural Networks (QNNs) bring together the powerful principles of quantum computing and classical neural networks to create a new paradigm in machine learning. They harness quantum superposition, entanglement, and interference to enhance computational capabilities, offering potential speedups in data processing and optimization tasks. This blog explores the fundamental concepts of QNNs, their mathematical foundations, and how they differ from classical neural networks.

## 1. Background on Neural Networks

Before diving into QNNs, it is helpful to briefly review classical neural networks (NNs):

- A neural network consists of interconnected nodes (neurons) organized into layers (input, hidden, and output layers).
- Each connection has a weight $w$, and each neuron applies an activation function $f$ to the weighted sum of its inputs:

$$
  y = f\left(\sum_i w_i x_i + b\right),
$$

  where $b$ is a bias term, and $x_i$ are inputs.

## 2. Introduction to Quantum Neural Networks

QNNs extend classical neural networks by leveraging quantum circuits to perform computations. Instead of traditional neurons, QNNs use qubits and quantum gates to process data.

### 2.1 Why Quantum?

Quantum systems operate on qubits, which can exist in superpositions of states $\lvert 0 \rangle$ and $\lvert 1 \rangle$:

$$
\lvert \psi \rangle = \alpha \lvert 0 \rangle + \beta \lvert 1 \rangle,
$$

where $\alpha, \beta \in \mathbb{C}$ and $\lvert \alpha \rvert^2 + \lvert \beta \rvert^2 = 1$. This allows quantum systems to represent and process exponentially more information than classical bits, providing potential speedups for certain tasks.

## 3. Architecture of Quantum Neural Networks

### 3.1 Qubits and Quantum Gates

- **Qubits**: QNNs operate on qubits, which can exist in multiple states due to superposition.
- **Quantum Gates**: Similar to weights in classical networks, quantum gates manipulate qubits. Common gates include:
  - **Pauli Gates ($X, Y, Z$)**
  - **Hadamard Gate ($H$)**: Creates superposition states.

$$
    H\lvert 0 \rangle = \frac{1}{\sqrt{2}} (\lvert 0 \rangle + \lvert 1 \rangle), \quad H\lvert 1 \rangle = \frac{1}{\sqrt{2}} (\lvert 0 \rangle - \lvert 1 \rangle).
$$

  - **Rotation Gates ($R_x, R_y, R_z$)**: Apply rotations around different axes of the Bloch sphere:

$$
    R_y(\theta) = \exp\left(-i \frac{\theta}{2} Y\right) = 
    \begin{bmatrix}
    \cos(\theta/2) & -\sin(\theta/2) \\
    \sin(\theta/2) & \cos(\theta/2)
    \end{bmatrix}.
$$

### 3.2 Quantum Circuits as Neural Layers

In QNNs, a quantum circuit represents a neural layer, consisting of:

- **Data Encoding**: Classical data is encoded into a quantum state using techniques such as angle embedding:

$$
  \lvert x \rangle = \sum_{i=1}^n x_i \lvert i \rangle,
$$

  where $x_i$ are features of the data.
- **Parameterized Quantum Gates**: Quantum gates with trainable parameters (similar to weights in classical NNs) are applied:

$$
  U(\theta) = \prod_i \exp(-i \theta_i G_i),
$$

  where $\theta_i$ are parameters and $G_i$ are generators (e.g., Pauli matrices).
- **Measurement**: The quantum state is measured, producing classical outputs used for further processing or optimization.

## 4. Mathematical Formulation of QNNs

### 4.1 Data Encoding

To feed classical data $x \in \mathbb{R}^n$ into a QNN, we encode it into a quantum state $\lvert \psi(x) \rangle$. One common method is amplitude encoding, which normalizes the data:

$$
\lvert \psi(x) \rangle = \frac{1}{\lVert x \rVert} \sum_{i=1}^n x_i \lvert i \rangle.
$$

### 4.2 Parameterized Quantum Circuits (PQC)

A parameterized quantum circuit $U(\theta)$ is applied to the input state:

$$
\lvert \psi_{\text{out}} \rangle = U(\theta) \lvert \psi(x) \rangle.
$$

This circuit consists of layers of parameterized gates, such as:

- **Rotation Gates**: Apply rotations based on trainable parameters $\theta$.
- **Entangling Gates**: Introduce correlations between qubits.

**Example: A Simple Quantum Layer**  
Consider a single-qubit QNN layer with a parameterized rotation gate:

$$
U(\theta) = R_y(\theta) = 
\begin{bmatrix}
\cos(\theta/2) & -\sin(\theta/2) \\
\sin(\theta/2) & \cos(\theta/2)
\end{bmatrix}.
$$

Given an input state $\lvert \psi_{\text{in}} \rangle = \lvert 0 \rangle$, the output state becomes:

$$
\lvert \psi_{\text{out}} \rangle = R_y(\theta) \lvert 0 \rangle = \cos(\theta/2) \lvert 0 \rangle + \sin(\theta/2) \lvert 1 \rangle.
$$

### 4.3 Measurement

To obtain classical outputs, the quantum state is measured using observables (e.g., Pauli-Z operator $\sigma_z$). The expectation value is computed as:

$$
\langle O \rangle = \langle \psi_{\text{out}} \lvert O \rvert \psi_{\text{out}} \rangle.
$$

This value serves as the output of the QNN layer.

## 5. Training Quantum Neural Networks

QNNs are trained using classical optimization methods. The parameters $\theta$ of the quantum gates are updated to minimize a cost function $C(\theta)$, similar to classical backpropagation.

### 5.1 Cost Function

A typical cost function is:

$$
C(\theta) = \sum_i (y_i - \langle O_i \rangle)^2,
$$

where $y_i$ are the target values, and $\langle O_i \rangle$ are the measured outputs.

### 5.2 Gradient-Based Optimization

To update the parameters $\theta$, we compute gradients using methods such as the parameter-shift rule:

$$
\frac{\partial C(\theta)}{\partial \theta_i} = \frac{C(\theta_i + \pi/2) - C(\theta_i - \pi/2)}{2}.
$$

## 6. Example QNN Architecture

**Problem: Binary Classification**  
Given a dataset with two features, we use a QNN to classify the data.

- **Data Encoding**: Encode the features into a 2-qubit quantum state using angle embedding.
- **Quantum Circuit**:
  - Apply parameterized rotation gates $R_y(\theta)$ to each qubit.
  - Introduce entanglement using a CNOT gate.
- **Measurement**: Measure the expectation value of the Pauli-Z operator on each qubit.
- **Cost Function**: Minimize the mean squared error between predicted and target values.

## 7. Advantages and Challenges

### Advantages

- **Quantum Parallelism**: QNNs can represent and process complex functions due to superposition and entanglement.
- **Potential Speedups**: For specific tasks, QNNs may offer exponential speedups over classical methods.

### Challenges

- **Noise and Decoherence**: Quantum systems are sensitive to noise, affecting QNN performance.
- **Hardware Limitations**: Current quantum hardware is limited in qubit count and coherence time.

## Conclusion

Quantum Neural Networks combine the strengths of quantum computing and classical neural networks to tackle complex computational problems. By encoding data into quantum states, applying parameterized quantum circuits, and leveraging quantum measurement, QNNs offer a novel approach to machine learning with promising potential for the future.

```python
import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Example dataset: Quantum chemistry data (e.g., molecular energies)
# For simplicity, we use a small synthetic dataset
data = {
    'Molecule': ['H2', 'LiH', 'BeH2', 'CH4', 'NH3'],
    'Feature1': [0.5, 1.2, 1.8, 2.5, 3.0],  # Example feature 1 (e.g., bond length)
    'Feature2': [0.3, 0.8, 1.1, 1.5, 2.0],  # Example feature 2 (e.g., bond angle)
    'Energy': [-1.1, -2.3, -3.5, -4.2, -5.0]  # Example target (e.g., ground state energy)
}

# Convert to numpy array
features = np.array([data['Feature1'], data['Feature2']]).T
energies = np.array(data['Energy'])

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_standardized, energies, test_size=0.2, random_state=42)

# Define the quantum device
n_qubits = features_standardized.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum node for the Quantum Neural Network
@qml.qnode(dev)
def quantum_neural_network(data, weights):
    qml.templates.AngleEmbedding(data, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Initialize random weights for the StronglyEntanglingLayers
n_layers = 2
weights = np.random.random((n_layers, n_qubits, 3))

# Define the cost function
def cost(weights, X, y):
    predictions = np.array([quantum_neural_network(x, weights) for x in X])
    return np.mean((predictions - y) ** 2)

# Optimize the weights using gradient descent
opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
for i in range(steps):
    weights, cost_val = opt.step_and_cost(lambda w: cost(w, X_train, y_train), weights)
    if i % 10 == 0:
        print(f'Step {i}, Cost: {cost_val}')

# Make predictions on the test set
predictions = np.array([quantum_neural_network(x, weights) for x in X_test])

# Calculate mean squared error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, c='blue', s=100)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Quantum Neural Network Regression for Quantum Chemistry')
plt.xlabel('True Energy')
plt.ylabel('Predicted Energy')
plt.grid(True)
plt.show()
```
