### Quantum Support Vector Machines (QSVM)

Quantum Support Vector Machines (QSVM) extend the classical Support Vector Machine (SVM) framework into the realm of quantum computing. By leveraging the computational power of quantum computers, QSVMs aim to accelerate and enhance SVM algorithms, which are fundamental in supervised machine learning for tasks such as classification and regression. In this technical blog, we will delve into the mathematical underpinnings of QSVMs, exploring their quantum enhancements, core algorithms, and potential for achieving quantum advantage. Our discussion will include rigorous mathematical derivations and explanations to elucidate the theoretical foundation of QSVMs.

#### 1. Introduction to Support Vector Machines (SVMs)

Classical SVMs are a supervised learning model that seeks to find the optimal hyperplane to separate data points of different classes. Given a dataset of labeled points  

$$
\{(x_i, y_i)\}_{i=1}^{N},
$$

where

$$
x_i \in \mathbb{R}^n
$$

is the input vector and 

$$
y_i \in \{-1, +1\}
$$

is the class label, SVMs aim to solve the following optimization problem:

$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{subject to} \quad y_i (w \cdot x_i + b) \geq 1 \quad \forall i,
$$  

where $w$ is the weight vector that defines the hyperplane and $b$ is the bias term.

#### 2. Quantum Embedding and Kernel Methods

QSVMs leverage quantum computing in two key ways:

- **Quantum Feature Maps:** Classical data is embedded into a higher-dimensional Hilbert space using a quantum feature map $\phi(x).$

This embedding transforms input data points into quantum states $|\phi(x)\rangle$ to facilitate classification tasks with higher expressiveness.

- **Quantum Kernels:** The QSVM computes a kernel matrix
- 
$$
K_{ij} = \langle \phi(x_i) | \phi(x_j) \rangle,
$$

where the inner product is evaluated using quantum circuits. The kernel matrix encodes the similarity between data points in the quantum-embedded feature space.

#### 3. Quantum Feature Mapping: The Mathematical Foundation

Given a classical data point  $x \in \mathbb{R}^n,$  we encode it into a quantum state using a parameterized quantum circuit $U(x).$ 
The state $|\phi(x)\rangle$  represents the transformed data point:

$$
|\phi(x)\rangle = U(x)|0\rangle^{\otimes m},
$$  

where  $|0\rangle^{\otimes m}$ is an $m$-qubit initial state, and  $U(x)$ is a unitary operator that encodes $x.$

The choice of $U(x)$ depends on the problem and can be tailored to exploit quantum interference and entanglement.

#### 4. Computing the Quantum Kernel Matrix

The quantum kernel is defined as:

$$
K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2.
$$  

To compute this kernel efficiently, we use a quantum computer to prepare the states  $|\phi(x_i)\rangle$ and $|\phi(x_j)\rangle,$ 
then measure their overlap using the swap test or Hadamard test:

**Swap Test Procedure:**

1. Prepare the quantum state  

$$
|\phi(x_i)\rangle \otimes |\phi(x_j)\rangle.
$$  

3. Introduce an ancillary qubit in the state
   
$$
|+\rangle = \frac{1}{\sqrt{2}} (|0\rangle + |1\rangle).
$$

5. Apply a controlled-swap operation conditioned on the ancillary qubit.
6. Measure the ancillary qubit in the $X$  basis. The probability of measuring $|0\rangle$  gives the value of  

$$
|\langle \phi(x_i) | \phi(x_j) \rangle|^2.
$$  

#### 5. Training the Quantum SVM

Once we have the quantum kernel matrix $K,$  the QSVM optimization problem becomes:

$$
\min_{\alpha} \frac{1}{2} \sum_{i, j = 1}^{N} \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_{i=1}^{N} \alpha_i \quad \text{subject to} \quad \sum_{i=1}^{N} \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C,
$$  

where $\alpha_i$ are the Lagrange multipliers and $C$ 

is a regularization parameter controlling the trade-off between maximizing the margin and minimizing the classification error. This optimization problem is solved using classical methods, but the kernel matrix is computed quantum mechanically, providing potential speedup for high-dimensional data.

#### 6. Mathematical Analysis of Quantum Advantage

The quantum advantage of QSVMs arises from their ability to map data to a feature space that is exponentially large compared to classical feature spaces, leveraging quantum interference and entanglement. Specifically, if the quantum feature map  $\phi(x)$  leads to non-trivial quantum states that cannot be efficiently simulated classically, then the quantum kernel can capture complex patterns that are inaccessible to classical SVMs.

Consider the quantum kernel complexity in terms of the number of qubits $m$ and data points $N$ The classical computation of kernel values in such high-dimensional spaces would generally require $O(N^2 \times 2^m)$ operations, which is infeasible for large $m.$ Quantum computation, however, can evaluate these kernel values in polynomial time with respect to $m.$   

#### 7. Example: QSVM for Binary Classification

Consider a simple binary classification problem where the data points  

$$
x_i \in \mathbb{R}^2
$$  

are mapped to quantum states using a feature map  $U(x_i).$ The QSVM computes the quantum kernel matrix:

$$
K_{ij} = |\langle \phi(x_i) | \phi(x_j) \rangle|^2,
$$  

and the optimization problem is solved to find the optimal hyperplane that separates the two classes. If the quantum feature map is carefully chosen, the QSVM can separate classes that are not linearly separable in the classical feature space.

#### 8. Challenges and Practical Considerations

While QSVMs offer theoretical advantages, practical challenges remain:

- **Noise and decoherence:** Quantum computers are still noisy, which affects the accuracy of quantum state preparation and measurements.
- **Scalability:** The size of the quantum kernel matrix grows quadratically with the number of data points, posing challenges for large datasets.


Quantum Support Vector Machines extend the power of classical SVMs by leveraging quantum computing to enhance feature mapping and kernel computation. By embedding data into high-dimensional quantum spaces, QSVMs can potentially achieve faster and more powerful classification, offering a glimpse into the potential advantages of quantum machine learning.

Example of the code:

```python
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import joblib
import matplotlib.pyplot as plt

# Download and load the Glass Identification dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
column_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
data = pd.read_csv(url, names=column_names, index_col="Id")

# Preprocess the data
X = data.drop(columns=["Type"]).values
y = data["Type"].values
y = np.where(y == 1, 1, -1)  # Convert labels to {-1, 1}

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the quantum device
n_qubits = X_train.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum kernel
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(x1[i], wires=i)
        qml.RZ(x2[i], wires=i)
    qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="ring")
    return qml.expval(qml.PauliZ(0))

def compute_kernel_row(x1, X2):
    return [quantum_kernel(x1, x2) for x2 in X2]

def kernel_matrix(X1, X2):
    n_jobs = -1  # Use all available CPU cores
    return np.array(Parallel(n_jobs=n_jobs)(delayed(compute_kernel_row)(x1, X2) for x1 in X1))

# Compute the kernel matrices
K_train = kernel_matrix(X_train, X_train)
K_test = kernel_matrix(X_test, X_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000]}
svm = SVC(kernel='precomputed')
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(K_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Save the trained model to a file
model_filename = 'qsvm_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")

# Load the trained model from a file
loaded_model = joblib.load(model_filename)
print("Model loaded from", model_filename)

# Evaluate the loaded model
y_pred_loaded = loaded_model.predict(K_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"Test accuracy of loaded model: {accuracy_loaded}")

# Visualize the decision boundary
def plot_decision_boundary(X, y, model, kernel_matrix_func):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Flatten the grid to pass into the kernel function
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute the kernel matrix for the grid points
    K_grid = kernel_matrix_func(grid_points, X_train)
    
    # Predict the class for each grid point
    Z = model.predict(K_grid)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("QSVM Decision Boundary")
    plt.show()

# Plot the decision boundary for the test set
plot_decision_boundary(X_test, y_test, loaded_model, kernel_matrix)
```

The visualization of the decision boundary can take a long time due to the computation of the quantum kernel for each point in the mesh grid. This involves a large number of quantum circuit evaluations, which can be computationally expensive.

