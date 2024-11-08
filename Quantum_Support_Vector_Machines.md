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

### Conclusion

Quantum Support Vector Machines extend the power of classical SVMs by leveraging quantum computing to enhance feature mapping and kernel computation. By embedding data into high-dimensional quantum spaces, QSVMs can potentially achieve faster and more powerful classification, offering a glimpse into the potential advantages of quantum machine learning.

