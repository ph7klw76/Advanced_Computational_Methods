# Quantum Support Vector Machines: A Mathematical Journey

## Introduction

Quantum Support Vector Machines (QSVMs) combine the principles of quantum computing with classical machine learning algorithms to provide a new approach to supervised learning. Specifically, QSVMs utilize the quantum computational power to perform kernel estimation in higher-dimensional spaces, potentially leading to improved classification capabilities compared to classical Support Vector Machines (SVMs). In this blog, we'll delve into the mathematical foundation behind QSVMs, the derivations involved, and the meaning of the mathematics, explaining each concept in a clear, scientifically accurate manner.

## Recap of Classical Support Vector Machines

Before diving into QSVMs, let us first recall the classical SVM formulation. SVMs are designed to find a hyperplane that maximally separates two classes of data points in a feature space. The optimal hyperplane is one that maximizes the margin between classes, which is the distance between the closest points of each class, called support vectors.

Mathematically, for a dataset consisting of $N$ training points ${(\mathbf{x}i, y_i)}{i=1}^N$, where $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in {-1, 1}$ are the class labels, the goal is to find a hyperplane parameterized by $\mathbf{w}$ and bias $b$ that satisfies:

To solve this problem, we convert it into a minimization problem. We aim to minimize the functional $L(\mathbf{w}, b)$:

subject to the constraints:

This is a constrained optimization problem, which can be solved using the method of Lagrange multipliers. The Lagrangian is defined as:

where $\alpha_i \geq 0$ are the Lagrange multipliers. The goal is to find the saddle point of $\mathcal{L}$ by minimizing with respect to $\mathbf{w}$ and $b$, and maximizing with respect to $\alpha_i$.

Setting the partial derivatives of $\mathcal{L}$ with respect to $\mathbf{w}$ and $b$ to zero gives:

Substituting these back into the Lagrangian yields the dual formulation of the SVM problem:

subject to:

The kernel trick is employed to handle non-linear decision boundaries. We replace the dot product $(\mathbf{x}_i^T \mathbf{x}_j)$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$ that implicitly maps the input data to a higher-dimensional feature space where it becomes linearly separable.

## Quantum Computing in Machine Learning

The advent of quantum computing offers new paradigms for computation, especially for problems that involve high-dimensional data. Quantum systems have the ability to represent data in exponentially larger spaces due to the principles of superposition and entanglement. This characteristic makes quantum computers well-suited for certain types of machine learning algorithms, such as QSVMs.

QSVMs can utilize quantum kernel estimation, which provides an efficient way of calculating the inner product between vectors in higher-dimensional feature spaces, a crucial step in classical SVMs.

## The Kernel Trick and Quantum Kernel

The key idea behind both classical SVMs and QSVMs is the kernel trick. Kernels allow us to implicitly map data from a low-dimensional space to a higher-dimensional one where a linear separator can be more easily found. In QSVMs, the kernel is computed via a quantum circuit, and this circuit can produce kernel values that would be infeasible to compute classically.

For QSVMs, let us denote the quantum state representing a data point $\mathbf{x}$ as $|\phi(\mathbf{x})
angle$. The quantum kernel is defined as:

This is the squared inner product of quantum states corresponding to different data points, which measures the similarity between them. The computation of this kernel function is done through a quantum circuit and requires quantum interference and entanglement, yielding results that can potentially outperform classical methods.

## Deriving the Quantum Support Vector Machine Formulation

To derive the formulation for QSVM, we start by constructing a quantum feature map. Suppose we have a feature map $\phi: \mathbb{R}^d 	o \mathcal{H}$, where $\mathcal{H}$ is the Hilbert space of a quantum system. The data vector $\mathbf{x}$ is encoded into a quantum state $|\phi(\mathbf{x})
angle$ using a unitary operator $U(\mathbf{x})$ such that:

The key idea is to use a parameterized quantum circuit $U(\mathbf{x})$ that transforms the initial state $|0
angle$ into a representation of the data point in a potentially high-dimensional quantum feature space.

The decision function of a classical SVM can be written as:

For QSVM, we replace the classical kernel $K(\mathbf{x}_i, \mathbf{x})$ with the quantum kernel $K_q(\mathbf{x}_i, \mathbf{x}) = |\langle \phi(\mathbf{x}_i)|\phi(\mathbf{x})
angle|^2$ computed by the quantum feature map. Thus, the QSVM decision function becomes:

## Training Process

Training the QSVM follows similar steps to the classical SVM, using the dual formulation with Lagrange multipliers. However, the key difference is that the computation of the kernel matrix is performed on a quantum computer. The quantum kernel matrix $K_q$ has entries $K_q(i, j) = |\langle \phi(\mathbf{x}_i) | \phi(\mathbf{x}_j)
angle|^2$, and this matrix is used during optimization.

## Detailed Derivation of the Dual Formulation

To derive the dual formulation in greater detail, consider the Lagrangian of the primal optimization problem. The goal is to minimize the primal objective function while satisfying the margin conditions:

To derive the dual form, we first take the gradient of $\mathcal{L}$ with respect to the primal variables $\mathbf{w}$ and $b$, and set them to zero:

Gradient with respect to $\mathbf{w}$:

Gradient with respect to $b$:

Substituting $\mathbf{w}$ back into the Lagrangian, we obtain the dual objective function. The dual form is concerned with maximizing $\mathcal{L}$ with respect to $\alpha_i$ while subject to certain constraints. The dual form of the objective function becomes:

subject to the constraints:

The kernel trick allows the substitution of the dot product $(\mathbf{x}_i^T \mathbf{x}_j)$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$, which represents an implicit mapping to a higher-dimensional feature space without explicitly computing that space.

Thus, the dual form for the kernelized SVM is given by:

The final classifier can be written as:

where $\alpha_i$ are the solution of the dual problem, and $b$ is determined from the support vectors.

## Quantum Circuit for Kernel Estimation

The practical implementation of QSVM requires a quantum circuit to evaluate the kernel matrix elements. The swap test is a common approach used to estimate the overlap $\langle \phi(\mathbf{x}_i)|\phi(\mathbf{x}_j)
angle$. The swap test circuit involves preparing two quantum registers in states $|\phi(\mathbf{x}_i)
angle$ and $|\phi(\mathbf{x}_j)
angle$, along with an ancilla qubit for measurement. By applying a series of controlled gates and measuring the ancilla qubit, we can estimate the inner product of the two states, thus providing the kernel value.

Mathematically, the swap test output gives us:

From this probability, we can compute the kernel value as:

This kernel estimation process allows QSVM to perform classification tasks similarly to classical SVMs but with a potentially more expressive feature space.

## Advantages and Challenges of QSVMs

# Advantages

High-dimensional Mapping: The ability of quantum computers to represent data in an exponentially large space allows QSVMs to leverage quantum kernels that are extremely difficult to compute classically. This leads to potential quantum speedups in training and classification tasks.

Expressive Power: The quantum feature map can represent complex relationships in data that may be inaccessible to classical kernels, improving the ability of SVMs to classify intricate datasets.

# Challenges

Quantum Noise: Current quantum hardware suffers from noise and decoherence, which affects the accuracy of kernel estimates and, consequently, the overall QSVM performance.

Scalability: For large datasets, the need to compute the quantum kernel for all pairs of data points can become computationally expensive, limiting the scalability of QSVMs.

Implementation Complexity: Designing and implementing quantum circuits that correctly encode data into quantum states and perform swap tests requires deep quantum expertise and precise calibration of quantum hardware.

## Summary

Quantum Support Vector Machines represent a powerful synergy between classical machine learning and quantum computing. By leveraging quantum feature spaces, QSVMs have the potential to outperform classical SVMs on certain types of data. The mathematics of QSVMs involves translating the classical kernel trick into the quantum domain, using quantum circuits to estimate kernel values that are classically intractable.

Despite current technological limitations, QSVMs are a promising step toward harnessing quantum power for practical machine learning tasks. They provide a glimpse into the possibilities of future quantum machine learning models, where the boundaries of computational complexity can be pushed even further.
