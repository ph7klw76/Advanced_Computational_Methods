### Foundation of Quantum Computing

## Q1

### Problem 1:
What is the inner product of:

$$
\frac{1}{\sqrt{2}} \lvert 0 \rangle + \frac{1}{\sqrt{2}} \lvert 1 \rangle
$$

and

$$
\frac{1}{\sqrt{3}} \lvert 0 \rangle + \frac{2}{\sqrt{3}} \lvert 1 \rangle ?
$$

### Problem 2:
And what is the inner product of:

$$
\frac{1}{\sqrt{2}} \lvert 0 \rangle + \frac{1}{\sqrt{2}} \lvert 1 \rangle
$$

and

$$
\frac{1}{\sqrt{2}} \lvert 0 \rangle - \frac{1}{\sqrt{2}} \lvert 1 \rangle ?
$$

## A1

### 1. First Inner Product:

The inner product is defined as:

$$
\langle \psi_1 | \psi_2 \rangle = \left( \frac{1}{\sqrt{2}} \lvert 0 \rangle + \frac{1}{\sqrt{2}} \lvert 1 \rangle \right) \cdot \left( \frac{1}{\sqrt{3}} \lvert 0 \rangle + \frac{2}{\sqrt{3}} \lvert 1 \rangle \right)
$$

We calculate the inner product term by term using the distributive property:

$$
\langle \psi_1 | \psi_2 \rangle = \left( \frac{1}{\sqrt{2}} \langle 0 | + \frac{1}{\sqrt{2}} \langle 1 | \right) \left( \frac{1}{\sqrt{3}} \lvert 0 \rangle + \frac{2}{\sqrt{3}} \lvert 1 \rangle \right)
$$

Now distribute:

$$
= \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{3}} \langle 0 | 0 \rangle + \frac{1}{\sqrt{2}} \cdot \frac{2}{\sqrt{3}} \langle 0 | 1 \rangle + \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{3}} \langle 1 | 0 \rangle + \frac{1}{\sqrt{2}} \cdot \frac{2}{\sqrt{3}} \langle 1 | 1 \rangle
$$

Using the orthogonality relations:

$$
\langle 0 | 0 \rangle = 1, \quad \langle 1 | 1 \rangle = 1, \quad \langle 0 | 1 \rangle = \langle 1 | 0 \rangle = 0
$$

We are left with:

$$
\langle \psi_1 | \psi_2 \rangle = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{3}} \cdot 1 + \frac{1}{\sqrt{2}} \cdot \frac{2}{\sqrt{3}} \cdot 1
$$

Simplifying:

$$
\langle \psi_1 | \psi_2 \rangle = \frac{1}{6} + \frac{2}{6} = \frac{3}{6} = \frac{1}{2}
$$

### 2. Second Inner Product:

Now, we calculate the inner product of:

$$
\langle \psi_1 | \psi_3 \rangle = \left( \frac{1}{\sqrt{2}} \lvert 0 \rangle + \frac{1}{\sqrt{2}} \lvert 1 \rangle \right) \cdot \left( \frac{1}{\sqrt{2}} \lvert 0 \rangle - \frac{1}{\sqrt{2}} \lvert 1 \rangle \right)
$$

Distribute:

$$
\langle \psi_1 | \psi_3 \rangle = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \langle 0 | 0 \rangle - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \langle 0 | 1 \rangle + \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \langle 1 | 0 \rangle - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \langle 1 | 1 \rangle
$$

Using orthogonality:

$$
\langle 0 | 0 \rangle = 1, \quad \langle 1 | 1 \rangle = 1, \quad \langle 0 | 1 \rangle = \langle 1 | 0 \rangle = 0
$$

So, we are left with:

$$
\langle \psi_1 | \psi_3 \rangle = \frac{1}{2} - \frac{1}{2} = 0
$$




![image](https://github.com/user-attachments/assets/c92f934b-1312-42db-b28f-03d793dc17dd)

### Quantum Gates

- **Identity (I):** Does nothing to the qubit.
- **Pauli-X (X):** Flips the qubit's state (like a classical NOT gate).
- **Hadamard (H):** Creates superposition states, combining the $\lvert 0 \rangle$ and $\lvert 1 \rangle$ states.
- **Controlled-NOT (CNOT):** Flips the second qubit if the first qubit is $\lvert 1 \rangle$.
- **Toffoli (CCNOT):** Flips the third qubit if both the first and second qubits are $\lvert 1 \rangle$.


## Q2: Pauli-X Gate: Unitarity, Inverse, and Action on a Qubit

### Check that the $X$ matrix is, indeed, unitary. 
What is the inverse of $X$? What is the action of $X$ on a general qubit in a state of the form:

$$
a \lvert 0 \rangle + b \lvert 1 \rangle
$$


## A2. Checking if the X Gate is Unitary

The **X gate**, also known as the Pauli-X gate, is defined by the matrix:

$$
X = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

A matrix is unitary if the inverse of the matrix is equal to its conjugate transpose (or Hermitian adjoint), i.e., 

$$
X^{-1} = X^{\dagger}
$$

To check this, we need to compute $X^{\dagger}$ and compare it to $X^{-1}$.

### Conjugate Transpose $X^{\dagger}$

Since $X$ is real, its conjugate transpose is the same as its transpose $X^{T}$:

$$
X^{\dagger} = X^{T} = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

Thus, 

$$
X^{\dagger} = X
$$

### Inverse of $X$

For a 2x2 matrix, the inverse can be calculated. However, for the Pauli-X gate:

$$
X^{-1} = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix} = X
$$

Thus, 

$$
X^{-1} = X
$$

Since $X^{-1} = X^{\dagger}$, the Pauli-X gate is indeed unitary.

## What is the Inverse of the X Gate?

As we saw above, the inverse of $X$ is itself:

$$
X^{-1} = X = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

## Action of the X Gate on a General Qubit

To find the action of the $X$ gate on a general qubit, we apply the matrix $X$ to the state $a\lvert 0 \rangle + b\lvert 1 \rangle$.

Let the general qubit state be:

$$
\lvert \psi \rangle = a \lvert 0 \rangle + b \lvert 1 \rangle = a \begin{pmatrix} 1 \\ 0 \end{pmatrix} + b \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} a \\ b \end{pmatrix}
$$

Now, apply the $X$ matrix to this state:

$$
X \lvert \psi \rangle = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} b \\ a \end{pmatrix}
$$

Thus, the action of $X$ on the qubit $a\lvert 0 \rangle + b\lvert 1 \rangle$ is to swap the amplitudes $a$ and $b$:

$$
X \left( a \lvert 0 \rangle + b \lvert 1 \rangle \right) = b \lvert 0 \rangle + a \lvert 1 \rangle
$$


# The Bloch Sphere and Qubit State Representation in Quantum Mechanics

The Bloch sphere offers a profound and geometrically intuitive representation of a single qubit’s state in quantum mechanics. It allows for a comprehensive understanding of quantum phenomena, including state superposition, unitary transformations, and measurement. This article delves deeply into the mathematics behind the Bloch sphere, focusing on the fundamental linear algebra, group theory, and operator formalism that make it an indispensable tool in quantum mechanics.

## 1. Qubit State as a Vector in Hilbert Space

A qubit exists in a complex two-dimensional vector space, Hilbert space $H$, spanned by the orthonormal basis vectors $\vert 0 \rangle$ and $\vert 1 \rangle$. Any pure qubit state $\vert \psi \rangle$ can be expressed as a linear combination of these basis vectors:

$$
\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle,
$$

where $\alpha, \beta \in \mathbb{C}$ are complex amplitudes satisfying the normalization condition:

$$
\vert \alpha \vert^2 + \vert \beta \vert^2 = 1.
$$

This normalization ensures that the probability of finding the qubit in one of its basis states sums to unity, a fundamental requirement of quantum mechanics.

### Pure State Parameterization and Spherical Coordinates

To map the qubit state $\vert \psi \rangle$ to the Bloch sphere, we convert $\alpha$ and $\beta$ into spherical coordinates. By defining:

$$
\alpha = \cos \left( \frac{\theta}{2} \right), \quad \beta = e^{i \varphi} \sin \left( \frac{\theta}{2} \right),
$$

we obtain:

$$
\vert \psi \rangle = \cos \left( \frac{\theta}{2} \right) \vert 0 \rangle + e^{i \varphi} \sin \left( \frac{\theta}{2} \right) \vert 1 \rangle,
$$

where:

- $\theta \in [0, \pi]$ represents the polar angle,
- $\varphi \in [0, 2\pi)$ represents the azimuthal angle.

This parameterization ensures a unique representation of the state on the Bloch sphere, with the angles $\theta$ and $\varphi$ corresponding to specific coordinates on the sphere.

### Cartesian Representation on the Bloch Sphere

The state $\vert \psi \rangle$ can now be associated with a unique point on the unit sphere, where the coordinates are given by:

$$
x = \sin(\theta) \cos(\varphi), \quad y = \sin(\theta) \sin(\varphi), \quad z = \cos(\theta).
$$

Thus, we represent $\vert \psi \rangle$ as a point $\vec{r} = (x, y, z)$ on the Bloch sphere, with $\vert \vec{r} \vert = 1$, forming the Bloch vector. This vector provides a full description of the qubit’s state in a geometric context.

## 2. Density Matrix Formalism and the Bloch Vector Representation

### Density Matrix for a Pure State

In quantum mechanics, the density matrix formalism is essential for describing both pure and mixed states. For a pure state $\vert \psi \rangle$, the density matrix $\rho$ is defined as:

$$
\rho = \vert \psi \rangle \langle \psi \vert,
$$

which expands as:

$$
\rho = \begin{pmatrix} \vert \alpha \vert^2 & \alpha \beta^* \\ \alpha^* \beta & \vert \beta \vert^2 \end{pmatrix}.
$$

### Bloch Vector and Pauli Matrix Expansion

The density matrix can be rewritten in terms of the Pauli matrices $\sigma_x, \sigma_y, \sigma_z$ and the identity matrix $I$ as follows:

$$
\rho = \frac{1}{2} \left( I + \vec{r} \cdot \vec{\sigma} \right),
$$

where:

- $\vec{r} = (x, y, z)$ is the Bloch vector,
- $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ represents the vector of Pauli matrices.

To derive this explicitly, recall that the Pauli matrices are given by:

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.
$$

The Bloch vector components $x, y, z$ can be computed as the expectation values of the Pauli matrices:

$$
x = \text{Tr}(\rho \sigma_x), \quad y = \text{Tr}(\rho \sigma_y), \quad z = \text{Tr}(\rho \sigma_z).
$$

The density matrix representation thus captures the geometric information of a qubit state through the Bloch vector. For pure states, $\vert \vec{r} \vert = 1$; for mixed states, $\vert \vec{r} \vert < 1$, indicating partial coherence.

## 3. Quantum Gates as Unitary Transformations on the Bloch Sphere

### Unitary Evolution and the Special Unitary Group SU(2)

Quantum gates correspond to unitary transformations that preserve the norm of the qubit state vector. For a single qubit, any unitary transformation can be represented as an element of the special unitary group $SU(2)$, the group of $2 \times 2$ unitary matrices with determinant $1$. Elements of $SU(2)$ can be written in exponential form as:

$$
U(\theta, \vec{n}) = e^{-i \frac{\theta}{2} \, \vec{n} \cdot \vec{\sigma}},
$$

where:

- $\theta$ is the rotation angle,
- $\vec{n} = (n_x, n_y, n_z)$ is a unit vector defining the rotation axis,
- $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ is the vector of Pauli matrices.

The rotation matrix for a qubit state on the Bloch sphere is then defined by the unitary transformation $R_{\vec{n}}(\theta)$, which corresponds to a rotation by $\theta$ about the axis $\vec{n}$ in real space.

### Examples of Quantum Gates as Rotations

- **Pauli-X Gate:** $X = e^{-i \frac{\pi}{2} \sigma_x}$  
  Acts as a $\pi$-rotation about the x-axis on the Bloch sphere.

- **Pauli-Y Gate:** $Y = e^{-i \frac{\pi}{2} \sigma_y}$  
  Acts as a $\pi$-rotation about the y-axis.

- **Pauli-Z Gate:** $Z = e^{-i \frac{\pi}{2} \sigma_z}$  
  Acts as a $\pi$-rotation about the z-axis.

- **Hadamard Gate:** The Hadamard gate can be decomposed as a $\pi$-rotation about the axis $\frac{1}{\sqrt{2}}(x+z)$ in the x-z plane.

Each gate modifies the qubit’s state vector on the Bloch sphere, allowing arbitrary transformations through combinations of basic rotations.

## 4. Measurement Theory in the Bloch Sphere Formalism

### Measurement in the Computational Basis

# Measurement Theory in the Bloch Sphere Formalism

Measurement in quantum mechanics collapses a qubit’s state onto one of the eigenstates of the measurement operator, a process that’s probabilistic and fundamentally different from unitary evolution. In the Bloch sphere formalism, measurements can be visualized as projecting the qubit’s Bloch vector along a chosen axis, typically reducing the qubit’s state to a specific basis state aligned with the measurement direction.

## 1. Projective Measurement in the Computational Basis

In the computational basis, a measurement on a qubit distinguishes between the states $\vert 0 \rangle$ and $\vert 1 \rangle$, which correspond to the z-axis on the Bloch sphere.

Given a qubit in a general state:

$$
\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle = \cos \left( \frac{\theta}{2} \right) \vert 0 \rangle + e^{i \varphi} \sin \left( \frac{\theta}{2} \right) \vert 1 \rangle,
$$

the density matrix of this state is:

$$
\rho = \vert \psi \rangle \langle \psi \vert = \begin{pmatrix} \cos^2 \left( \frac{\theta}{2} \right) & \cos \left( \frac{\theta}{2} \right) \sin \left( \frac{\theta}{2} \right) e^{-i \varphi} \\ \cos \left( \frac{\theta}{2} \right) \sin \left( \frac{\theta}{2} \right) e^{i \varphi} & \sin^2 \left( \frac{\theta}{2} \right) \end{pmatrix}.
$$

A measurement in the computational basis projects $\vert \psi \rangle$ onto $\vert 0 \rangle$ or $\vert 1 \rangle$, with probabilities:

$$
P(0) = \langle 0 \vert \rho \vert 0 \rangle = \cos^2 \left( \frac{\theta}{2} \right), \quad P(1) = \langle 1 \vert \rho \vert 1 \rangle = \sin^2 \left( \frac{\theta}{2} \right).
$$

If the measurement result is $\vert 0 \rangle$, the post-measurement state of the qubit becomes:

$$
\rho' = \vert 0 \rangle \langle 0 \vert = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.
$$

This outcome corresponds to the Bloch vector pointing to the north pole of the Bloch sphere, $(0, 0, 1)$. Similarly, if the result is $\vert 1 \rangle$, the state becomes:

$$
\rho' = \vert 1 \rangle \langle 1 \vert = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix},
$$

corresponding to the Bloch vector pointing to the south pole of the Bloch sphere, $(0, 0, -1)$. Thus, measurement in the computational basis projects the qubit state onto the z-axis.

### Example: Measurement of a Superposition State

Consider a qubit in the superposition state:

$$
\vert \psi \rangle = \frac{1}{\sqrt{2}} \left( \vert 0 \rangle + \vert 1 \rangle \right),
$$

which corresponds to a Bloch vector along the x-axis (equator) of the Bloch sphere, $\vec{r} = (1, 0, 0)$. The density matrix is:

$$
\rho = \vert \psi \rangle \langle \psi \vert = \frac{1}{2} \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}.
$$

The probability of measuring $\vert 0 \rangle$ is:

$$
P(0) = \langle 0 \vert \rho \vert 0 \rangle = \frac{1}{2},
$$

and similarly, $P(1) = \frac{1}{2}$. After measuring the state and finding it in $\vert 0 \rangle$, the state collapses to:

$$
\rho' = \vert 0 \rangle \langle 0 \vert = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.
$$

If the measurement result is $\vert 1 \rangle$, the state collapses to:

$$
\rho' = \vert 1 \rangle \langle 1 \vert = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}.
$$

Each measurement outcome projects the qubit from the equator of the Bloch sphere (superposition) onto the z-axis, representing a collapse to a definitive state.

## 2. Projective Measurement Along an Arbitrary Axis

To measure a qubit along an arbitrary axis defined by a unit vector $\vec{n} = (n_x, n_y, n_z)$ on the Bloch sphere, we construct projectors based on the two eigenstates of $\vec{n} \cdot \vec{\sigma}$, where $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ is the vector of Pauli matrices. The measurement operators are:

$$
M_{\pm} = \frac{1}{2} (I \pm \vec{n} \cdot \vec{\sigma}),
$$

where:

- $M_+$ projects onto the state aligned with $\vec{n}$,
- $M_-$ projects onto the state anti-aligned with $\vec{n}$.

If the state before measurement is represented by density matrix $\rho$, the probability of obtaining the measurement result $M_+$ is:

$$
P(+) = \text{Tr}(\rho M_+) = \frac{1}{2} \left( 1 + \vec{r} \cdot \vec{n} \right),
$$

and the probability of obtaining $M_-$ is:

$$
P(-) = \text{Tr}(\rho M_-) = \frac{1}{2} \left( 1 - \vec{r} \cdot \vec{n} \right).
$$

If $M_+$ is the outcome, the post-measurement state is:

$$
\rho' = \frac{M_+ \rho M_+}{\text{Tr}(\rho M_+)} = M_+.
$$

### Example: Measurement Along the X-Axis

Consider a qubit in the state:

$$
\vert \psi \rangle = \cos \left( \frac{\pi}{8} \right) \vert 0 \rangle + \sin \left( \frac{\pi}{8} \right) \vert 1 \rangle.
$$

This corresponds to a Bloch vector in an arbitrary direction between the z- and x-axes. To measure along the x-axis, we set $\vec{n} = (1, 0, 0)$. The measurement operators are:

$$
M_{\pm} = \frac{1}{2} (I \pm \sigma_x) = \frac{1}{2} \begin{pmatrix} 1 & \pm 1 \\ \pm 1 & 1 \end{pmatrix}.
$$

The probability of measuring the state aligned with the x-axis (outcome $M_+$) is:

$$
P(+) = \text{Tr}(\rho M_+) = \frac{1}{2} \left( 1 + \langle \sigma_x \rangle \right).
$$

The post-measurement state, if the outcome is $M_+$, is the projector $M_+$.

## 3. Generalized Measurement: POVMs

Positive Operator-Valued Measurements (POVMs) extend projective measurements to include non-orthogonal outcomes, a broader class of measurements that are particularly useful in quantum information tasks, such as state discrimination.

For a POVM measurement, we have a set of measurement operators $\{E_i\}$, which satisfy:

$$
\sum_i E_i = I, \quad E_i \geq 0.
$$

The probability of obtaining outcome $i$ when the system is in state $\rho$ is given by:

$$
P(i) = \text{Tr}(\rho E_i).
$$

If the measurement outcome is $i$, the state after the measurement (in the generalized sense) can be given by a corresponding post-measurement map, although in some cases no definitive post-measurement state exists.

### Example: POVM for Quantum State Discrimination

Consider two non-orthogonal states $\vert \psi_1 \rangle = \vert 0 \rangle$ and $\vert \psi_2 \rangle = \cos(\theta) \vert 0 \rangle + \sin(\theta) \vert 1 \rangle$, and suppose we wish to distinguish between them. A POVM with elements:

$$
E_1 = \begin{pmatrix} 1 & 0 \\
0 & 0 \end{pmatrix}, \quad E_2 = \begin{pmatrix} 0 & 0 \\
0 & 1 \end{pmatrix},
$$

represents a measurement that discriminates between the two states probabilistically, with outcome probabilities based on the overlap between $\vert \psi_1 \rangle$ and $\vert \psi_2 \rangle$.


# Tensor Product Spaces in Quantum Mechanics

In quantum mechanics, the tensor product of vector spaces allows us to represent composite quantum systems. For two-qubit systems, the tensor product combines the state spaces of individual qubits into a joint state space, capturing all possible configurations of the combined system.

If a single qubit has a state space $\mathbb{C}^2$, then a system of two qubits resides in the space $\mathbb{C}^2 \otimes \mathbb{C}^2 = \mathbb{C}^4$. This combined space can describe entangled states and the effects of quantum gates on multiple qubits.

## 1. Defining the Tensor Product of Vectors and Matrices

### Tensor Product of Basis States

For two qubits, each qubit has basis states $\vert 0 \rangle$ and $\vert 1 \rangle$, represented by vectors in $\mathbb{C}^2$:

$$
\vert 0 \rangle = \begin{pmatrix} 1 \\
0 \end{pmatrix}, \quad \vert 1 \rangle = \begin{pmatrix} 0 \\
1 \end{pmatrix}.
$$

The tensor product of two states $\vert a \rangle$ and $\vert b \rangle$ from separate qubits is denoted $\vert a \rangle \otimes \vert b \rangle$ or simply $\vert ab \rangle$. The tensor product is computed as follows:

$$
\vert 0 \rangle \otimes \vert 0 \rangle = \begin{pmatrix} 1 \\
0 \end{pmatrix} \otimes \begin{pmatrix} 1 \\
0 \end{pmatrix} = \begin{pmatrix} 1 \cdot \begin{pmatrix} 1 \\
0 \end{pmatrix} \\
0 \cdot \begin{pmatrix} 1 \\
0 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 1 \\
0 \\
0 \\
0 \end{pmatrix},
$$

$$
\vert 0 \rangle \otimes \vert 1 \rangle = \begin{pmatrix} 1 \\
0 \end{pmatrix} \otimes \begin{pmatrix} 0 \\
1 \end{pmatrix} = \begin{pmatrix} 1 \cdot \begin{pmatrix} 0 \\
1 \end{pmatrix} \\
0 \cdot \begin{pmatrix} 0 \\
1 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 0 \\
1 \\
0 \\
0 \end{pmatrix},
$$

$$
\vert 1 \rangle \otimes \vert 0 \rangle = \begin{pmatrix} 0 \\
1 \end{pmatrix} \otimes \begin{pmatrix} 1 \\
0 \end{pmatrix} = \begin{pmatrix} 0 \cdot \begin{pmatrix} 1 \\
0 \end{pmatrix} \\
1 \cdot \begin{pmatrix} 1 \\
0 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 0 \\
0 \\
1 \\
0 \end{pmatrix},
$$

$$
\vert 1 \rangle \otimes \vert 1 \rangle = \begin{pmatrix} 0 \\
1 \end{pmatrix} \otimes \begin{pmatrix} 0 \\
1 \end{pmatrix} = \begin{pmatrix} 0 \cdot \begin{pmatrix} 0 \\
1 \end{pmatrix} \\
1 \cdot \begin{pmatrix} 0 \\ 1 \end{pmatrix} \end{pmatrix} = \begin{pmatrix} 0 \\
0 \\
0 \\
1 \end{pmatrix}.
$$

These results represent the four possible basis states for the two-qubit system in the combined $\mathbb{C}^4$ space:

$$
\{ \vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle \} = \{\{ \begin{pmatrix} 1 \\
0 \\
0 \\
0 \end{pmatrix}, \begin{pmatrix} 0 \\
1 \\
0 \\
0 \end{pmatrix}, \begin{pmatrix} 0 \\
0 \\
1 \\
0 \end{pmatrix}, \begin{pmatrix} 0 \\
0 \\
0 \\
1 \end{pmatrix}\}\}
$$


### Tensor Product of Matrices

For operators acting on multiple qubits, we also use the tensor product. If $A$ and $B$ are operators on single qubits, their combined action on a two-qubit system is represented by the tensor product $A \otimes B$.

For example, if:

$$
A = \begin{pmatrix} a_{11} & a_{12} \\
a_{21} & a_{22} \end{pmatrix}, \quad B = \begin{pmatrix} b_{11} & b_{12} \\
b_{21} & b_{22} \end{pmatrix},
$$

then:

$$
A \otimes B = \begin{pmatrix} a_{11} B & a_{12} B \\
a_{21} B & a_{22} B \end{pmatrix} = \begin{pmatrix} a_{11} b_{11} & a_{11} b_{12} & a_{12} b_{11} & a_{12} b_{12} \\
a_{11} b_{21} & a_{11} b_{22} & a_{12} b_{21} & a_{12} b_{22} \\
a_{21} b_{11} & a_{21} b_{12} & a_{22} b_{11} & a_{22} b_{12} \\
a_{21} b_{21} & a_{21} b_{22} & a_{22} b_{21} & a_{22} b_{22} \end{pmatrix}.
$$

This tensor product matrix acts on the combined $\mathbb{C}^4$ space of the two qubits.

### Example: Tensor Product of Pauli Matrices

Let’s compute the tensor product of two Pauli matrices $\sigma_x$ and $\sigma_y$, where:

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}.
$$

Then:

$$
\sigma_x \otimes \sigma_y = \begin{pmatrix} 0 & 1 \\
1 & 0 \end{pmatrix} \otimes \begin{pmatrix} 0 & -i \\
i & 0 \end{pmatrix} = \begin{pmatrix} 0 \cdot \begin{pmatrix} 0 & -i \\
i & 0 \end{pmatrix} & 1 \cdot \begin{pmatrix} 0 & -i \\
i & 0 \end{pmatrix} \\
1 \cdot \begin{pmatrix} 0 & -i \\
i & 0 \end{pmatrix} & 0 \cdot \begin{pmatrix} 0 & -i \\
i & 0 \end{pmatrix} \end{pmatrix}.
$$

Carrying out the multiplications gives:

$$
\sigma_x \otimes \sigma_y = \begin{pmatrix} 0 & 0 & 0 & -i \\
0 & 0 & i & 0 \\
0 & -i & 0 & 0 \\
i & 0 & 0 & 0 \end{pmatrix}.
$$

This combined matrix represents the action of $\sigma_x$ on the first qubit and $\sigma_y$ on the second qubit in the two-qubit system.

## 2. Examples of Tensor Product States and Operations

### Example 1: Constructing a Product State

Suppose qubit $A$ is in the state $\vert + \rangle = \frac{1}{\sqrt{2}}(\vert 0 \rangle + \vert 1 \rangle)$, and qubit $B$ is in the state $\vert 0 \rangle$. The two-qubit state $\vert \Psi \rangle$ is then:

$$
\vert \Psi \rangle = \vert + \rangle \otimes \vert 0 \rangle = \frac{1}{\sqrt{2}}(\vert 0 \rangle + \vert 1 \rangle) \otimes \vert 0 \rangle = \frac{1}{\sqrt{2}}(\vert 0 \rangle \otimes \vert 0 \rangle + \vert 1 \rangle \otimes \vert 0 \rangle).
$$

Expanding this, we get:

$$
\vert \Psi \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 10 \rangle) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\
0 \\
1 \\
0 \end{pmatrix}.
$$

This result represents a separable two-qubit state in $\mathbb{C}^4$.

### Example 2: Entangled State Construction - The Bell State $\vert \Phi^+ \rangle$

The Bell state $\vert \Phi^+ \rangle$ is one of the maximally entangled states for a two-qubit system and is defined as:

$$
\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 11 \rangle).
$$

In vector form, $\vert \Phi^+ \rangle$ is represented as:

$$
\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\
0 \\
0 \\
1 \end{pmatrix}.
$$

Unlike separable states, $\vert \Phi^+ \rangle$ cannot be factored into the tensor product of two single-qubit states. This is an example of an entangled state.

### Example 3: Applying a Gate to a Two-Qubit State

Consider the CNOT gate, a two-qubit gate often used to create entanglement. The CNOT gate flips the second qubit (target qubit) if the first qubit (control qubit) is $\vert 1 \rangle$. The CNOT gate is represented by the matrix:

$$
\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 1 \\ 
0 & 0 & 1 & 0 \end{pmatrix}.
$$

Suppose we start with the two-qubit state $\vert + \rangle \otimes \vert 0 \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 10 \rangle)$. Applying the CNOT gate to this state, we get:

$$
\text{CNOT} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\
0 \\
1 \\ 
0 \end{pmatrix} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\
0 \\
0 \\
1 \end{pmatrix} = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 11 \rangle),
$$

which is the entangled Bell state $\vert \Phi^+ \rangle$. This shows that the CNOT gate, when applied to a product state, can create entanglement between qubits.

# 1. Representing Two-Qubit States

## Tensor Product Spaces

A single qubit exists in a two-dimensional complex vector space, typically denoted $\mathbb{C}^2$, with basis states $\vert 0 \rangle$ and $\vert 1 \rangle$. For a two-qubit system, the total state space is the tensor product of two single-qubit spaces:

$$
H_{AB} = H_A \otimes H_B = \mathbb{C}^2 \otimes \mathbb{C}^2 = \mathbb{C}^4.
$$

The basis states of $H_{AB}$ are given by the tensor products of the basis states of each qubit:

$$
\{ \vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle \},
$$

where, for instance, $\vert 00 \rangle = \vert 0 \rangle \otimes \vert 0 \rangle$, meaning that both qubits are in the $\vert 0 \rangle$ state.

## Constructing Two-Qubit States

Given qubits $A$ and $B$ in states $\vert \psi \rangle_A = \alpha \vert 0 \rangle + \beta \vert 1 \rangle$ and $\vert \phi \rangle_B = \gamma \vert 0 \rangle + \delta \vert 1 \rangle$, the product state $\vert \psi \rangle_A \otimes \vert \phi \rangle_B$ is:

$$
\vert \Psi \rangle = (\alpha \vert 0 \rangle + \beta \vert 1 \rangle) \otimes (\gamma \vert 0 \rangle + \delta \vert 1 \rangle) \\
\\
= \alpha \gamma \vert 00 \rangle + \alpha \delta \vert 01 \rangle + \beta \gamma \vert 10 \rangle + \beta \delta \vert 11 \rangle.
$$

This tensor product produces a four-dimensional vector that represents all possible combinations of states for qubits $A$ and $B$.

### Example: Product State Calculation

Suppose qubit $A$ is in the state $\vert \psi \rangle_A = \frac{1}{\sqrt{2}}(\vert 0 \rangle + \vert 1 \rangle)$ and qubit $B$ is in the state $\vert \phi \rangle_B = \vert 0 \rangle$. Then the joint state is:

$$
\vert \Psi \rangle = \left( \frac{1}{\sqrt{2}} (\vert 0 \rangle + \vert 1 \rangle) \right) \otimes \vert 0 \rangle \\
\\
= \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 10 \rangle).
$$

This is a separable state, meaning that it can be written as a tensor product of individual qubit states.

# 2. Separable and Entangled States

## Definition of Separable States

A two-qubit state $\vert \Psi \rangle \in H_{AB}$ is separable if it can be written as:

$$
\vert \Psi \rangle = \vert \psi \rangle_A \otimes \vert \phi \rangle_B,
$$

where $\vert \psi \rangle_A \in H_A$ and $\vert \phi \rangle_B \in H_B$ are the states of qubits $A$ and $B$, respectively. If a state cannot be factored into the tensor product of two single-qubit states, it is entangled.

For example, the state $\frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 11 \rangle)$ cannot be written as a tensor product of individual qubit states, hence it is entangled.

## Definition of Entangled States

Entanglement is a uniquely quantum mechanical phenomenon where the state of each qubit in a two-qubit system cannot be independently described. Mathematically, a state $\vert \Psi \rangle \in H_{AB}$ is entangled if it cannot be decomposed into a product state.

### Example: Bell States

The Bell states are maximally entangled two-qubit states, and they form an orthonormal basis for the two-qubit Hilbert space $H_{AB}$. They are defined as follows:

- **$\vert \Phi^+ \rangle$:**

$$
  \vert \Phi^+ \rangle = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 11 \rangle).
$$

- **$\vert \Phi^- \rangle$:**

$$
  \vert \Phi^- \rangle = \frac{1}{\sqrt{2}} (\vert 00 \rangle - \vert 11 \rangle).
$$

- **$\vert \Psi^+ \rangle$:**

$$
  \vert \Psi^+ \rangle = \frac{1}{\sqrt{2}} (\vert 01 \rangle + \vert 10 \rangle).
$$

- **$\vert \Psi^- \rangle$:**

$$
  \vert \Psi^- \rangle = \frac{1}{\sqrt{2}} (\vert 01 \rangle - \vert 10 \rangle).
$$

Each Bell state is entangled, as it cannot be factored into individual qubit states. These states are essential in quantum information theory and serve as foundational states in quantum communication and quantum cryptography.

# 3. Entanglement and Density Matrices

The density matrix formalism provides a complete description of both pure and mixed states in quantum mechanics. For two qubits, the density matrix $\rho$ of a state $\vert \Psi \rangle$ is given by:

$$
\rho = \vert \Psi \rangle \langle \Psi \vert.
$$

## Reduced Density Matrices and Entanglement

For entangled states, examining the reduced density matrix of each subsystem helps determine the degree of entanglement. Given a joint density matrix $\rho_{AB}$ of two qubits, the reduced density matrix of qubit $A$ (ignoring qubit $B$) is obtained by taking the partial trace over the $B$ subsystem:

$$
\rho_A = \operatorname{Tr}_B (\rho_{AB})
$$

### Example: Reduced Density Matrix of a Bell State

Consider the Bell state $\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 11 \rangle)$. The density matrix $\rho_{AB} = \vert \Phi^+ \rangle \langle \Phi^+ \vert$ is:

$$
\rho_{AB} = \frac{1}{2} \begin{pmatrix} 1 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 \end{pmatrix}.
$$

To obtain the reduced density matrix $\rho_A$, we trace out qubit $B$:

$$
\rho_A = \text{Tr}_B (\rho_{AB}) \\
\\
= \frac{1}{2} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \frac{1}{2} I.
$$

The reduced density matrix $\rho_A = \frac{1}{2} I$ represents a completely mixed state, meaning that measurements on qubit $A$ alone yield no information about the original Bell state. This mixed state result is a hallmark of entanglement, as the individual qubits appear maximally uncertain when considered independently.

# 4. Entanglement Measures: Concurrence and Entropy

To quantify entanglement in a two-qubit system, we use mathematical measures like concurrence and entropy of entanglement.

## Concurrence

# Concurrence

## Definition of Concurrence

Concurrence is a measure of entanglement introduced by William Wootters, specifically useful for quantifying the entanglement of formation in a two-qubit system. For a two-qubit pure state $\vert \Psi \rangle$, concurrence $C$ can be calculated as follows:

### Step 1: Define the Spin-Flipped State

Define the spin-flipped state $\vert \tilde{\Psi} \rangle$ of $\vert \Psi \rangle$:

$$
\vert \tilde{\Psi} \rangle = (\sigma_y \otimes \sigma_y) \vert \Psi^* \rangle,
$$

where $\vert \Psi^* \rangle$ is the complex conjugate of $\vert \Psi \rangle$ in the computational basis, and $\sigma_y$ is the Pauli-Y matrix:

$$
\sigma_y = \begin{pmatrix} 0 & -i \\
i & 0 \end{pmatrix}.
$$

### Step 2: Compute the Overlap

Compute the overlap between $\vert \Psi \rangle$ and $\vert \tilde{\Psi} \rangle$:

$$
C = \vert \langle \Psi \vert \tilde{\Psi} \rangle \vert,
$$

where $C \in [0,1]$, with $C = 0$ for separable (non-entangled) states and $C = 1$ for maximally entangled states.

For mixed states represented by a density matrix $\rho$, concurrence is calculated using an extended formula, which we’ll cover after illustrating the pure state example.

### Example: Concurrence of a Bell State

Consider the Bell state $\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 11 \rangle)$, which is a maximally entangled two-qubit state.

The complex conjugate $\vert \Phi^+ \rangle^*$ of $\vert \Phi^+ \rangle$ in the computational basis is:

$$
\vert \Phi^+ \rangle^* = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 11 \rangle).
$$

The spin-flipped state $\vert \tilde{\Phi}^+ \rangle$ is:

$$
\vert \tilde{\Phi}^+ \rangle = (\sigma_y \otimes \sigma_y) \vert \Phi^+ \rangle^* = \frac{1}{\sqrt{2}} (\vert 11 \rangle + \vert 00 \rangle).
$$

Calculating the overlap between $\vert \Phi^+ \rangle$ and $\vert \tilde{\Phi}^+ \rangle$:

$$
C = \vert \langle \Phi^+ \vert \tilde{\Phi}^+ \rangle \vert = \left\vert \frac{1}{\sqrt{2}} \langle 00 \vert + \frac{1}{\sqrt{2}} \langle 11 \vert \right\vert \frac{1}{\sqrt{2}} (\vert 11 \rangle + \vert 00 \rangle) = 1.
$$

Since $C = 1$, $\vert \Phi^+ \rangle$ is maximally entangled.

## Concurrence for Mixed States

For a two-qubit mixed state $\rho$, the concurrence is computed by a more involved process:

### Step 1: Define the Spin-Flipped Density Matrix

Define the spin-flipped density matrix $\tilde{\rho}$ as:

$$
\tilde{\rho} = (\sigma_y \otimes \sigma_y) \rho^* (\sigma_y \otimes \sigma_y),
$$

where $\rho^*$ is the complex conjugate of $\rho$.

### Step 2: Calculate the Eigenvalues

Calculate the eigenvalues $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ of the matrix $R = \sqrt{\rho \tilde{\rho}}$ in descending order.

### Step 3: Compute Concurrence

Compute concurrence $C$ as:

$$
C = \max (0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4).
$$

This method allows us to measure entanglement even when dealing with mixed states, accounting for the loss of purity due to decoherence or environmental interaction.

# 2. Entropy of Entanglement

## Definition of Entropy of Entanglement

The entropy of entanglement is a measure that quantifies the amount of entanglement in a pure two-qubit state. It uses the concept of von Neumann entropy to evaluate the uncertainty of the state of one qubit when the other qubit is known. For a pure state $\vert \Psi \rangle$ in a two-qubit system, we compute the entropy of entanglement by first finding the reduced density matrix of one qubit.

### Step 1: Calculate the Reduced Density Matrix

Calculate the reduced density matrix $\rho_A$ of qubit $A$ by taking the partial trace over qubit $B$:

$$
\rho_A = \text{Tr}_B (\vert \Psi \rangle \langle \Psi \vert).
$$

### Step 2: Compute the von Neumann Entropy

Compute the von Neumann entropy $S(\rho_A)$ of the reduced density matrix $\rho_A$:

$$
S(\rho_A) = - \text{Tr}(\rho_A \log_2 \rho_A).
$$

The von Neumann entropy $S(\rho_A)$ represents the entropy of entanglement for the state $\vert \Psi \rangle$. For maximally entangled states, $S(\rho_A) = 1$, and for separable states, $S(\rho_A) = 0$.

### Example: Entropy of Entanglement of a Bell State

Let’s calculate the entropy of entanglement for the Bell state $\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 11 \rangle)$.

The density matrix of $\vert \Phi^+ \rangle$ is:

$$
\rho_{AB} = \vert \Phi^+ \rangle \langle \Phi^+ \vert = \frac{1}{2} \begin{pmatrix} 1 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 \end{pmatrix}.
$$

We take the partial trace over qubit $B$ to obtain the reduced density matrix $\rho_A$:

$$
\rho_A = \text{Tr}_B (\rho_{AB}) = \frac{1}{2} \begin{pmatrix} 1 & 0 \\
0 & 1 \end{pmatrix} = \frac{1}{2} I.
$$

The eigenvalues of $\rho_A$ are $\frac{1}{2}$ and $\frac{1}{2}$, and the von Neumann entropy is:

$$
S(\rho_A) = -\left( \frac{1}{2} \log_2 \frac{1}{2} + \frac{1}{2} \log_2 \frac{1}{2} \right) = 1.
$$

Since $S(\rho_A) = 1$, $\vert \Phi^+ \rangle$ is maximally entangled, as expected for a Bell state.

## Entropy of Entanglement for a Separable State

Consider a separable state $\vert \Psi \rangle = \vert 0 \rangle \otimes \vert + \rangle$, where $\vert + \rangle = \frac{1}{\sqrt{2}} (\vert 0 \rangle + \vert 1 \rangle)$.

The density matrix of $\vert \Psi \rangle$ is:

$$
\rho_{AB} = \vert \Psi \rangle \langle \Psi \vert = \frac{1}{2} \begin{pmatrix} 1 & 1 \\
1 & 1 \end{pmatrix} \otimes \begin{pmatrix} 1 & 0 \\
0 & 0 \end{pmatrix} = \frac{1}{2} \begin{pmatrix} 1 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 \end{pmatrix}.
$$

We calculate the reduced density matrix $\rho_A$ by tracing out qubit $B$:

$$
\rho_A = \text{Tr}_B (\rho_{AB}) = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.
$$

The von Neumann entropy $S(\rho_A)$ is:

$$
S(\rho_A) = -\left( 1 \log_2 1 + 0 \log_2 0 \right) = 0.
$$

Since $S(\rho_A) = 0$, the state $\vert \Psi \rangle = \vert 0 \rangle \otimes \vert + \rangle$ is separable, as expected.

For a maximally entangled state, such as a Bell state, $S(\rho_A) = 1$, representing maximum entanglement. For a separable state, $S(\rho_A) = 0$, as there is no entanglement.

# 5. Bell Test and Non-Local Correlations

One of the most profound implications of entanglement is the existence of non-local correlations that defy classical intuition, as demonstrated by Bell’s theorem. In an entangled two-qubit system, measurements on qubits $A$ and $B$ exhibit correlations that cannot be explained by classical mechanics or local hidden variables.

Bell's theorem shows that entangled states violate Bell inequalities, suggesting that quantum mechanics allows for correlations stronger than any classical system. Bell states are especially important because they maximize these correlations.

### Example: Violation of Bell Inequalities

For a Bell state like $\vert \Phi^+ \rangle$, measuring specific observables on each qubit in different directions produces correlations that exceed the bounds set by classical probability. This Bell inequality violation is direct evidence of entanglement, emphasizing the unique properties of two-qubit states in quantum mechanics.

# Mathematical Statement of the No-Cloning Theorem

If we have an arbitrary quantum state $\vert \psi \rangle$ that we wish to duplicate, we might imagine a hypothetical "quantum cloning machine" that could take this state and produce a copy. Mathematically, we want to find an operation $U$ that can take an initial state $\vert \psi \rangle \otimes \vert e \rangle$ (where $\vert e \rangle$ is some blank or auxiliary state) and transform it into $\vert \psi \rangle \otimes \vert \psi \rangle$. That is, for a general unknown quantum state $\vert \psi \rangle$, we want:

$$
U(\vert \psi \rangle \otimes \vert e \rangle) = \vert \psi \rangle \otimes \vert \psi \rangle.
$$

However, the no-cloning theorem states that there is no unitary operator $U$ that can achieve this for all possible quantum states $\vert \psi \rangle$.

# Proof Outline of the No-Cloning Theorem

The proof of the no-cloning theorem can be outlined by assuming that cloning is possible and showing that this assumption leads to a contradiction.

### Assume Cloning is Possible

Suppose there exists a unitary operator $U$ that can clone any two distinct, arbitrary quantum states $\vert \psi \rangle$ and $\vert \phi \rangle$. Then we would have:

$$
U(\vert \psi \rangle \otimes \vert e \rangle) = \vert \psi \rangle \otimes \vert \psi \rangle,
$$

$$
U(\vert \phi \rangle \otimes \vert e \rangle) = \vert \phi \rangle \otimes \vert \phi \rangle.
$$

### Apply the Linearity of Quantum Mechanics

In quantum mechanics, unitary transformations are linear. So if $U$ is indeed a unitary operator, it must satisfy linearity:

$$
U((\alpha \vert \psi \rangle + \beta \vert \phi \rangle) \otimes \vert e \rangle) = \alpha U(\vert \psi \rangle \otimes \vert e \rangle) + \beta U(\vert \phi \rangle \otimes \vert e \rangle).
$$

### Check the Desired Cloning Result

If cloning were possible, then for the superposition $\alpha \vert \psi \rangle + \beta \vert \phi \rangle$, we should also have:

$$
U((\alpha \vert \psi \rangle + \beta \vert \phi \rangle) \otimes \vert e \rangle) = (\alpha \vert \psi \rangle + \beta \vert \phi \rangle) \otimes (\alpha \vert \psi \rangle + \beta \vert \phi \rangle).
$$

### Arrive at a Contradiction

The two results from steps 2 and 3 do not generally match. The output of a linear transformation (step 2) does not match the result of cloning the superposition state (step 3), except for special cases where $\vert \psi \rangle$ and $\vert \phi \rangle$ are orthogonal (i.e., they are mutually exclusive and do not overlap in any superposition). Thus, our assumption that $U$ could clone any arbitrary quantum state leads to a contradiction.

Since a unitary transformation cannot satisfy these requirements simultaneously, we conclude that an unknown quantum state cannot be cloned.

# Implications of the No-Cloning Theorem

The no-cloning theorem has several critical implications for quantum mechanics and quantum information:

- **Quantum Cryptography**: The no-cloning theorem is fundamental to quantum key distribution (QKD) protocols, like the BB84 protocol. Since an unknown quantum state cannot be copied, any eavesdropper attempting to intercept and duplicate a quantum key will inevitably disturb the state, revealing their presence.

- **Quantum Computing and Quantum Information**: In classical computing, data can be duplicated and transmitted without loss of information. However, in quantum computing, the no-cloning theorem implies that qubits cannot be copied arbitrarily. This imposes constraints on quantum algorithms and error correction techniques, which must work around the inability to duplicate quantum information.

- **Limits of Quantum Communication**: Quantum information cannot be "amplified" in the classical sense because amplification would imply creating multiple copies of a quantum state. Therefore, long-distance quantum communication must rely on quantum repeaters that do not violate the no-cloning theorem but instead use entanglement swapping and quantum teleportation to extend communication channels.

- **Quantum Teleportation**: The no-cloning theorem plays a role in quantum teleportation, which allows the transfer of quantum states between distant locations without physically transmitting the qubit itself. Teleportation depends on entanglement and classical communication rather than cloning, adhering to the no-cloning restriction.

The no-cloning theorem establishes that arbitrary unknown quantum states cannot be duplicated. It is a direct consequence of the linear structure of quantum mechanics and the superposition principle, fundamentally distinguishing quantum systems from classical systems. This theorem underpins several core areas of quantum information science, ensuring the security of quantum communication and shaping the operational limitations in quantum computing and information transfer.


# Controlled Gate

Controlled gates are indispensable in quantum computing for creating conditional operations, enabling entanglement, achieving universality of quantum computation, implementing quantum algorithms, and ensuring quantum error correction. Their ability to perform operations based on the state of other qubits forms the foundation for the conditional logic necessary in quantum circuits. Without controlled gates, we would lack the ability to construct the full range of operations needed for quantum computation, effectively limiting the power and functionality of quantum computers.

## 1. Controlled-NOT (CNOT) Gate

The CNOT gate is a two-qubit gate where:

- The first qubit acts as the control qubit.
- The second qubit acts as the target qubit.

The CNOT gate flips (performs an $X$ operation on) the target qubit if the control qubit is in the state $\vert 1 \rangle$. Mathematically, the action of the CNOT gate on the computational basis states can be summarized as follows:

$$
\text{CNOT}(\vert 00 \rangle) = \vert 00 \rangle, \\
\text{CNOT}(\vert 01 \rangle) = \vert 01 \rangle, \\
\text{CNOT}(\vert 10 \rangle) = \vert 11 \rangle, \\
\text{CNOT}(\vert 11 \rangle) = \vert 10 \rangle.
$$

### Matrix Representation of the CNOT Gate

In the computational basis $\{\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle\}$, the matrix representation of the CNOT gate is:

$$
\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \end{pmatrix}.
$$

### Example of CNOT Operation

Suppose we have a two-qubit state $\vert \psi \rangle = \vert 10 \rangle$. Applying the CNOT gate on this state results in:

$$
\text{CNOT}(\vert 10 \rangle) = \vert 11 \rangle,
$$

because the control qubit is in state $\vert 1 \rangle$, so the target qubit is flipped from $\vert 0 \rangle$ to $\vert 1 \rangle$.

## 2. General Controlled-U Gate

A controlled-U gate is a generalization of the CNOT gate, where the target qubit undergoes an arbitrary unitary operation $U$ if the control qubit is in the state $\vert 1 \rangle$. The controlled-U gate can be represented by a 4x4 matrix that depends on the 2x2 matrix $U$ acting on the target qubit.

The matrix representation of the controlled-U gate is:

$$
\text{Controlled-}U = \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & u_{11} & u_{12} \\
0 & 0 & u_{21} & u_{22} \end{pmatrix},
$$

where

$$
U = \begin{pmatrix} u_{11} & u_{12} \\
u_{21} & u_{22} \end{pmatrix}
$$

is the unitary matrix acting on the target qubit.

### Example: Controlled-Z Gate

The controlled-Z (CZ) gate is a controlled-U gate where the unitary matrix $U$ is the Pauli-Z gate:

$$
Z = \begin{pmatrix} 1 & 0 \\
0 & -1 \end{pmatrix}.
$$

The controlled-Z gate applies a phase flip (multiplies by $-1$) to the target qubit if the control qubit is $\vert 1 \rangle$. Its matrix representation is:

$$
\text{Controlled-}Z = \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & -1 \end{pmatrix}.
$$

This gate is commonly used to create entanglement between qubits, as it performs a non-trivial operation only when both qubits are in the $\vert 11 \rangle$ state.

## 3. Multi-Controlled Gates

Multi-controlled gates extend the concept of a controlled gate to multiple control qubits. A multi-controlled gate applies an operation to the target qubit only when all control qubits meet a specified condition (usually all being $\vert 1 \rangle$).

### Toffoli (CCNOT) Gate

The Toffoli gate, or CCNOT gate, is a three-qubit gate with two control qubits and one target qubit. It performs an $X$ (NOT) operation on the target qubit only if both control qubits are in the state $\vert 1 \rangle$. The Toffoli gate is essential in quantum computing as a universal gate, meaning it can implement any reversible Boolean function.

Its matrix representation in the computational basis $\{\vert 000 \rangle, \vert 001 \rangle, \dots, \vert 111 \rangle\}$ is an 8x8 matrix:

$$
\text{Toffoli} = \begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \end{pmatrix}.
$$

### Example of Toffoli Gate

Consider a three-qubit state $\vert 110 \rangle$. Applying the Toffoli gate will result in:

$$
\text{Toffoli}(\vert 110 \rangle) = \vert 111 \rangle,
$$

because both control qubits are in the state $\vert 1 \rangle$, so the NOT operation is applied to the target qubit, flipping it from $\vert 0 \rangle$ to $\vert 1 \rangle$.

# Universality in Quantum Computing

In quantum computing, the concept of universality is essential, as it allows us to construct any quantum operation on multi-qubit systems using a small set of fundamental gates. For multi-qubit systems, gate universality means that any quantum computation can be decomposed into sequences of single-qubit gates and two-qubit controlled gates. Here, we’ll provide the mathematical description of multi-qubit systems and how universality is achieved through specific gate sets.

## 1. Mathematical Description of Multi-Qubit States

A single qubit exists in a two-dimensional complex vector space $C^2$, with basis states $\vert 0 \rangle$ and $\vert 1 \rangle$. For a system of $n$ qubits, the state space is the tensor product of $n$ single-qubit spaces, which is a $2^n$-dimensional complex vector space:

$$
H^{(n)} = \underbrace{C^2 \otimes C^2 \otimes \dots \otimes C^2}_{n \text{ times}} = C^{2^n}.
$$

### Example: Two-Qubit and Three-Qubit Systems

For a two-qubit system, the basis states are $\vert 00 \rangle$, $\vert 01 \rangle$, $\vert 10 \rangle$, and $\vert 11 \rangle$, which form a basis for the four-dimensional vector space $C^4$. An arbitrary two-qubit state $\vert \psi \rangle$ can be written as:

$$
\vert \psi \rangle = \alpha_{00} \vert 00 \rangle + \alpha_{01} \vert 01 \rangle + \alpha_{10} \vert 10 \rangle + \alpha_{11} \vert 11 \rangle,
$$

where $\alpha_{ij} \in C$ and satisfy the normalization condition:

$$
\vert \alpha_{00} \vert^2 + \vert \alpha_{01} \vert^2 + \vert \alpha_{10} \vert^2 + \vert \alpha_{11} \vert^2 = 1.
$$

For a three-qubit system, the basis states are $\vert 000 \rangle$, $\vert 001 \rangle$, …, $\vert 111 \rangle$, forming an eight-dimensional vector space $C^8$. An arbitrary three-qubit state is:

$$
\vert \psi \rangle = \sum_{i,j,k \in \{0,1\}} \alpha_{ijk} \vert ijk \rangle,
$$

where the coefficients $\alpha_{ijk} \in C$ satisfy the normalization condition:

$$
\sum_{i,j,k} \vert \alpha_{ijk} \vert^2 = 1.
$$

## 2. Quantum Gates and Unitaries in Multi-Qubit Systems

### Single-Qubit Gates

Single-qubit gates are represented by $2 \times 2$ unitary matrices. Any single-qubit gate $U$ acting on a qubit in state $\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle$ is represented as:

$$
U \vert \psi \rangle = \begin{pmatrix} u_{11} & u_{12} \\
u_{21} & u_{22} \end{pmatrix} \begin{pmatrix} \alpha \\
\beta \end{pmatrix} = \begin{pmatrix} u_{11} \alpha + u_{12} \beta \\
u_{21} \alpha + u_{22} \beta \end{pmatrix}.
$$

Examples of single-qubit gates include the Pauli gates $X$, $Y$, and $Z$, as well as the Hadamard gate $H$.

### Two-Qubit Gates (Controlled Gates)

A two-qubit gate acts on a two-qubit system, represented by a $4 \times 4$ matrix. A crucial two-qubit gate is the Controlled-NOT (CNOT) gate, which performs a NOT operation on the target qubit if the control qubit is in the state $\vert 1 \rangle$. In the computational basis $\{\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle\}$, the CNOT gate is represented as:

$$
\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \end{pmatrix}.
$$

The action of the CNOT gate on a two-qubit state $\vert \psi \rangle = \alpha_{00} \vert 00 \rangle + \alpha_{01} \vert 01 \rangle + \alpha_{10} \vert 10 \rangle + \alpha_{11} \vert 11 \rangle$ can be derived by matrix multiplication with the state vector.

## 3. Gate Universality in Quantum Computing

Universality in quantum computing refers to the ability to approximate any arbitrary unitary operation on $n$-qubit states using a finite set of basic gates. A set of gates is universal if any unitary transformation $U \in C^{2^n \times 2^n}$ can be approximated to arbitrary precision using a sequence of gates from this set.

### Universal Gate Set: Single-Qubit and Two-Qubit Gates

It is a foundational result in quantum computing that any unitary operation on $n$ qubits can be constructed from:

- Single-qubit gates (e.g., Hadamard $H$, phase $S$, and T gates),
- Two-qubit entangling gates, such as the CNOT gate.

This universality theorem implies that any complex quantum circuit can be decomposed into a sequence of single-qubit rotations and two-qubit CNOT gates.

### Mathematical Statement of Universality

For any unitary operator $U$ on $n$-qubits (i.e., a $2^n \times 2^n$ unitary matrix), there exists a finite sequence of gates $G_1, G_2, \dots, G_k$ from the set of single-qubit and two-qubit CNOT gates such that:

$$
U \approx G_k G_{k-1} \dots G_2 G_1,
$$

where the approximation can be made arbitrarily close by choosing a sufficient number of gates.

### Example of Universal Gate Decomposition

To see universality in practice, consider a general single-qubit rotation gate $R(\theta, \phi, \lambda)$, which can be decomposed as:

$$
R(\theta, \phi, \lambda) = e^{i \alpha} R_z(\phi) R_y(\theta) R_z(\lambda),
$$

where:

$$
R_y(\theta) = e^{-i \theta \sigma_y / 2} \quad \text{and} \quad R_z(\phi) = e^{-i \phi \sigma_z / 2}
$$

are rotations around the $y$- and $z$-axes, respectively. For an arbitrary two-qubit gate, the decomposition may involve a sequence of CNOT gates interleaved with single-qubit rotations on each qubit.

## 4. Examples of Universality in Quantum Circuits

### Example 1: Constructing a Quantum Circuit for the CNOT Gate

To show universality, consider the implementation of the CNOT gate in terms of the universal set of single-qubit rotations and two-qubit gates. Since CNOT itself is a part of the universal gate set, any circuit requiring conditional operations between qubits can be constructed using a combination of CNOT gates and single-qubit gates.

### Example 2: Building a Multi-Qubit Entangled State

Suppose we want to create a two-qubit entangled state, specifically the Bell state $\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 11 \rangle)$. This can be constructed using the following steps:

1. **Apply the Hadamard gate $H$ to the first qubit:**

$$
   H \vert 0 \rangle = \frac{1}{\sqrt{2}} (\vert 0 \rangle + \vert 1 \rangle).
  $$

   After this step, the state becomes:

$$
   \vert \psi \rangle = \frac{1}{\sqrt{2}} (\vert 0 \rangle + \vert 1 \rangle) \otimes \vert 0 \rangle = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 10 \rangle).
  $$

2. **Apply a CNOT gate with the first qubit as the control and the second qubit as the target:**

$$
   \text{CNOT} \left( \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 10 \rangle) \right) = \frac{1}{\sqrt{2}} (\vert 00 \rangle + \vert 11 \rangle).
  $$

Thus, using only a single-qubit gate (Hadamard) and a two-qubit gate (CNOT), we’ve created an entangled Bell state, demonstrating the principle of universality.


