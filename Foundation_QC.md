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

For a pure two-qubit state $\vert \Psi \rangle$, the concurrence $C$ is given by:

$$
C = \vert \langle \Psi \vert \tilde{\Psi} \rangle \vert,
$$

where $\vert \tilde{\Psi} \rangle = (\sigma_y \otimes \sigma_y) \vert \Psi^* \rangle$, with $\vert \Psi^* \rangle$ being the complex conjugate of $\vert \Psi \rangle$ in the standard basis. For maximally entangled states (e.g., Bell states), $C = 1$, while for separable states, $C = 0$.

## Entropy of Entanglement

The entropy of entanglement $S$ is another measure, defined as the von Neumann entropy of the reduced density matrix $\rho_A$ (or $\rho_B$):

$$
S(\rho_A) = - \text{Tr}(\rho_A \log_2 \rho_A).
$$

For a maximally entangled state, such as a Bell state, $S(\rho_A) = 1$, representing maximum entanglement. For a separable state, $S(\rho_A) = 0$, as there is no entanglement.

# 5. Bell Test and Non-Local Correlations

One of the most profound implications of entanglement is the existence of non-local correlations that defy classical intuition, as demonstrated by Bell’s theorem. In an entangled two-qubit system, measurements on qubits $A$ and $B$ exhibit correlations that cannot be explained by classical mechanics or local hidden variables.

Bell's theorem shows that entangled states violate Bell inequalities, suggesting that quantum mechanics allows for correlations stronger than any classical system. Bell states are especially important because they maximize these correlations.

### Example: Violation of Bell Inequalities

For a Bell state like $\vert \Phi^+ \rangle$, measuring specific observables on each qubit in different directions produces correlations that exceed the bounds set by classical probability. This Bell inequality violation is direct evidence of entanglement, emphasizing the unique properties of two-qubit states in quantum mechanics.

