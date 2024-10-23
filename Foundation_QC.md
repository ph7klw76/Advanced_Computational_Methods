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

To check this, we need to compute \( X^{\dagger} \) and compare it to \( X^{-1} \).

### Conjugate Transpose \( X^{\dagger} \)

Since \( X \) is real, its conjugate transpose is the same as its transpose \( X^{T} \):

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

### Inverse of \( X \)

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

Since \( X^{-1} = X^{\dagger} \), the Pauli-X gate is indeed unitary.

## 2. What is the Inverse of the X Gate?

As we saw above, the inverse of \( X \) is itself:

$$
X^{-1} = X = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
$$

## 3. Action of the X Gate on a General Qubit

To find the action of the \( X \) gate on a general qubit, we apply the matrix \( X \) to the state \( a\lvert 0 \rangle + b\lvert 1 \rangle \).

Let the general qubit state be:

$$
\lvert \psi \rangle = a \lvert 0 \rangle + b \lvert 1 \rangle = a \begin{pmatrix} 1 \\ 0 \end{pmatrix} + b \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} a \\ b \end{pmatrix}
$$

Now, apply the \( X \) matrix to this state:

$$
X \lvert \psi \rangle = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} b \\ a \end{pmatrix}
$$

Thus, the action of \( X \) on the qubit \( a\lvert 0 \rangle + b\lvert 1 \rangle \) is to swap the amplitudes \( a \) and \( b \):

$$
X \left( a \lvert 0 \rangle + b \lvert 1 \rangle \right) = b \lvert 0 \rangle + a \lvert 1 \rangle
$$
