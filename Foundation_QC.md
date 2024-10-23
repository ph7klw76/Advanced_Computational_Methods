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

### Final Results:
- First inner product: \(\langle \psi_1 | \psi_2 \rangle = \frac{1}{2}\)
- Second inner product: \(\langle \psi_1 | \psi_3 \rangle = 0\)




![image](https://github.com/user-attachments/assets/c92f934b-1312-42db-b28f-03d793dc17dd)


