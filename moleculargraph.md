## 1) Core Idea: The “Molecular Graph”

- **Vertices (nodes)** = atoms  
  - Often **heavy atoms only** (hydrogens can be omitted to simplify).
  
- **Edges** = chemical bonds  
  - Typically **undirected** and **simple** (no self-loops).
  - **Bond multiplicity** (single/double/triple) can be stored as an **edge weight** or **label**.
  - **Atom types** and **charges** are stored as **vertex labels**.

### Two Common Choices:

- **Hydrogen-suppressed graph**  
  - Vertices = non-hydrogen atoms  
  - A vertex label stores how many hydrogens are attached.

- **Explicit-hydrogen graph**  
  - Include hydrogen atoms as vertices  
  - Useful for **stereochemistry** and **counting hydrogen bonds**

## 2) Matrices that Represent the Graph

These are exact, checkable objects that let you compute properties.

### Adjacency matrix $A$:  
$A_{ij} = 1$ if atoms $i$ and $j$ are bonded, else 0 (or the bond order as a weight).

### Degree matrix $D$:  
Diagonal with $D_{ii} = $ number of bonds (or sum of bond orders) at atom $i$.

### Graph Laplacian:  

$$
L = D - A
$$  
(combinatorial) or normalized Laplacian:  
$$
D^{-1/2} L D^{-1/2}
$$

---

### Example (benzene ring, hydrogen-suppressed, order the carbons around the ring):

- $A$ is the adjacency of a 6-cycle: ones on  
  $(1,2), (2,3), \dots, (6,1)$ and symmetric.
- Every vertex degree is 2, so:
  
$$
D = 2I
$$  

and

$$
L = 2I - A
$$

## Details example of the matrices

## A) Ethanol (hydrogen-suppressed): C1—C2—O3

Order the atoms as `[C1,C2,O3]`.

### Adjacency $A$  

$$
A =
\begin{bmatrix}
0 & 1 & 0 \\\\
1 & 0 & 1 \\\\
0 & 1 & 0
\end{bmatrix}
$$

- $A_{12} = A_{21} = 1$ (C1–C2)  
- $A_{23} = A_{32} = 1$ (C2–O3)

### Degree $D$

Degrees are $(1,2,1)$, so  

$$
D = \mathrm{diag}(1,2,1)
$$

### (Combinatorial) Laplacian $L = D - A$  

$$
L =
\begin{bmatrix}
1 & -1 & 0 \\\\
-1 & 2 & -1 \\\\
0 & -1 & 1
\end{bmatrix}
$$

---

### Sanity checks (always true for connected simple graphs):

- Rows sum to 0 ⇒ $L \mathbf{1} = 0$
- $L$ is symmetric positive semidefinite; the smallest eigenvalue is 0.

---

### Normalized Laplacian  

$$
L = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
$$

Here,  
$$
D^{-1/2} = \mathrm{diag}(1, 1/ \sqrt{2}, 1)
$$

Thus,  
$$
D^{-1/2} A D^{-1/2} =
\begin{bmatrix}
0 & \frac{1}{\sqrt{2}} & 0 \\\\
\frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\\\
0 & \frac{1}{\sqrt{2}} & 0
\end{bmatrix}, \quad
L =
\begin{bmatrix}
1 & -\frac{1}{\sqrt{2}} & 0 \\\\
-\frac{1}{\sqrt{2}} & 1 & -\frac{1}{\sqrt{2}} \\\\
0 & -\frac{1}{\sqrt{2}} & 1
\end{bmatrix}
$$

---

### Eigenfacts you can verify

- $\mathrm{spec}(A) = \{2, 0, -2\}$
- $\mathrm{spec}(L) = \{0, 2, 2\}$ because $L = 2I - A$ does **not** hold here (degrees differ), but direct calculation gives $\{0, 2 - \sqrt{2}, 2 + \sqrt{2}\}$ for $L$?  
Careful: for a 3-node path,
- $\mathrm{spec}(A) = \{2, 0, -2\}$
- $\mathrm{spec}(L) = \{0, 2 - \sqrt{2}, 2 + \sqrt{2}\}$

---

### $\mathrm{spec}(L) = \{0, 1, 2\}$ for a path of length 3?

More precisely, for this 3-node path the **normalized Laplacian** eigenvalues are  

$$
\mathrm{spec}(L) = \{0, 1, 2\}
$$

(These are standard for a 3-node path; you can confirm with `np.linalg.eigvalsh`.)

---

## What They Tell You

- One zero eigenvalue of $L$ or $\mathcal{L}$ ⇒ the graph has exactly one connected component (the molecule is connected).
- Nonzero eigenvalues capture “stiffness”/connectivity; larger values indicate tighter “coupling” around C2.

---

## B) Benzene (C₆ ring), unweighted

Order the carbons cyclically `[1,2,3,4,5,6]`.

### Adjacency $A$  

$$
A =
\begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 1 \\\\
1 & 0 & 1 & 0 & 0 & 0 \\\\
0 & 1 & 0 & 1 & 0 & 0 \\\\
0 & 0 & 1 & 0 & 1 & 0 \\\\
0 & 0 & 0 & 1 & 0 & 1 \\\\
1 & 0 & 0 & 0 & 1 & 0
\end{bmatrix}
$$

### Degree $D$  
Every vertex has degree 2 ⇒  

$$
D = 2 I_6
$$

### Laplacian $L = D - A = 2I_6 - A$  

$$
L =
\begin{bmatrix}
2 & -1 &  0 &  0 &  0 & -1 \\\\
-1 & 2 & -1 &  0 &  0 &  0 \\\\
0 & -1 & 2 & -1 &  0 &  0 \\\\
0 &  0 & -1 & 2 & -1 &  0 \\\\
0 &  0 &  0 & -1 & 2 & -1 \\\\
-1 & 0 &  0 &  0 & -1 & 2
\end{bmatrix}
$$

### Normalized Laplacian  

$$
L = I - D^{-1/2} A D^{-1/2}
$$

Since $D = 2I$,  
$$
D^{-1/2} = \frac{1}{\sqrt{2}} I, \quad \Rightarrow \quad L = I - \frac{1}{2} A
$$

---

### Eigenfacts (classic, and easy to check)

- $\mathrm{spec}(A) = \{2, 1, 1, -1, -1, -2\}$
- $\mathrm{spec}(L) = 2 - \mathrm{spec}(A) = \{0, 1, 1, 3, 3, 4\}$
- $\mathrm{spec}(\mathcal{L}) = 1 - \frac{1}{2} \mathrm{spec}(A) = \{0, \frac{1}{2}, \frac{1}{2}, \frac{3}{2}, \frac{3}{2}, 2\}$

---

### What They Tell You

- Exactly one zero eigenvalue ⇒ the ring is connected.
- The two repeated small nonzero eigenvalues ($1$ in $L$, or $\frac{1}{2}$ in $\mathcal{L}$) reflect the 6-cycle’s symmetry.
- Large eigenvalues (e.g., 4 or 2) correspond to “high-frequency” patterns on the ring—important in diffusion/spectral partitioning analogies.

---

## C) Benzene with Bond-Order Weights (e.g., set each C–C to 1.5)

Let every edge weight be $w = 1.5$. Then:  

$$
A_w = 1.5 A
$$

Each vertex degree is the sum of incident bond orders:  

$$
d_i = 1.5 + 1.5 = 3 \quad \Rightarrow \quad D_w = 3 I_6
$$

Then,  

$$
L_w = D_w - A_w = 3 I_6 - 1.5 A
$$

Normalized version:  

$$
L_w = D_w^{-1/2} L_w D_w^{-1/2} = I - \frac{1}{3} A_w = I - \frac{1}{3}(1.5 A) = I - \frac{1}{2} A
$$

---

### Consequences (useful rule of thumb):

- $\mathrm{spec}(A_w) = 1.5 \, \mathrm{spec}(A) = \{3, 1.5, 1.5, -1.5, -1.5, -3\}$
- $\mathrm{spec}(L_w) = 3 - \mathrm{spec}(A_w) = \{0, 1.5, 1.5, 4.5, 4.5, 6\}$

Normalized Laplacian spectrum is unchanged from the unweighted case because the graph is still regular and all edges scaled equally:  

$$
\mathrm{spec}(\mathcal{L}_w) = \{0, \frac{1}{2}, \frac{1}{2}, \frac{3}{2}, \frac{3}{2}, 2\}
$$

This illustrates a general, verifiable fact: for a $r$-regular graph, scaling all edge weights by a constant keeps $\mathcal{L}$’s eigenvalues the same.

---

## D) What These Matrices Buy You

- **Row-sum zero & PSD**: $L$ and $\mathcal{L}$ are symmetric, positive semidefinite, and have row sums 0.  
  The multiplicity of eigenvalue 0 equals the number of connected components.

- **Distances/flows**: The Moore-Penrose pseudoinverse $L^+$ gives effective resistance distances $R_{ij}$  
  (resistor network analogy), which are useful topological descriptors.

- **Regular graphs shortcut**: For $r$-regular graphs (like unweighted benzene):

$$
L = rI - A, \quad \mathcal{L} = \frac{1}{r} L = I - \frac{1}{r} A
$$

So eigenvalues transform linearly:  

$$
\lambda(L) = r - \lambda(A), \quad \lambda(\mathcal{L}) = 1 - \frac{\lambda(A)}{r}
$$


