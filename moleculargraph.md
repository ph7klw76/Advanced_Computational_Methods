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

## Demonstrating: “Edges = Atoms That Are Close Enough”

This demonstrates how `"edges = atoms that are close enough"` works, using `radius_graph` (and a pure-NumPy fallback if PyG isn’t installed). It:

- Defines 3 atoms (C, C, O) with 3D positions  
- Builds edges with a distance cutoff (in Ångströms)  
- Prints the sparse `edge_index` and a dense adjacency matrix you can inspect  
- Checks there are **no self-loops** and that the graph is **undirected** (symmetric)

---

### What This Illustrates (and How It Maps to the Math)

- `edge_index` is the **sparse COO representation** of the adjacency matrix:  
  Each column is an ordered pair $(i, j)$  
  If you see both $(0,1)$ and $(1,0)$, that means:

$$
A_{01} = A_{10} = 1
$$

- `loop=False` ⇒ no self-loops, so:
  
$$
A_{ii} = 0
$$

---

```python
# --- Edges via radius graph: connect atoms whose distance <= cutoff ---

# Try PyTorch Geometric; if unavailable, use a NumPy fallback.
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import radius_graph
    HAVE_PYG = True
except Exception:
    HAVE_PYG = False

# ---- Toy molecule: ethanol heavy atoms (C1, C2, O3) ----
# Coordinates in angstroms (Å), rows = atoms, cols = x,y,z
pos_np = np.array([
    [-0.748,  0.000, 0.000],   # C1
    [ 0.748,  0.000, 0.000],   # C2
    [ 1.520,  1.210, 0.000],   # O3
], dtype=float)

# --- helper: make dense adjacency from edge_index ---
def dense_adjacency(edge_index, n):
    A = np.zeros((n, n), dtype=int)
    for i, j in edge_index.T:
        A[i, j] = 1
    return A

# --- fallback radius_graph (NumPy), same semantics as PyG's: directed pairs, no self-loops ---
def numpy_radius_graph(pos, r, loop=False):
    n = pos.shape[0]
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if not loop and i == j:
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d <= r:
                rows.append(i); cols.append(j)
    return np.vstack([np.array(rows, int), np.array(cols, int)])  # shape [2, E]

def build_edges(pos_np, cutoff):
    n = pos_np.shape[0]
    if HAVE_PYG:
        pos = torch.tensor(pos_np, dtype=torch.float)
        ei = radius_graph(pos=pos, r=cutoff, loop=False)  # shape [2, E]
        edge_index = ei.numpy()
    else:
        edge_index = numpy_radius_graph(pos_np, r=cutoff, loop=False)
    A = dense_adjacency(edge_index, n)

    # Checks (graph-theory sanity):
    # 1) No self-loops -> diagonal zero
    assert np.all(np.diag(A) == 0), "Found self-loops; expected A_ii = 0."
    # 2) Undirected (since distances are symmetric) -> A should be symmetric
    assert np.array_equal(A, A.T), "Adjacency not symmetric; expected undirected edges."

    return edge_index, A

# ---- Try two cutoffs to see edges appear/disappear ----
for cutoff in [1.2, 1.7, 2.1]:
    edge_index, A = build_edges(pos_np, cutoff)
    print(f"\nCutoff r = {cutoff} Å")
    print("edge_index (source;target) columns:")
    print(edge_index)  # pairs (i,j); both directions usually present
    print("Dense adjacency A:")
    print(A)
    print("Degrees:", A.sum(axis=1))  # number of neighbors per atom
```
### The Dense $A$ We Reconstruct Is Exactly:

$$
A_{ij} =
\begin{cases}
1, & \| \text{pos}_i - \text{pos}_j \| \leq r_\text{cutoff}, \quad i \ne j \\\\
0, & \text{otherwise}
\end{cases}
$$

---

### Interpretation of Cutoff Distance

- **Changing `cutoff` changes which pairs are considered “bonded/nearby.”**
- Smaller cutoff → fewer edges (just covalent neighbors)  
- Larger cutoff → may include nonbonded close contacts too

---

### Optional Extension

If you’d like, I can add one more line to compute:

- The degree matrix:  
  $$
  D = \mathrm{diag}(A \cdot \mathbf{1})
  $$

- The Laplacian:  
  $$
  L = D - A
  $$

to show the full pipeline.

## What “Degrees & (Implicit) Laplacian via Message Passing” Means

- **Degree of a node** = how many neighbors it has  
- In matrix form:
  
$$
D = \mathrm{diag}(d_1, \dots, d_n), \quad \text{with } d_i = \sum_j A_{ij}
$$

---

### Message Passing Layers

A **message passing layer** aggregates neighbor features and mixes them with the node’s own feature. Two common aggregations are:

- **Sum of neighbors**:
   
$$
x' = A x
$$

- **Mean of neighbors (degree-normalized)**:
  
$$
x' = D^{-1} A x
$$

---

### Deeper Insight

Stack several such layers and you effectively apply a **polynomial in** $A$ or $D^{-1}A$ —  
i.e., **repeated smoothing/diffusion across the graph** (the **spectral view**).

When you also keep some of the original signal (a **residual/skip**), you get something like:

$$
x' = (1 - \alpha) x + \alpha D^{-1} A x
$$

This is a **discrete diffusion step**, closely related to the (normalized) Laplacian:

$$
L = I - D^{-1/2} A D^{-1/2}
$$

---

### So What’s “Laplacian-Like” Here?

- Averaging neighbors is the core of **diffusion**  
- **Diffusion is governed by the Laplacian**

So, a **message-passing layer that averages neighbors** is just the matrix $D^{-1}A$ acting on your node features $x$.  
Repeating layers ≈ repeatedly applying $D^{-1}A$ → **a diffusion/smoothing like a Laplacian step**.

---

### Summary of Key Matrices

- $A$: **Adjacency matrix** (who’s connected)  
- $D$: **Degree matrix** (how many neighbors)  

---


### Residual/Skip Version

If you keep some of the old feature (skip/residual), you get:

$$
x' = (1 - \alpha) x + \alpha D^{-1} A x = x - \alpha (I - D^{-1} A) x
$$

This is **one step of random-walk Laplacian diffusion**, where:

$$
L_{\text{rw}} = I - D^{-1} A
$$

---

### Tiny Graphs, Step-by-Step

```python
import numpy as np

def deg(A): return np.diag(A.sum(1))
def neigh_sum(A, x): return A @ x               # A x
def neigh_mean(A, x): return np.linalg.inv(deg(A)) @ (A @ x)  # D^{-1} A x
def residual(A, x, alpha=0.5): return (1-alpha)*x + alpha*neigh_mean(A, x)
def multilayer_mean(A, x, k): 
    y = x.copy()
    for _ in range(k): y = neigh_mean(A, y)
    return y
```

## Example 1: A Single Bond (2 Atoms)

Atoms 0—1.

$$
A =
\begin{bmatrix}
0 & 1 \\\\
1 & 0
\end{bmatrix},
\quad
D = \mathrm{diag}(1,1) = I
$$

Let:  

$$
x = [x_0, x_1]^T = [1, 3]^T
$$

- **Sum**:
  
$$
Ax = [3, 1]
$$
  (Each node “sees” the other.)

- **Mean**:
  
$$
D^{-1}Ax = Ax = [3, 1]
$$ 

  (Same as sum because $D = I$.)

- **After another mean layer**:
  
$[1,3] \rightarrow [3,1] \rightarrow [1,3] \rightarrow [3,1] \cdots$  
(It just swaps.)

- **With a residual (say $\alpha = 0.5$)**:
  
$$
x' = 0.5x + 0.5Ax = [2, 2]
$$

  (They meet in the middle—pure smoothing.)

- **Why**: Mean = average of neighbors; both nodes have one neighbor → they average each other.

---

## Example 2: Ethanol Heavy Atoms (3-Node Path: C1—C2—O3)

$$
A =
\begin{bmatrix}
0 & 1 & 0 \\\\
1 & 0 & 1 \\\\
0 & 1 & 0
\end{bmatrix}, \quad
D = \mathrm{diag}(1, 2, 1)
$$

Let:  

$$
x = [1, 2, 3]^T
$$

- **Sum**:
  
$$
Ax = [2, 4, 2]
$$

- **Mean**:
  
$$
D^{-1}Ax = \left[\frac{2}{1}, \frac{4}{2}, \frac{2}{1}\right] = [2, 2, 2]
$$

- One step and all three become the same (perfect smoothing on a path of length 2).

- **Two layers (mean then mean)** stays $[2, 2, 2]$

---

## Example 3: Triangle (3-Cycle)

$$
A =
\begin{bmatrix}
0 & 1 & 1 \\\\
1 & 0 & 1 \\\\
1 & 1 & 0
\end{bmatrix}, \quad
D = \mathrm{diag}(2, 2, 2) = 2I
$$

Start with a **spike**:  

$$
x = [1, 0, 0]^T
$$

- **Mean (1 step)**:
   
$$
D^{-1}Ax = \frac{1}{2}[0, 1, 1] = [0, 0.5, 0.5]
$$

- **Mean (2 steps)**:
  
  Node 1: $(0 + 0.5)/2 = 0.25$  
  Node 2: $(0 + 0.5)/2 = 0.25$  
  Node 0: $(0.5 + 0.5)/2 = 0.5$  
  ⇒ $[0.5, 0.25, 0.25]$

- **Many layers**: approaches the **uniform vector** $[1/3, 1/3, 1/3]$

- **Why**: On a regular graph, neighbor-mean is a random walk → it mixes to uniform.

---

## Example 4: Square vs Triangle (Different Smoothing Rates)

- Square (4-cycle) is **less tightly connected** than a triangle.

- Start: $x = [1, 0, 0, 0]$  
  On the square, the spike spreads **slower** than on the triangle.

- You can verify by repeating `neigh_mean` $k$ times: more layers are needed to look uniform.

- **Why**: the **eigenvalues of** $D^{-1}A$ (or the Laplacian) control the **mixing speed**.

---

## Example 5: Star Graph (A Hub + Leaves)

Node 0 connected to 1..4; leaves not connected to each other.

$$
D = \mathrm{diag}(4,1,1,1,1)
$$

Take:  

$$
x = [10, 0, 0, 0, 0]
$$

- **Mean (1 step)**:  
  - Leaves take the hub’s value: each becomes 10  
  - The hub becomes the average of leaves: 0  
  ⇒ $[10,0,0,0,0] \rightarrow [0,10,10,10,10]$

- **Residual ($\alpha = 0.5$)**:
  
$$
x' = 0.5x + 0.5D^{-1}Ax = [5,5,5,5,5]
$$  

  (Instant equalization.)

- **Takeaway**:  
  - Degree normalization matters  
  - High-degree hubs average many neighbors and can move slowly without a residual  
  - Leaves move fast

---

## Example 6: Benzene Ring (6-Cycle, All Degrees = 2)

Let:  

$$
x = [1,2,3,4,5,6]^T
$$

- **Sum**:
  
$$
Ax = [8,4,6,8,10,6]
$$

- **Mean**:  
  Divide each by 2 ⇒ $[4,2,3,4,5,3]$

- **Two layers (mean twice)**:
  
$[2.5, 3.5, 3, 4, 3.5, 4.5]$

- **Four layers**:
  
$[2.875, 3.875, 3, 4, 3.125, 4.125]$  
(Approaching uniform trend around mean value = 3.5)

---

### Residual vs Laplacian Diffusion

Take $\alpha = 0.5$

- **Residual output**:
  
$[2.5, 2, 3, 4, 5, 4.5]$

- This exactly equals one step of **random-walk Laplacian diffusion**:
  
$$
x' = x - \alpha(I - D^{-1}A)x \quad \text{with } \alpha = 0.5
$$

---

## What “Implicit Laplacian” Means in  GNN

- Your code aggregates with `scatter_mean` → that’s computing:
  
$$
D^{-1}Ax
$$  

  per layer (degree-normalized neighbor average)

- Add a **skip connection** (common in “conv” blocks):
  
$$
x' = (1 - \alpha)x + \alpha D^{-1}Ax = x - \alpha L_{\text{rw}} x
$$

- That’s exactly a **Laplacian-like diffusion step**

---

## Final Insight

Stack $L$ layers (you used `DEFAULT_NUM_CONV_LAYERS = 6`) →  
You're effectively applying a **polynomial in** $D^{-1}A$ (or in $L_{\text{rw}}$).  
This is the **spectral view of message passing**.

---

## Minimal, Verifiable Python (Run These to See the Numbers)

### (A) Ethanol Path

```python
import numpy as np
A = np.array([[0,1,0],[1,0,1],[0,1,0]], float)
D = np.diag(A.sum(1))
x = np.array([1.,2.,3.])

def mean(A,x): return np.linalg.inv(np.diag(A.sum(1))) @ (A @ x)
print("A x  =", A@x)           # [2,4,2]
print("D^-1 A x =", mean(A,x)) # [2,2,2]
```

### (B) Triangle smoothing

```python
A = np.array([[0,1,1],[1,0,1],[1,1,0.]], float)
x = np.array([1.,0.,0.])
for k in range(4):
    print(k, x)
    x = np.linalg.inv(np.diag(A.sum(1))) @ (A @ x)
# Approaches [1/3,1/3,1/3]
```

### (C) Benzene ring: residual vs Laplacian diffusion

```python
A = np.roll(np.eye(6),1,axis=1) + np.roll(np.eye(6),-1,axis=1)  # 6-cycle
D = np.diag(A.sum(1))  # 2I
x = np.array([1,2,3,4,5,6.], float)
alpha = 0.5
residual = (1-alpha)*x + alpha*np.linalg.inv(D)@(A@x)
lap_rw = x - alpha*(x - np.linalg.inv(D)@(A@x))
print(np.allclose(residual, lap_rw))  # True
print(residual)  # [2.5, 2., 3., 4., 5., 4.5]
```

## Why This Matters for Molecules

- **Local chemistry**:  
  Edges come from spatial proximity (or bonds). Message passing lets atoms “share” information across bonds/contacts.

- **Smoothing = context**:  
  Each layer mixes an atom’s feature with its neighborhood ⇒ the atom’s representation gains local chemical context (hybridization, substituent effects, ring currents, etc.).

- **Depth = reach**:  
  With $k$ layers, information travels roughly $k$ hops—capturing progressively larger structural motifs (rings, substituent patterns, chains).

## edge features from geometry

“Atoms i and j are connected” (a 0/1 in the adjacency), we attach continuous features to that edge:

- A **distance profile** via radial basis functions (RBFs), and  
- A **direction profile** via spherical harmonics (or their practical equivalents).

This lets a GNN learn **how much** and **in what orientation** atoms interact.

---

### 1) Radial Basis Features: Distances → A Soft Histogram

Think of RBFs as a smooth set of distance bins. Each edge’s distance lights up the nearby bins.

---

#### Mini Example (No Chemistry Yet)

Say we choose 5 Gaussian RBFs centered at  
`[0.8, 1.2, 1.6, 2.0, 2.4]` Å with the same width $\sigma = 0.2$ Å:

$$
\text{RBF}_k(r) = \exp\left( -\frac{(r - \mu_k)^2}{2\sigma^2} \right)
$$

- If an edge has length $r = 1.23$ Å, the 1.2 Å bin activates strongly; others less so.

- If $r = 2.05$ Å, the 2.0 Å bin is strongest, etc.

---

### Toy Code

```python
import numpy as np

def radial_basis(r, mus, sigma):
    r = float(r)
    return np.exp(-0.5 * ((r - mus) / sigma)**2)  # shape [num_centers]

mus   = np.array([0.8, 1.2, 1.6, 2.0, 2.4])  # Å
sigma = 0.2

for r in [1.00, 1.23, 1.50, 2.05]:
    print(f"r={r:.2f} Å ->", np.round(radial_basis(r, mus, sigma), 3))
```

## Intuition: The RBF Vector is a Soft Fingerprint of the Edge Length

The network can then learn, e.g.,  
“edges near 1.4 Å (aromatic C–C) should contribute differently than 1.1 Å (C–H) or 2.8 Å (nonbonded contact).”

---

### Small Chemistry-Flavored Intuition

- Short covalent bonds (e.g., ~1.2–1.6 Å) will activate low-μ RBFs.
- Hydrogen bonds (e.g., donor–acceptor ~2.6–3.2 Å) will light up higher-μ RBFs.
- The model can learn distance-dependent weights instead of a one-size-fits-all “connected”.

---

## 2) Directional Features: Unit Vector → Spherical Harmonics (or Simple Surrogates)

- **Distances alone ignore which way neighbors lie around an atom (angles)**  
  Many properties depend on orientation (e.g., sp² vs sp³, hydrogen-bond directionality, π–stacking).

We encode the edge direction  

$$
\hat{r} = (\Delta x, \Delta y, \Delta z) / \| \Delta \|
$$  

into a vector of **spherical harmonics values**.  
In practice, libraries like `e3nn.o3` do this for $l = 0, 1, 2, \dots$ (degrees of angular patterns).

---

### Meaning of $l$ Values

- $l = 0$ (monopole) ≈ a constant (isotropic)
- $l = 1$ (dipole) ≈ measures x, y, z-like patterns (direction)
- $l = 2$ (quadrupole) ≈ measures planar vs axial anisotropy

If you rotate the molecule, these features transform in a mathematically consistent way (**equivariance**),  
so the network can respect physics.
<img width="1145" height="474" alt="image" src="https://github.com/user-attachments/assets/a1d4a443-3f62-4017-a138-4ad200e12a33" />
<img width="604" height="493" alt="image" src="https://github.com/user-attachments/assets/a9de8779-f6b1-46a6-b781-7d8c6ec848a5" />
<img width="577" height="499" alt="image" src="https://github.com/user-attachments/assets/b2e41b8f-0dc3-4c7e-82a9-debca1d56da9" />
<img width="550" height="492" alt="image" src="https://github.com/user-attachments/assets/ddba2e06-dd38-4156-9ad7-59f45d2a0899" />
<img width="549" height="488" alt="image" src="https://github.com/user-attachments/assets/46c2afec-5e37-4311-b915-97a21eaa0aa9" />
<img width="554" height="486" alt="image" src="https://github.com/user-attachments/assets/8288a814-d9d2-475c-a649-5a9df07ccef8" />
<img width="587" height="483" alt="image" src="https://github.com/user-attachments/assets/4df1a9b3-8309-4133-919d-88cf396382dd" />
<img width="559" height="478" alt="image" src="https://github.com/user-attachments/assets/1c890571-e782-4a55-aa8b-59aef2a55a1a" />





```python
import numpy as np

def direction_features(p_i, p_j):
    v = np.array(p_j) - np.array(p_i)
    r = np.linalg.norm(v)
    if r == 0: 
        raise ValueError("Coincident atoms")
    u = v / r  # unit vector (ux, uy, uz)
    ux, uy, uz = u

    # l=0 (monopole)
    f0 = np.array([1.0])

    # l=1 (dipole-like)
    f1 = np.array([ux, uy, uz])

    # l=2 (quadrupole-like; simple real basis)
    f2 = np.array([
        ux*uy, ux*uz, uy*uz,           # cross terms
        ux*ux - uy*uy,                 # planar anisotropy
        3*uz*uz - 1                    # axial vs planar
    ])

    return np.concatenate([f0, f1, f2])  # shape [1 + 3 + 5 = 9]

# Example: atom i at origin, j at (1,1,0)
print(direction_features([0,0,0], [1,1,0]))
```

## Intuition:

If a neighbor sits along +z, then  
$u = (0, 0, 1)$:  
- the **$l=1$** part is high on the z-component  
- the **$l=2$** axial term $3u_z^2 - 1$ is **positive**

If neighbors are in the same plane but different directions:  
- their **$l=1$** parts differ in sign  
- the **$l=2$** parts capture **planar vs axial** patterns

In your scripts, `e3nn.o3` computes the **true spherical harmonics** and handles rotations properly.  
The simplified basis above is just to **build intuition**.

---

## 3) Combine Distance & Direction → Edge Feature Vector

For each edge $i \rightarrow j$, you typically concatenate:

- **RBF distance embedding** (say 16–64 numbers), and  
- **Directional embedding** (from spherical harmonics; number depends on chosen $l_{\max}$)

---

The GNN **message** from $i$ to $j$ is then a learned function of:

- Features at $i$ and $j$ (the nodes),  
- The **edge feature vector** (distance + direction)

---

### Mini End-to-End Edge Feature Builder

```python
def edge_features(p_i, p_j, mus, sigma):
    # distance part
    v = np.array(p_j) - np.array(p_i)
    r = float(np.linalg.norm(v))
    rbf = np.exp(-0.5 * ((r - mus) / sigma)**2)  # radial basis vector

    # direction part (simple surrogate shown above)
    vdir = direction_features(p_i, p_j)

    return np.concatenate([rbf, vdir])  # final edge feature
```
If you compute this for two edges with the same distance but different direction, you’ll get same RBF part but different direction part.  
If two edges have same direction but different distance, you’ll get different RBF parts.

---

## 4) Small, Chemistry-Like Examples

**Ethanol heavy atoms (C1—C2—O3, coordinates in Å)**



### Super-Simple Surrogate (Real-Valued) for Intuition

Without invoking special functions, you can use:

- **$l=0$ “feature”**:  
  `1` (constant)

- **$l=1$ “features”**:  
  $(\hat{r}_x, \hat{r}_y, \hat{r}_z)$ (the unit vector)

- **$l=2$ “features”**:  
  A few quadratic forms like:
  
$$
  \left(
  \hat{r}_x^2 - \hat{r}_y^2,\  
  \hat{r}_x \hat{r}_y,\  
  \hat{r}_x \hat{r}_z,\  
  \hat{r}_y \hat{r}_z,\  
  3\hat{r}_z^2 - 1
  \right)
$$

(These resemble real spherical harmonics up to constants.)

---

### Tiny Code to Build Direction Features

```text
C1: (-0.748, 0.000, 0.000)
C2: ( 0.748, 0.000, 0.000)
O3: ( 1.520, 1.210, 0.000)
```

## Distance C1–C2 ≈ 1.496 Å → RBF lights the ~1.5 Å bin.

Direction C1→C2 is mostly +x (unit vector ≈ (1,0,0)) → l=1 is strong in x.

Distance C2–O3 ≈ 1.444 Å (short) but direction is tilted toward +x and +y (unit ≈ (0.64, 0.77, 0)).  
→ RBF part similar to C1–C2 (both ~1.45–1.50 Å), direction part different (nonzero y-component).

---

### Why This Matters

Even if two edges have similar bond lengths, their geometry around C2 differs (one along x, one angled up in y).  
**Directional features let the model tell them apart**, which is important for **torsions, angles, and stereochemistry**.

---

## Benzene Neighbors

- All nearest C–C distances are similar (~1.39–1.42 Å), so **RBFs look alike**.
- But the **edge directions around a carbon** are at ~60° separations in the ring plane.  
  **Directional features encode this periodicity**—useful for recognizing:
  - **Planar conjugation**
  - **Out-of-plane contacts** (e.g., π–stacking in 3D datasets)

---

## Hydrogen Bonds (Direction-Sensitive)

- Donor–acceptor edges may be longer (~2.7–3.1 Å) → **RBFs activate at larger μ**
- Good H-bonds are **directional** (near-linear D–H···A)  
  → The **directional embedding helps the model** learn that **alignment matters**, not just distance.

---

## 5) Why Modelers Do This (Practical Benefits)

- **Richer messages**:  
  Edges aren’t all equal; a 1.2 Å bond should not weigh like a 3.0 Å contact.

- **Rotation sanity**:  
  Spherical-harmonic features rotate in a **mathematically consistent way**;  
  the network can be:
  - **Equivariant** (outputs rotate with inputs), or
  - **Invariant** (scalar predictions stay the same)

- **Generalization**:  
  Distance/direction-aware messages help capture **chemically meaningful patterns**  
  (hybridization, H-bond geometry, aromatic planes) that a **raw 0/1 adjacency can’t express**.


```python
import numpy as np

def radial_basis(r, mus, sigma):  # Gaussians
    return np.exp(-0.5 * ((r - mus) / sigma)**2)

def direction_features(p_i, p_j):
    v = np.array(p_j) - np.array(p_i)
    r = np.linalg.norm(v)
    u = v / r
    ux, uy, uz = u
    f0 = np.array([1.0])
    f1 = np.array([ux, uy, uz])
    f2 = np.array([ux*uy, ux*uz, uy*uz, ux*ux - uy*uy, 3*uz*uz - 1])
    return np.concatenate([f0, f1, f2])

def edge_feats(p_i, p_j, mus, sigma):
    r = np.linalg.norm(np.array(p_j) - np.array(p_i))
    rbf = radial_basis(r, mus, sigma)
    ang = direction_features(p_i, p_j)
    return r, rbf, ang

mus   = np.array([0.8, 1.2, 1.6, 2.0, 2.4])
sigma = 0.2

# Same distance, different directions
p0 = [0,0,0]
p1 = [1,0,0]     # along +x, distance 1
p2 = [0,1,0]     # along +y, distance 1
for pj in [p1, p2]:
    r, rbf, ang = edge_feats(p0, pj, mus, sigma)
    print("\nEdge 0->", pj, "r=", r)
    print("RBF:", np.round(rbf, 3))
    print("Dir:", np.round(ang, 3))

# Similar direction, different distance
p3 = [1.0, 0, 0]   # r=1.0
p4 = [1.6, 0, 0]   # r=1.6 (same direction)
for pj in [p3, p4]:
    r, rbf, ang = edge_feats(p0, pj, mus, sigma)
    print("\nEdge 0->", pj, "r=", r)
    print("RBF:", np.round(rbf, 3))
    print("Dir:", np.round(ang, 3))
```

For same distance (1.0 Å) but different direction (+x vs +y):  
RBFs identical, direction vectors different.

For same direction (+x) but different distances (1.0 vs 1.6 Å):  
Direction vectors identical, RBFs different (the 1.6 Å bin lights up more).

**RBFs** turn an edge length into a smooth, learnable multi-bin signal →  
  “how strong” the interaction is by distance.

**Spherical-harmonics–based direction features** turn the edge orientation into a vector  
  that behaves correctly under rotations →   “which way” the interaction points.


Together, they **upgrade a binary edge into a physics-aware edge descriptor**,  
which  GNN uses to pass **better, more realistic messages between atoms**.




