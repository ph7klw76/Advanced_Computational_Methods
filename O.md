# Big-O 

notation is the language we use to talk about how painful your simulation will become as you scale it up.

Let’s dig into what “O” actually means, connect it to real physics codes, and then systematically see how you can improve it with concrete examples.

## 1. What Big-O Actually Means (Carefully, Not Hand-Wavy)

Suppose you have some algorithm with runtime $T(N)$ as a function of problem size $N$. Big-O notation says:

$$
T(N) = O(f(N))
$$

if there exist constants $C > 0$ and $N_0$ such that

$$
T(N) \leq C\,f(N) \quad \text{for all } N \geq N_0.
$$

### Important points:

#### Asymptotic behavior

We care about what happens when $N$ is large.

Small-$N$ behavior and constant factors are ignored.

#### Upper bound, not exact equality

Saying $T(N) = O(N^2)$ does not mean $T(N) = kN^2$.

It means: “for large enough $N$, it grows no faster than some constant times $N^2$.”

#### We drop lower-order terms

If:

$$
T(N) = 3N^2 + 5N + 7
$$

then for large $N$, the $3N^2$ term dominates. We say:

$$
T(N) = O(N^2)
$$

even though technically it’s $3N^2 + 5N + 7$.

#### Why constants don’t matter asymptotically

If algorithm A: $T_A(N) = 10^{-6}N^2$, algorithm B: $T_B(N) = 10^{-3}N\log N$.

For small $N$, A might be faster (smaller constant). For sufficiently large $N$, $N^2$ will dominate $N\log N$, and B wins asymptotically.

So Big-O is the shape of growth as you scale up.

## 2. What Is “N” in Physics Codes?

In computational physics, $N$ is not abstract; it’s tied directly to physical choices:

- **$N =$ number of particles**  
  Molecular dynamics, N-body gravity, Coulomb systems.

- **$N =$ number of grid points**  
  Finite-difference / finite-element PDE solvers.

- **$N =$ number of basis functions / orbitals**  
  Electronic structure (DFT, Hartree–Fock, CI).

- **$N =$ dimension of the Hilbert space**  
  Exact diagonalization of spin systems, Fock space.

- **$N =$ number of Monte Carlo samples**  
  Path-integral Monte Carlo, Metropolis sampling.

Changing $N$ means:

- Finer resolution,  
- Larger physical systems,  
- More accurate statistics,  
- More realistic models.

And that’s exactly what researchers want. This is why Big-O matters so much.

## 3. Typical Complexity Classes and Physical Intuition

Let’s list common scalings and attach physics meanings:

### 3.1 Constant – $O(1)$

Runtime independent of $N$. Example:

- Compute the force on one particle, given its fixed neighbor list.
- Evaluate a local observable at a fixed site.

### 3.2 Linear – $O(N)$

Cost grows proportionally with system size. Examples:

- Loop over N particles once and update positions.

```python
for i in range(N):
    x[i] += v[i] * dt
```
One Monte Carlo “sweep” visiting each spin once.

### 3.3 Near-Linear – $O(N\log N)$

Common when there is:

- Divide-and-conquer recursion (like FFT),
- Hierarchical spatial structures (like trees).

Examples:

- FFT on N grid points.
- Tree-based N-body simulations (Barnes–Hut).

### 3.4 Quadratic – $O(N^2)$

Every degree of freedom interacts with every other. Examples:

- All-pairs particle interactions:  
  for all $(i,j), i \ne j$
- Computing all pairwise distances in a set of N particles.

### 3.5 Cubic – $O(N^3)$

Typical of dense linear algebra:

- Diagonalizing an $N \times N$ dense matrix.
- LU decomposition or matrix inversion.

In electronic structure:

If the basis size grows with system size, you get $O(N^3)$ scaling w.r.t. number of electrons / atoms.

### 3.6 Exponential / Factorial – $O(2^N), O(N!)$

Catastrophic scaling: only small N is possible. Examples:

- Exact diagonalization of many-body spin systems where Hilbert space dimension scales like $2^N$.
- Full configuration interaction (FCI): number of Slater determinants grows combinatorially.

---

## 4. Example 1 – N-Body Gravity / Coulomb: From $O(N^2)$ to $O(N)$–$O(N \log N)$

Consider N particles each exerting gravitational or Coulomb force on all others.

### 4.1 Physics Setup

For gravity:

$$
F_i = Gm_i \sum_{j \ne i} \frac{m_j (r_j - r_i)}{|r_j - r_i|^3}
$$

Similar form for Coulomb forces.

### 4.2 Naive All-Pairs Algorithm – $O(N^2)$

Pseudocode:
```python
for i in range(N):
    F[i] = 0
    for j in range(N):
        if i != j:
            F[i] += force(p[i], p[j])
```
Inner loop: N – 1 force evaluations per $i$.

Outer loop: N values of $i$.

Total operations ~ $N(N−1) ≈ N^2$.

If $N$ doubles:

Cost goes from $CN^2$ to $C(2N)^2 = 4CN^2$.

Runtime ~ 4× longer.

For N ~ $10^6$, $N^2 \sim 10^{12}$ pair interactions per step – easily impossible.

### 4.3 Barnes–Hut Tree – $O(N \log N)$

Key physical/algorithmic idea:

Far-away particles look almost like a single clump (multipole expansion).

So you hierarchically cluster particles in space using a tree.

Algorithm sketch:

**Build an octree (3D):**

- Start with a cube containing all particles.
- Recursively subdivide each cell into 8 subcells until cells have ≲1 particle.
- Each node stores total mass/charge and center of mass.

**Compute force on each particle:**

- Traverse the tree.
- For a given node (cell):

  If it is sufficiently far away such that

  $$
  \theta = \frac{s}{d} \ll 1
  $$

  where $s$ = cell size, $d$ = distance from particle to cell center, then treat the entire cell as a single “superparticle”.

  Otherwise, recurse into children for finer detail.

**Complexity:**

- Tree depth ~ $\log N$ (approximately balanced).
- For each particle, we visit $O(\log N)$ nodes (approx).
- Total cost ~ $O(N \log N)$.

**Error control:**

- Tolerance parameter $\theta$ controls approximation error.
- Smaller $\theta$ → more accurate but more expensive (larger constant in $O(N \log N)$).

### 4.4 Fast Multipole Method (FMM) – $O(N)$

FMM goes further:

Uses a more sophisticated combination

- multipole expansions for far fields,
- local expansions for near fields,
- careful grouping of interactions between cell pairs.

Under certain conditions, the total number of operations scales ~ linear in N.

So you can move from:

$$
O(N^2) \rightarrow O(N \log N) \rightarrow O(N),
$$

by using more physics-aware and math-aware algorithms.

### 5. Example 5 – Sparse vs Dense Linear Algebra in PDE Solvers

Consider discretizing the Laplacian $\nabla^2$ on a grid to solve Poisson’s equation:

$$
\nabla^2 u(r) = f(r)
$$

#### 5.1 Dense Matrix – $O(N^3)$ and $O(N^2)$ Memory

If you foolishly store the operator as a full dense matrix A of size $N \times N$, where:

N = number of grid points,

then:

- Memory: store $N^2$ entries → $O(N^2)$.
- Solving $Ax = b$ by direct methods: $O(N^3)$.

This quickly becomes impossible.

#### 5.2 Physical Insight: Local Operator ⇒ Sparse Matrix

The finite difference Laplacian in 3D typically couples only a small stencil:

A grid point is coupled to itself and its six nearest neighbors in 3D (7-point stencil) or some similar local pattern.

This means each row of A has ≈ constant number of nonzero entries.

So:

- Number of nonzero entries, nnz ~ constant × N.
- A is sparse.

#### 5.3 Iterative Methods – $O(N \times \text{iterations})$

Use iterative solvers like:

- Conjugate Gradient (CG) for symmetric positive definite problems,
- GMRES / BiCGSTAB for more general ones.

**Key facts:**

- Each matrix-vector multiply costs ~$O(\text{nnz}) \sim O(N)$.
- Per iteration cost: $O(N)$.
- Total cost:
$O(N \times \text{number\_of\_iterations})$.

**Preconditioning and multigrid:**

Good preconditioners and multigrid methods can make iteration count ~$O(1)$ or $O(\log N)$ for elliptic problems.

**Overall complexity:** ~$O(N)$ or $O(N \log N)$.

> This is a classic example:  
> Same PDE,  
> Same physics,  
> But dense vs sparse assumptions change complexity by orders of magnitude.

---

### 6. Example 6 – Monte Carlo: Cost per Independent Sample

In Monte Carlo, we must distinguish:

- Cost per update (sweep),
- Cost per independent sample (decorrelated configuration).

#### 6.1 Single-Spin Flip Metropolis in the Ising Model

For a lattice of N spins:

- One Monte Carlo sweep:  
  Attempt to flip each spin once.  
  Cost per sweep: $O(N)$.

However:

Near the critical temperature, correlation length $\xi$ diverges.

Autocorrelation time $\tau$ grows as:

$$
\tau \sim L^z,
$$

where $L$ is linear system size and $z$ is dynamic critical exponent.

Number of sweeps to get an independent configuration ~ $\tau$.  
So effective cost per independent configuration:

$$
\text{Cost} \sim O(N \times \tau) \sim O(L^d \times L^z) = O(L^{d+z}),
$$

with $d$ = spatial dimension.

This can be a high power of L: effectively bad scaling.

#### 6.2 Cluster Algorithms (Wolff, Swendsen–Wang)

**Cluster algorithms:**

Identify clusters of aligned spins via Fortuin–Kasteleyn mapping, then flip whole clusters.

**Effect:**

Dynamic exponent $z$ is much smaller; critical slowing down is greatly reduced.

Effective cost per independent configuration scales much better.

So even though:

- One cluster update may be comparable or slightly more expensive than one Metropolis sweep,

- The number of updates to achieve independence is much smaller.

**In Big-O terms:**

You’ve improved the effective scaling of getting physically relevant data, not just raw updates.

