# 1) Spherical harmonics (SH): patterns on a sphere

## the big idea

Take any function that depends only on direction $\hat{r}$ (a point on a sphere $S^2$). You can decompose it into a sum of angular patterns. Those building blocks are the spherical harmonics $Y_{\ell m}(\theta, \phi)$.

$\ell = 0, 1, 2, \dots$ is the order (how many lobes / how wiggly the pattern is).

$m = -\ell, \dots, \ell$ indexes different orientations of that $\ell$-pattern.

They form an orthonormal basis on the sphere: like sine/cosine on a circle but now on $S^2$.

## plain-english definitions

- **sphere angles**: $\theta$ = polar (0 at north pole), $\phi$ = azimuth (compass angle).
- **orthonormal basis**: you can express any nice function on the sphere as a weighted sum of these; different $Y_{\ell m}$ are mutually “perpendicular” under the sphere’s integral.

## why SH are useful

They isolate pure angular content independent of distance.

In 3D geometry problems (physics, molecules, messages along edges), a direction $\hat{r}_{ij}$ shows up constantly. Expanding it into SH gives you features that transform in a controlled way under rotations.

## low-order pictures (mental images)

- $\ell = 0$: one “blob” uniformly covering the sphere → a scalar.
- $\ell = 1$: two lobes (positive/negative) separated by a plane → like a vector (p-orbital look).
- $\ell = 2$: four lobes (actually five independent patterns total) → a quadrupole (d-orbital vibes).
- Higher $\ell$: more lobes, finer angular detail.

## complex vs. real SH

- **Complex SH** $Y_{\ell m}$ are standard in math/physics texts.
- **Real SH** are specific real-valued combinations of the complex ones; they’re what code often uses (no complex numbers needed).

Under rotation, complex SH transform with unitary matrices; real SH use orthogonal matrices. Same idea, different basis.

## key property (the rotation rule)

If you rotate the sphere by a 3D rotation $R$,

$$
Y_{\ell m}(R \hat{r}) = \sum_{m'=-\ell}^{\ell} D_{mm'}^{(\ell)}(R) Y_{\ell m'}(\hat{r})
$$

**Translation**: rotate the input direction, and your $\ell$-patterns mix among themselves via the matrix $D^{(\ell)}(R)$. They never mix with patterns of a different $\ell$.

**Intuition**: $\ell$ is a “frequency band” on the sphere; rotations don’t cross bands, they only rotate content within a band.

## quick examples you can feel

- A constant function on the sphere is pure $\ell=0$. Rotate it: unchanged.
- The z-coordinate function ($\cos \theta$) is $\ell=1$. Rotate it: it becomes a different combination of the three $\ell=1$ patterns (like reorienting a vector).
- A dumbbell-like pattern with four lobes is $\ell=2$. Rotate the globe and you just turn the dumbbell; still $\ell=2$.

**Try it**: imagine a heat map on a sphere with hot at the north pole and cold at the south pole. That’s $\ell=1$. If you spin the globe, the hot-cold axis moves—but you never create new extra lobes. That’s the $\ell$-band staying intact.

# 2) Wigner D-matrices: the rotation tables

## what they are

For each $\ell$, the Wigner D-matrix $D^{(\ell)}(R)$ is a $(2\ell + 1) \times (2\ell + 1)$ matrix that tells you how an $\ell$-type object transforms under a 3D rotation $R$.

If your feature lives in the $\ell$-space (e.g., the coefficients of the $\ell$ spherical harmonics), applying $R$ means multiplying by $D^{(\ell)}(R)$.

**Composition works as expected**:

$$
D^{(\ell)}(R_1 R_2) = D^{(\ell)}(R_1) D^{(\ell)}(R_2)
$$

## concrete cases you know

- $\ell = 0$: $D^{(0)}(R) = [1]$. Scalars are unchanged by rotation.
- $\ell = 1$: $D^{(1)}(R)$ acts just like the usual $3 \times 3$ rotation matrix on vectors (in a suitable basis).
- $\ell = 2$: $D^{(2)}(R)$ is $5 \times 5$. It rotates the five coefficients that encode a quadrupole pattern.

**Intuition**: think of $D^{(\ell)}(R)$ as the lookup table for “how does an $\ell$-thing rotate?” Vectors use the 3×3 table, quadrupoles use the 5×5 table, etc.

## small-angle link to generators

For a tiny rotation by angle $\epsilon$ about axis $\hat{n}$,

$$
D^{(\ell)}(R) \approx I + \epsilon (\hat{n} \cdot J^{(\ell)})
$$

where $J^{(\ell)}$ are the $(2\ell + 1) \times (2\ell + 1)$ generator matrices for that $\ell$.

This mirrors how $e^{\epsilon J}$ builds a finite rotation from an infinitesimal one.

**Try it**: If $\ell=1$, the generators $J^{(1)}$ are (up to convention) the standard 3D angular momentum matrices—exactly what you get when rotating vectors.

# 3) Tensor products & Clebsch–Gordan (CG): combining angular content

## the intuitive story

Sometimes you have two angular things and you combine them. Examples:

- Multiply two functions on a sphere.
- Take the outer product of two vectors.
- Send node features (type $\ell_1$) along an edge with direction SH of order $\ell_2$ in a message-passing layer.

**Question**: What angular types can the result contain?

**Answer (the rule)**: When you combine $\ell_1$ and $\ell_2$, the result can contain every $\ell$ from $|\ell_1 - \ell_2|$ up to $\ell_1 + \ell_2$, each at most once:

$$
(\ell_1) \otimes (\ell_2) \cong \bigoplus_{\ell = |\ell_1 - \ell_2|}^{\ell_1 + \ell_2} (\ell)
$$

This is the Clebsch–Gordan decomposition (CG for short).

## everyday examples

- **vector ⊗ vector**: $(\ell=1) \otimes (\ell=1) = (0) \oplus (1) \oplus (2)$
  - Scalar part (dot product) → $\ell=0$
  - Antisymmetric part (cross product) → $\ell=1$ (pseudovector)
  - Symmetric trace-free part (quadrupole) → $\ell=2$

- **scalar ⊗ anything**: $(0) \otimes (\ell) = (\ell)$ (scalars don’t change angular type)

- **vector ⊗ quadrupole**: $(1) \otimes (2) = (1) \oplus (2) \oplus (3)$

## parity (with reflections allowed)

If you also care about inversion parity (O(3)):

- even × even → even,
- odd × odd → even,
- even × odd → odd.

(Parity multiplies.)

## the book-keeping matrices: CG coefficients

The Clebsch–Gordan coefficients are just the change-of-basis numbers that convert between:

- the **uncoupled basis** $|\ell_1 m_1\rangle \otimes |\ell_2 m_2\rangle$
- and the **coupled basis** $|\ell m\rangle$.

They enforce the selection rules:

- **triangle rule**: $\ell \in [|\ell_1 - \ell_2|, \ell_1 + \ell_2]$
- **magnetic numbers**: $m = m_1 + m_2$
- **parity**: as above (if working in O(3))

You don’t have to memorize the numbers; you just need to know they exist and that they’re fixed by symmetry. In equivariant neural layers, these CG tensors are built in; the network only learns scalar (rotation-invariant) radial weights.

## how this powers equivariant message passing (one clear picture)

For an edge $i \leftarrow j$, compute the direction $\hat{r}_{ij}$.

Expand that direction into SH up to some order (say $\ell \leq 3$): you now have edge features of orders $0,1,2,3$.

Your node feature at $j$ lives in some mix of $\ell$-types (e.g., scalars, vectors, quadrupoles).

To send a message, you combine node’s $\ell_1$ with edge’s $\ell_2$ using CG → you get allowed outputs $\ell$.

You scale each output $\ell$ by a learned radial scalar (a function of the edge distance only), which is safe because scalars commute with rotations.

You sum over neighbors and (later) map to the final desired type (often a scalar $\ell = 0$ energy).

**Everything stays equivariant because:**

- rotations only mix within the same $\ell$ via $D^{(\ell)}$, and
- we never cheat by using non-scalar learned gates to couple different $\ell$’s.

# quick glossary

- $\hat{r}$: a unit vector (direction)
- $Y_{\ell m}$: spherical harmonic, an angular “basis pattern” on $S^2$
- **real vs. complex SH**: two equivalent bases; real SH avoid complex numbers, useful in code
- $D^{(\ell)}(R)$: the rotation matrix for $\ell$-type features (size $2\ell+1$)
- **generator**: matrix that produces an infinitesimal rotation; exponentiate to get a finite one
- **tensor product $\otimes$**: the mathematical way to combine two representations
- **CG coefficients**: numbers that recouple $(\ell_1, m_1), (\ell_2, m_2)$ into $(\ell, m)$ with $m = m_1 + m_2$
- **selection rules**: triangle rule for $\ell$, and $m$-addition

# tiny self-checks (answers below)

**Q1**: what $\ell$-values appear in $(1) \otimes (1)$?  
**Q2**: if a feature is pure $\ell=0$, how does it transform under rotation?  
**Q3**: combining an odd-parity vector with an odd-parity vector gives what parity?  
**Q4**: which matrix rotates a quadrupole’s 5 coefficients?  
**Q5**: why do we multiply messages by radial scalars only?

## answers

- $0 \oplus 1 \oplus 2$
- it doesn’t change (multiply by $D^{(0)} = [1]$)
- even (odd×odd→even)
- $D^{(2)}(R)$, the $5 \times 5$ Wigner matrix
- scalars commute with rotations, so they don’t break equivariance
