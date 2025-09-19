# part A — rotations in 3-D: SO(3), O(3), Euler angles, determinants

## what is SO(3)?

SO(3) is the set of all proper rotations of 3-D space (turns around some axis, by some angle), written as 3×3 matrices $R$.

“proper” rotation = a pure turn (no mirror flip).

matrix test: $R^\top R = I$ (so it preserves lengths and angles) and $\det R = +1$ (so it keeps orientation / handedness).

connected & compact: you can smoothly wiggle from any rotation to any other; the whole set is bounded (angles wrap around).

### examples you know

- rotate a cube 90° about the z-axis → that’s an element of SO(3).
- spin your phone around its long edge → SO(3).

### how many “knobs” (parameters) describe a rotation?

three. one way to see this: pick an axis (a unit vector on a sphere → 2 parameters), and an angle (1 parameter). total = 3.

## Euler angles (α, β, γ)

A common 3-parameter recipe to build any 3-D rotation:

- rotate by α around z,
- then by β around the (new) y,
- then by γ around the (new) z.

Different conventions exist (zyz, zxz, …), but the point is: three angles suffice.

## what is O(3)?

O(3) includes all distance/angle-preserving 3×3 matrices—rotations and reflections.

formally: $O(3) = \{ Q \mid Q^\top Q = I \}$

it splits into two pieces:

- SO(3) ($\det = +1$): proper rotations,
- the improper ones ($\det = -1$): reflections and rotoreflections.

### parity (P = −I)

The matrix $P = -I$ sends every vector $x$ to $-x$.

$\det(-I) = (-1)^3 = -1$, so it flips orientation (a mirror flip of the entire space).

In physics, we say “parity inversion” or just “parity”.

### quick picture

- SO(3): all the “turns.”
- O(3) \ SO(3): “turns with a mirror flip” (improper rotations).

Try-it: rotate your right hand 180° around any axis—still a right hand (SO(3)). Now look at your hand in a mirror—left hand (O(3) but not SO(3)).

# part B — the Lie algebra so(3): infinitesimal rotations, generators, brackets

## what is a Lie algebra?

Think of the Lie algebra as the tangent space of the group at the identity—the place where infinitesimal moves live. For rotations, the Lie algebra is called so(3) (little letters).

$so(3)$ = all real, antisymmetric 3×3 matrices $A$ (i.e., $A^\top = -A$).

Each such $A$ represents an “infinitesimal rotation.”

## generators $J_x$, $J_y$, $J_z$

A convenient basis of so(3) is three special matrices that generate tiny rotations about the x, y, and z axes. One common choice:

$$
J_x =
\begin{pmatrix}
0 & 0 & 0 \\
0 & 0 & -1 \\
0 & 1 & 0
\end{pmatrix},\quad
J_y =
\begin{pmatrix}
0 & 0 & 1 \\
0 & 0 & 0 \\
-1 & 0 & 0
\end{pmatrix},\quad
J_z =
\begin{pmatrix}
0 & -1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 0
\end{pmatrix}
$$

(there are sign-convention variants; all are equivalent up to re-labeling.)

## from “tiny” to “finite”: the exponential map

A finite rotation by angle $\theta$ about a unit axis $\hat{n}$ is

$$
R(\hat{n}, \theta) = \exp\left(\theta (\hat{n} \cdot J)\right)
$$

where $\hat{n} \cdot J = n_x J_x + n_y J_y + n_z J_z$.

Intuition: accumulate many tiny spins → a real rotation.

## the commutators (the right-hand rule in algebra form)

The commutator $[A, B] = AB - BA$ measures “how much order matters.”

For so(3):

- $[J_x, J_y] = J_z$
- $[J_y, J_z] = J_x$
- $[J_z, J_x] = J_y$

This encodes the right-hand rule: spin about x then y is like also having a bit of z-spin, etc. It’s the algebraic way of saying 3-D rotations don’t commute.

Try-it: multiply the matrices above to verify one of the brackets (e.g., compute $J_x J_y - J_y J_x$ and check it equals $J_z$).

# part C — irreducible representations (irreps) of SO(3): labels, sizes, parity

## what’s a representation again?

A representation tells you how features transform when space is rotated. For SO(3), the most important family of representations is labeled by an integer $\ell = 0, 1, 2, \dots$ (think “angular momentum”).

the $\ell$-irrep has dimension $2\ell + 1$.

under a rotation $R$, a feature vector $x^{(\ell)} \in \mathbb{C}^{2\ell+1}$ transforms as:

$$
x^{(\ell)} \mapsto D^{(\ell)}(R)\,x^{(\ell)}
$$

where $D^{(\ell)}(R)$ is the Wigner D-matrix for order $\ell$.

## mental model

$\ell$ tells you the “spin/order” of the geometric object:

- $\ell = 0$: scalars (no direction),
- $\ell = 1$: vectors (arrows),
- $\ell = 2$: quadrupoles (ellipsoid-like patterns),
- higher $\ell$: more “lobes” / finer angular detail.

## dimensions you can feel

- $\ell = 0$: $2\ell + 1 = 1$ → like temperature at a point (unchanged by rotation).
- $\ell = 1$: 3 numbers → a vector $(x, y, z)$ rotates like your usual 3-D arrow.
- $\ell = 2$: 5 numbers → the “shape” part of a symmetric, trace-free 3×3 tensor.

Chemistry tie-in: s, p, d orbitals around an atom can be matched with $\ell = 0, 1, 2$ angular patterns. p-orbitals (px, py, pz) are the $\ell = 1$ triplet.

## parity: even vs odd (extend to O(3))

When you also allow reflections (O(3)), each $\ell$ comes with a parity tag:

- even (e): unchanged under inversion $x \mapsto -x$
- odd (o): flips sign under inversion

### quick examples

- $\ell = 0e$: scalar — energy, temperature → even (no sign change)
- $\ell = 1o$: vector — velocity, electric dipole → odd (arrow reverses under a mirror flip)
- $\ell = 2e$: quadrupole — even

That shorthand you often see:  
“1x1o + 1x2e”  
means: one channel of a vector-type feature ($\ell = 1$, odd parity) plus one channel of a quadrupole-type feature ($\ell = 2$, even parity)

### Try-it

Is angular momentum (a pseudovector) even or odd under inversion?

- a true vector (like position) flips sign under inversion → odd.
- a pseudovector (like torque or angular momentum) does not flip sign → even.

(This is why careful contexts distinguish vectors from pseudovectors; in many ML libraries the $\ell = 1$ block is treated as “odd” by default, matching true vectors.)

# part D — worked mini-examples (concrete transformations)

## D1. scalar, vector, quadrupole under a rotation

- scalar $s$: $s \mapsto s$ (unchanged).
- vector $v$: $v \mapsto Rv$ (multiply by the 3×3 rotation).
- quadrupole $Q$ (symmetric trace-free tensor): $Q \mapsto R Q R^\top$.

If you repack $Q$ into a 5-vector $x^{(2)}$, that 5-vector transforms by the 5×5 $D^{(2)}(R)$ matrix.

**intuition**: the $\ell$ label picks the right “rotation table” $D^{(\ell)}(R)$ for that object.

## D2. see the 3 commutators in action

Take a small rotation $\epsilon$ about x, then a small rotation $\epsilon$ about y. The mismatch between “x then y” vs “y then x” is a tiny z-rotation proportional to $\epsilon^2$. That’s what $[J_x, J_y] = J_z$ encodes.

## D3. what parity does to features

Flip space by $P = -I$:

- a scalar feature $s$ stays $s$ (even),
- a vector $v$ becomes $-v$ (odd),
- a quadrupole stays the same pattern (even).

# part E — glossary (bite-sized definitions)

- **SO(3)**: all proper 3-D rotations ($\det = +1$)
- **O(3)**: all 3-D rotations and reflections ($\det = \pm1$)
- **parity (P)**: the inversion $x \mapsto -x$; $P = -I$
- **Lie algebra so(3)**: all real antisymmetric 3×3 matrices; encodes infinitesimal rotations
- **generators $J_x$, $J_y$, $J_z$**: basis of so(3) corresponding to tiny rotations about axes
- **commutator $[A,B]=AB−BA$**: measures non-commutativity; for so(3) it matches the right-hand rule
- **irrep (irreducible representation)**: a “cannot-be-split” way a feature can transform; labeled by $\ell$ for SO(3)
- **dimension of an irrep**: $2\ell + 1$
- **Wigner $D^{(\ell)}(R)$**: the $(2\ell+1)\times(2\ell+1)$ matrix that rotates an $\ell$-type feature
- **parity tag (e/o)**: how a feature behaves under inversion (even/odd)

# part F — quick self-check (answers inline)

**Q1**: is every element of O(3) a rotation?  
**A**: no. O(3) also includes reflections ($\det = -1$).

**Q2**: how many parameters describe an SO(3) rotation?  
**A**: three. (axis 2 + angle 1, or Euler angles.)

**Q3**: what matrices live in so(3)?  
**A**: real antisymmetric 3×3 matrices.

**Q4**: what is $[J_x, J_y]$?  
**A**: $J_z$ (and cyclic permutations).

**Q5**: what’s the size of the $\ell=2$ irrep?  
**A**: 5 (since $2\ell+1 = 5$).

**Q6**: under inversion, does a vector change sign?  
**A**: yes (odd).  
Does a quadrupole?  
**A**: no (even).
