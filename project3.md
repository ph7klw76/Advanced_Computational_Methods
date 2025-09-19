# 1) groups: the language of symmetry

## what’s a “group” (plain-english)

a group is a collection of “moves” you can do that:

- you can combine (do one move after another),
- there’s a do-nothing move (identity),
- every move can be undone (inverse),
- and combining moves is associative $((a·b)·c = a·(b·c))$.

## definitions (short + intuitive)

set $G$: the collection of moves.

operation “·”: how to combine two moves (e.g., do move A then B).

associative: order of bracketing doesn’t matter: $(a⋅b)⋅c = a⋅(b⋅c)$.

identity $e$: a special “do-nothing” move so $e⋅g = g⋅e = g$.

inverse $g^{-1}$: an undo-move so $g⋅g^{-1} = g^{-1}⋅g = e$.

## everyday examples

- clock arithmetic (mod 12): the “moves” are adding hours; identity = add 0; inverse of +3 is +9 (because $3+9=12≡0$).
- shuffling a deck: permutations form a group; identity = “don’t shuffle”; inverse = “unshuffle”.
- translations in 3D $(\mathbb{R}^3, +)$: move by a vector $t$; identity = move by $(0,0,0)$; inverse of $t$ is $-t$.

## key adjectives you’ll see

- abelian: order doesn’t matter, $a⋅b = b⋅a$ (e.g., translations; but rotations in 3D are not abelian).
- compact (intuitive): “bounded + closed” as a space of moves; e.g., the set of 3D rotations is compact (angles wrap around), but all translations are not (you can wander off to infinity).

## important symmetry groups in geometry

**SO(3)**: all proper 3D rotations. as matrices:

$$
SO(3) = \{ R \in \mathbb{R}^{3\times3} \mid R^\top R = I,\ \det R = 1 \}
$$

non-abelian; think: rotate around x, then y ≠ y then x.

**O(3)**: rotations and reflections (includes parity). this is $SO(3)$ plus flips (determinant ±1).

# 2) representations: turning moves into matrices

## the idea

a representation says: “let each abstract move $g \in G$ act as a linear transformation on a vector space $V$.” that’s perfect for math & ML: vectors in $V$ are your features, and group moves become matrices that act on features.

## formal definition

a representation is a homomorphism

$$
\rho: G \to GL(V)
$$

where $GL(V)$ is the set of all invertible linear maps on $V$. it must preserve multiplication:

$$
\rho(g_1 g_2) = \rho(g_1)\rho(g_2), \quad \rho(e) = I
$$

## intuition

“doing $g_1$ then $g_2$” equals “apply matrix $\rho(g_1)$, then $\rho(g_2)$.”

## terms & pictures

- vector space $V$: where your data lives (e.g., $\mathbb{R}^3$ for 3D vectors).
- linear map: matrix that scales/rotates/shears but respects addition & scalar multiplication.
- homomorphism: structure-preserving map (here, it preserves the way moves combine).
- $GL(V)$: “general linear group” = all invertible matrices on $V$.

## concrete examples

### rotations acting on 3D vectors

$G = SO(3)$, $V = \mathbb{R}^3$. define $\rho(R) = R$ (the same 3×3 rotation matrix). that’s the standard representation.

### complex exponentials (1D irrep of rotations in 2D)

let $G = S^1$ (angles on a circle). define $\rho_\ell(\theta) = e^{i\ell\theta}$ for integer $\ell$. each $\rho_\ell$ is a 1-dimensional irreducible representation (a “character”).

### permutation representation

$G$ = permutations of $n$ items; $V = \mathbb{R}^n$. a permutation $\sigma$ acts by reordering coordinates. $\rho(\sigma)$ is the $n \times n$ permutation matrix.

## equivalence of representations

two reps $\rho: G \to GL(V)$ and $\pi: G \to GL(W)$ are equivalent if there exists an invertible linear map $T: V \to W$ with

$$
T\,\rho(g) = \pi(g)\,T \quad \forall g \in G
$$

intuition: they’re the same action in different coordinates.

## invariant subspaces & irreducibility

a subspace $U \subset V$ is invariant if $\rho(g)U \subseteq U \ \forall g$ (the action never leaves $U$).

a rep is **irreducible** (an **irrep**) if it has no nontrivial invariant subspaces (only $\{0\}$ and $V$).

intuition: you can’t split the feature space into smaller independent “blocks” that the group preserves.

**big fact (compact groups)**: any finite-dimensional unitary representation can be decomposed into a direct sum of irreps (think “block-diagonalize” into smallest indivisible pieces).

# 3) intertwiners: symmetry-respecting linear maps

## definition + intuition

given two representations $\rho: G \to GL(V)$ and $\pi: G \to GL(W)$, an **intertwiner** is a linear map

$$
L: V \to W \quad \text{such that} \quad L\,\rho(g) = \pi(g)\,L \quad \forall g \in G
$$

intuition: do the group move first then apply $L$, or apply $L$ first then the group move—same result.

intertwiners **commute with the group action**. they are the legal “wires” between equivariant layers.

## pictures & examples

### copying scalars

if $V = W = \mathbb{R}$ and the group acts trivially ($\rho(g) = \pi(g) = 1$), any linear map $L(x) = ax$ is an intertwiner (scalars are invariant → safe).

### 3D rotations on vectors

if both $V$ and $W$ carry the standard SO(3) vector action, then any rotation-equivariant linear map must be a scalar multiple of the identity (you can’t prefer a direction without breaking symmetry).

### different types (vector→scalar)

with standard SO(3) action on vectors for $V$, and trivial action on $W = \mathbb{R}$, there is no nonzero intertwiner: a linear map that turns a vector into a scalar while commuting with every rotation would have to ignore direction—in linear algebra terms, the only such map is the zero map.

## schur’s lemma (the two bullets that rule everything)

for **unitary irreps**:

- if $V$ and $W$ carry **inequivalent irreps**, any intertwiner $L: V \to W$ is 0.
- if $V = W$ is a single irrep, any intertwiner is a scalar multiple of the identity.

## why you should care (ML intuition)

- you cannot linearly mix two different irreps unless you go through an allowed coupling (e.g., tensor products + projections).
- within the same irrep type, the only linear, symmetry-respecting map is “scale each channel the same way.” that’s why equivariant layers carefully route features by irrep type and use scalar gates (which commute with symmetry).

# 4) putting it all together (with mini-labs)

## story so far

- groups capture symmetry (e.g., rotate a molecule).
- representations tell us how features should transform under those symmetries (vectors rotate, scalars don’t).
- intertwiners are exactly the linear maps that respect those transformations (equivariant maps).
- schur’s lemma explains why equivariant architectures have a block-structured, selection-rule feel.

## tiny “try it” checks (no code needed)

### is this a group?

the set of 2D rotations with composition.

– do we have identity? yes, rotate by 0°.  
– inverses? yes, rotate by −θ.  
– associativity? yes (matrix mult).  
– closed? yes.  
→ group.

### abelian or not?

- 2D rotations: abelian ($\theta_1$ then $\theta_2$ = $\theta_2$ then $\theta_1$).
- 3D rotations: not abelian (rotate about x then y ≠ y then x).

### representation or not?

define $\rho(\theta)$ as the 2×2 rotation matrix in the plane. does $\rho(\theta_1 + \theta_2) = \rho(\theta_1)\rho(\theta_2)$?  
yes → representation.

### invariant subspace?

for 3D rotations acting on $\mathbb{R}^3$, is the x-axis invariant under all rotations?  
no (a rotation about y tilts it).  
but the zero subspace and the whole $\mathbb{R}^3$ are invariant (always).  
**conclusion**: the standard 3D vector rep of SO(3) is irreducible over $\mathbb{R}$ in the sense used in physics (more formally: as an $SO(3)$ real irrep of type 1o).

### intertwiner test

suppose $L: \mathbb{R}^3 \to \mathbb{R}^3$ commutes with every rotation matrix $R$: $LR = RL$ for all $R \in SO(3)$. what is $L$?  
by schur: $L = \lambda I$ (scale + identity). any anisotropic linear map (like a stretch in x only) would pick a preferred direction and fail to commute with some rotation.

# 5) glossary (quick definitions you can point to)

- group $(G,⋅)$: set with associative operation, identity, inverses.
- abelian: commutative group (order doesn’t matter).
- compact group: roughly, bounded/closed as a topological space (e.g., all 3D rotations).
- representation $\rho: G \to GL(V)$: assigns each group element an invertible linear map on $V$, preserving multiplication.
- $GL(V)$: group of all invertible linear maps on $V$.
- equivalent representations: the same action in different coordinates, related by an invertible change of basis $T$.
- invariant subspace: subspace preserved by the action of every group element.
- irreducible representation (irrep): no nontrivial invariant subspaces.
- intertwiner $L$: linear map that commutes with the group action: $L\rho(g) = \pi(g)L$.
- schur’s lemma: between inequivalent irreps, intertwiners are 0; on the same irrep, intertwiners are scalars times identity.

# 6) why this matters for equivariant models (intuitive bridge)

when we say a layer is equivariant, we mean: “if you transform the input by a group move, the output transforms in the same way.”

to guarantee that, every linear piece in the layer must be an intertwiner.

schur’s lemma then forces a block structure: you can only linearly mix channels that belong to the same irrep type, and even then only by a scalar multiple (or by carefully constructed tensor-product couplings that produce allowed irreps).

that’s exactly why modern geometric nets track channels like “8×0e+1×1o+1×2e” and route them through symmetry-respecting operations.

# 7) extra examples (to cement the intuition)

## scalars vs. vectors (SO(3))

- scalars (temperature, energy): unchanged by rotation → trivial rep ($\rho(R) = 1$).
- vectors (force, dipole): rotate with $R$ → standard rep ($\rho(R) = R$).
- an equivariant linear map from vector→scalar must be zero (no direction is special).

## images under translations (2D)

let $G = \mathbb{R}^2$ (image shifts). a representation on pixel arrays shifts the array. a translation-equivariant filter (e.g., convolution) is an intertwiner between two such shift reps.

## permutation symmetry

in a set of indistinguishable particles, any reordering (permutation) should leave predictions consistent. intertwiners for the permutation representation are exactly the permutation-equivariant linear maps (they commute with any reindexing).

# 8) quick self-check quiz

**Q1**: give an example of a group that’s not abelian.  
**A**: $SO(3)$ (3D rotations).

**Q2**: what’s the identity and inverse in $(\mathbb{R}^3,+)$?  
**A**: identity is $(0,0,0)$; inverse of $t$ is $-t$.

**Q3**: what does it mean for two representations to be equivalent?  
**A**: they differ by a change of basis $T$: $T\rho(g) = \pi(g)T$.

**Q4**: why is any linear rotation-equivariant map $\mathbb{R}^3 \to \mathbb{R}^3$ a scalar multiple of $I$?  
**A**: schur’s lemma on the vector irrep.

**Q5**: can a nonzero linear intertwiner map a vector (standard SO(3) rep) to a scalar (trivial rep)?  
**A**: no—inequivalent irreps → only zero map.
