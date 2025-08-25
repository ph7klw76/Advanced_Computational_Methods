# A Gentle, Thorough Walkthrough of the HOMO→LUMO(+2) Excitonic Integrals Script (with the math)

This post explains what the script does, how it does it, and why each step matters—mathematically and computationally.  
The code computes two “excitonic” two-electron integrals for a specific orbital transition (HOMO → LUMO+2) from a Molden file:

- the **direct Coulomb term**  
  $J = (HH \mid LL)$

- the **singlet exchange term**  
  $K = (HL \mid LH)$

Both are reported in **Hartree** and **eV**, with careful numerical checks and a **density-fitting acceleration** that falls back to an exact streamed build if needed.

---

## 1) Big Picture

Given molecular orbitals (MOs) $\{\phi_p(r)\}$, the script:

- Loads MOs, energies, occupations, and the AO basis from a Molden file.
- Ensures MO coefficients are $S$-orthonormal (so later algebra is simple and numerically stable).
- Picks a spin block (α or β) and identifies HOMO and LUMO indices.
- Chooses a specific orbital transition $H \rightarrow L$ (here $H = \text{HOMO}$, $L = \text{LUMO+2}$).
- Builds two AO-space “densities” tied to that pair:
  - a **pure density** for the LUMO:
    
$$D_L = \lvert L \rangle \langle L \rvert$$
    
  - a **transition density**:
    
$$X = \lvert H \rangle \langle L \rvert$$
- Computes:
  - the direct Coulomb energy:
    
$$
J = \langle H \lvert J[D_L] \rvert H \rangle = (HH \mid LL)
$$

- the exchange energy:
    
$$
K = \mathrm{Tr}(X^T J[X]) = (HL \mid LH)
$$

- Prints diagnostics such as $\langle H \vert L \rangle$ and the ratio $K/J$.

---

##  Core Computation: Coulomb Builder

The heart of the computation is calling a Coulomb builder $J[D]$ that maps an AO density $D$ to an AO matrix $J[D]$ with elements:

$$
J_{\mu \nu}[D] = \sum_{\lambda \sigma} (\mu \nu \mid \lambda \sigma) \, D_{\lambda \sigma}
$$

where $(\mu \nu \mid \lambda \sigma)$ are the usual **electron repulsion integrals (ERIs)** in the AO basis.

## 2)  Inputs: Molden and Spin Blocks

`molden.load` gives you a **PySCF** `mol` object plus MO arrays.

```python
mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(MOLDEN_PATH)
mo_energy, mo_coeff, mo_occ = pick_spin_block(mo_energy, mo_coeff, mo_occ, which=SPIN_BLOCK)
```

If the calculation was **unrestricted**, there are separate **α/β blocks**; `pick_spin_block` chooses one.  
In **closed-shell** systems, $\alpha = \beta$, so either block is fine to use.

🧪 **Sanity Check:**  
The script reports `nao` and `nmo` to ensure correct orientation and dimensions.

---

## 3)  $S$-Orthonormalization: Why and How

We want the molecular orbital coefficients $C$ to satisfy:

$$
C^T S C = I
$$

```python
S = mol.intor('int1e_ovlp')
mo_coeff, fixed = ensure_S_orthonormal(mo_coeff, S)
```

This ensures MO orthonormality **in the AO metric**.

Minor inconsistencies can arise from:
- file formats,
- upstream processing, or
- numerical issues.

🔧 So the script:
- **verifies** the condition above, and
- **fixes** $C$ if necessary to enforce $S$-orthonormality.

##  The Math: Symmetric (Löwdin) Orthonormalization

Let:

$$
O = C^T S C
$$

If $O \ne I$, we diagonalize it:

$$
O = V \, \mathrm{diag}(w) \, V^T
$$

with all eigenvalues $w_i > 0$.  
Now define the inverse square root of $O$:

$$
O^{-1/2} = V \, \mathrm{diag}(w^{-1/2}) \, V^T
$$

We update the MO coefficients:

$$
C \leftarrow C \, O^{-1/2}
$$

Then the orthonormality condition is satisfied:

$$
C^T S C = O^{-1/2} \, O \, O^{-1/2} = I
$$

**Numerical Check:**  
The script validates this transformation. If the metric is ill-conditioned (i.e., any $w_i \le 0$), it raises an error.

---

### ⚙️ Why It Matters

With $C^T S C = I$, important simplifications follow:

- A **rank-1 MO density** $\lvert p \rangle \langle p \rvert$ in AO space becomes:

$$
D_p = C_p \, C_p^T
$$

  where $C_p$ is the AO column vector for orbital $p$.

- A **transition density** $X = \lvert H \rangle \langle L \rvert$ becomes:

$$
X = C_H \, C_L^T
$$

  which is generally **non-Hermitian**.

---

## 4) Picking HOMO and LUMO(+2)

(Next section starts here…)

```python
homo, lumo = homo_lumo_indices(mo_energy, mo_occ)
i_from = homo
i_to   = lumo + 2
```

##  4) Finding HOMO and LUMO(+2) Indices

The indices are determined based on MO **occupations**:

- Any occupation $> 10^{-8}$ is treated as **filled**.
- The script will **warn** if:
  - The selected "HOMO" is **not actually occupied**.
  - The chosen "LUMO+2" appears to be **occupied** (e.g., due to **fractional occupations** in post-HF or open-shell systems).

These checks ensure valid assumptions for the transition.

---

##  5) Building the Densities

(Next section begins here…)

```python
C_H = mo_coeff[:, i_from].reshape(-1, 1)
C_L = mo_coeff[:, i_to  ].reshape(-1, 1)

D_L, X = build_densities(C_H, C_L)
# D_L = C_L C_L^T   (Hermitian)
# X   = C_H C_L^T   (non-Hermitian)
```

##  5) Building the Densities

With $S$-orthonormal MOs, these are the correct AO-space matrices:

- **Pure LUMO density**:

$$
D_L = \lvert L \rangle \langle L \rvert
$$

  → used to compute the **Coulomb term**:

$$
J = (HH \mid LL)
$$

- **Transition density**:

$$
X = \lvert H \rangle \langle L \rvert
$$

  → used to compute the **singlet exchange term**:

$$
K = (HL \mid LH)
$$

 **Positivity Note:**  
Both $J$ and $K$ are **non-negative** in exact arithmetic since they are **Coulomb-type integrals**.  
Small negative values may appear due to **numerical noise** or **density fitting (DF)** artifacts — the script uses **tolerant non-negativity checks** to account for this.

---

## 6) From Densities to Coulomb Matrices $J[D]$

Two methods are available in the script, with **automatic fallback** between them:

```python
def get_J_DF(mol, D):   # density fitting (RI)
def get_J_exact(mol, D): # streamed exact integrals (no 4-index in RAM)

def get_J(mol, D):
    try:    return get_J_DF(...), "DF", elapsed
    except: return get_J_exact(...), "Exact", elapsed
```

###  6.1) Density Fitting (RI)

**Idea:** Approximate the **electron repulsion integrals (ERIs)** using an **auxiliary basis** $\{P\}$:

$$
(\mu \nu \mid \lambda \sigma) \approx \sum_{PQ} (\mu \nu \mid P) \, (J^{-1})_{PQ} \, (Q \mid \lambda \sigma)
$$

where:

$$
J_{PQ} = (P \mid Q)
$$

is the **Coulomb metric** over the auxiliary basis.

This approximation reduces:
- **4-index storage/work** → **3-index tensors**
- Computational cost from $\mathcal{O}(N^4)$ → roughly $\mathcal{O}(N^3)$

---

```python
vj, _ = dfobj.get_jk([D], hermi=0, with_j=True, with_k=False)
```

returns the AO Coulomb matrix $J[D]$. `hermi=0` is crucial: $D$ may be non-Hermitian (the transition density $X$), and we must not symmetrize it.

### 6.2) Exact, streamed fallback

If DF fails (e.g., not enough disk space for _cderi), the script uses:

```python
vj, _ = hf.get_jk(mol, [D], hermi=0, with_j=True, with_k=False)
```
which computes $J[D]$ exactly without forming/storing the full 4-index ERIs in RAM (it streams AO integrals).

## 7) Extracting the scalar $J$ and $K$

Once we have AO matrices $J[D]$, the target scalars are simple quadratic traces.

### 7.1) Direct Coulomb $J = (HH \mid LL)$

We first build $J[D_L]$ then evaluate

$$
J = \langle H \mid J[D_L] \mid H \rangle = C_H^T \, J[D_L] \, C_H
$$

```python
J_AO, method_JL, tJ = get_J(mol, D_L)
J = quad_form(C_H, J_AO, C_H)  # float(C_H.T @ J_AO @ C_H)
```
### 7.2) Singlet exchange $K = (HL \mid LH)$

For the transition density $X = \lvert H \rangle \langle L \rvert$, form $J[X]$ and compute

$$
K = \mathrm{Tr}(X^T J[X]) = \sum_{\mu \nu} X_{\mu \nu} \, J[X]_{\mu \nu} = (HL \mid LH)
$$

Code:

```python
JX_AO, method_X, tX = get_J(mol, X)
K = float(np.sum(X * JX_AO))  # == trace(X.T @ J[X])
```

### Why this matches the textbook formula

Write $X_{\mu \nu} = C_{\mu H} C_{\nu L}$. Then

$$
\mathrm{Tr}(X^T J[X]) = \sum_{\mu \nu \lambda \sigma} C_{\mu H} C_{\nu L} (\mu \nu \mid \lambda \sigma) C_{\lambda L} C_{\sigma H} = (HL \mid LH),
$$

exactly the exchange integral between $H$ and $L$.

**Note on spin:** In CIS/TDHF matrix elements, the singlet channel couples with $+K$ while the triplet channel couples with $-K$.  
This script computes the basic Coulomb-type integral $K = (HL \mid LH)$; how it enters an excitation energy depends on the electronic structure method.

---

## 8) Units, diagnostics, and physical interpretation

**Units:** results are printed in Hartree and eV using  
$1 \ \text{Ha} = 27.211386245988 \ \text{eV}$.

**Non-negativity checks:** exact $J$ and $K$ are $\ge 0$.  
Tiny negative values beyond `TOL_NONNEG` indicate numerical or DF issues.

**MO overlap** $\langle H \mid L \rangle$ (computed with the AO overlap $S$):

$$
S_{HL} = \langle H \mid L \rangle = C_H^T S C_L
$$

If $|\langle H \mid L \rangle|$ is small but $K/J$ isn’t, consider basis quality or DF accuracy.  
Conversely, a large $|\langle H \mid L \rangle|$ with tiny $K$ suggests a mismatch (wrong spin block, wrong orbitals, etc.).

---

### Physical read:

- $J = (HH \mid LL)$ correlates with electron–hole Coulomb attraction; larger $J$ → tighter exciton (all else equal).

- $K = (HL \mid LH)$ is sensitive to spatial overlap of $H$ and $L$; it’s typically small for spatially separated frontier orbitals and larger when they reside on the same fragment.

---

## 9) Why `hermi=0` appears repeatedly

Both `get_jk` calls set `hermi=0`.  
This tells PySCF **do not symmetrize the density**.

For $D_L$ that doesn’t matter (it’s Hermitian), but for the transition density $X$ it’s essential: $X \ne X^T$.

If you set `hermi=1`, you would silently compute with $(X + X^T)/2$, corrupting $K$.

---

## 10) Practical knobs & robustness

At the top of the script:

- `DF_AUXBASIS = "weigend"`: robust default for RI/DF.
- `DF_MAX_MEMORY_MB`: cap to keep chunks safe.
- `DF_BLOCKDIM`: controls chunk sizes in the 3-index intermediates.  
  If you see HDF5 broadcast errors, lower it.
- `DF_KEEP_IN_RAM`: `True` avoids HDF5 entirely but needs more memory.
- `MOLDEN_PATH`: must point to a valid Molden file with MOs and occupations.
- `SPIN_BLOCK`: `"alpha"` or `"beta"` for unrestricted inputs.

 The code times each build and **falls back to exact streamed Coulomb** if DF fails, printing the reason.

---

## 11) Verifiable claims & quick checks

- **Löwdin orthonormalization works**:  
  after calling it, verify numerically that  
  $\|C^T S C - I\|_\infty$ is within ~1e-8 — exactly what the script checks.

- **Rank-1 density**:  
  for $D_L = C_L C_L^T$,  
  $\mathrm{Tr}(S D_L) = \langle L \mid L \rangle = 1$  
  (with $S$-orthonormal MOs).

- **Non-negativity**:  
  finite basis and exact ERIs guarantee $J \ge 0$, $K \ge 0$.  
  Small negative outputs flag numerical or DF approximations;  
  the script prints warnings if below tolerance.

- **Equivalence of formulas**:
  - $J = \langle H \mid J[D_L] \mid H \rangle$ equals $(HH \mid LL)$ by expanding in AO indices.
  - $K = \mathrm{Tr}(X^T J[X])$ equals $(HL \mid LH)$ by the expansion shown above.

---

## 12) How to run and interpret

Make sure you have **PySCF installed** and a valid `111T1cis.molden` file in the working directory.

Run the script. You’ll see:

- whether the MOs needed orthonormalization,
- the selected transition indices,
- the method used to build each Coulomb (DF or Exact) and timings,
- $J$ and $K$ in Hartree/eV,
- diagnostics: $S_{HL}$, $K/J$, and notes if something looks off.

---

### What to expect

- $J$ tends to be larger than $K$, especially when $H$ and $L$ live on different fragments.
- If $H$ and $L$ strongly overlap, both $S_{HL}$ and $K$ rise; $K/J$ increases.


## Key Code Hotspots (at a Glance)

- **Orthogonalization:**  
  `ensure_S_orthonormal` → Löwdin $O^{-1/2}$.

- **Density construction:**  
  `build_densities(C_H, C_L)` →  
  $D_L = C_L C_L^T$,  
  $X = C_H C_L^T$.

- **Coulomb build:**  
  `get_J_DF` / `get_J_exact` with `hermi=0`.

- **Scalar extraction:**
  - `quad_form(C_H, J_AO, C_H)` → $J$.
  - `np.sum(X * JX_AO)` →
    
$$
K = \mathrm{Tr}(X^T J[X])
$$



