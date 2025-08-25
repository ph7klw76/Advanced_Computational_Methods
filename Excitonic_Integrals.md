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

 **Sanity Check:**  
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

 So the script:
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

###  Why It Matters

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


```python

import os
import shutil
import tempfile
from time import time
import numpy as np

# ---- Python 3.6 compatibility: polyfill contextlib.nullcontext (needed by PySCF exact JK path) ----
import contextlib
if not hasattr(contextlib, "nullcontext"):
    class _NullContext(object):
        def __init__(self, enter_result=None): self.enter_result = enter_result
        def __enter__(self): return self.enter_result
        def __exit__(self, *exc): return False
    contextlib.nullcontext = _NullContext

from pyscf.tools import molden
from pyscf import df
from pyscf.scf import hf

# -------------------- User knobs --------------------
MOLDEN_PATH       = "111T1cis.molden"   # hardcoded as requested
SPIN_BLOCK        = "alpha"          # "alpha" or "beta" if unrestricted
DF_AUXBASIS       = "weigend"        # typical RI/DF aux basis
DF_MAX_MEMORY_MB  = 4000             # cap DF memory to keep HDF5 chunks safe
DF_BLOCKDIM       = 8000             # conservative; lower if you still see HDF5 broadcast errors
DF_KEEP_IN_RAM    = False            # True -> no HDF5 I/O (needs more RAM)

TOL_ORTHO  = 1e-8
TOL_NONNEG = 1e-8
HARTREE_TO_EV = 27.211386245988

# -------------------- Helpers --------------------
def pick_spin_block(mo_energy, mo_coeff, mo_occ, which="alpha"):
    """Pick one spin block from Molden (alpha/beta). For closed-shell, alpha==beta."""
    if isinstance(mo_coeff, (tuple, list)):
        idx = 0 if which.lower().startswith("a") else 1
        return mo_energy[idx], mo_coeff[idx], mo_occ[idx]
    return mo_energy, mo_coeff, mo_occ

def homo_lumo_indices(mo_energy, mo_occ, tol=1e-8):
    """HOMO = last occ>tol ; LUMO = HOMO+1."""
    occ = np.asarray(mo_occ)
    occ_idx = np.where(occ > tol)[0]
    if occ_idx.size == 0:
        raise RuntimeError("No occupied orbitals found (mo_occ <= tol).")
    homo = int(occ_idx[-1])
    lumo = homo + 1
    if lumo >= occ.size:
        raise RuntimeError("No LUMO (HOMO is last).")
    return homo, lumo

def ensure_S_orthonormal(C, S, tol=TOL_ORTHO):
    """
    Ensure C^T S C = I. If not, fix via symmetric orthonormalization:
    C <- C @ ( (C^T S C)^(-1/2) ).
    """
    O = C.T @ S @ C
    I = np.eye(O.shape[0])
    if np.allclose(O, I, atol=tol):
        return C, False
    w, V = np.linalg.eigh(O)
    if np.any(w <= 0):
        raise RuntimeError("Non-positive metric in MO overlap; cannot orthonormalize.")
    Ominushalf = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    C_fix = C @ Ominushalf
    # verify
    Ofix = C_fix.T @ S @ C_fix
    if not np.allclose(Ofix, I, atol=10*tol):
        raise RuntimeError("Failed to enforce S-orthonormality (post-check failed).")
    return C_fix, True

def quad_form(vL, M, vR):
    """Return vL^T M vR for column vectors (nao x 1)."""
    return float(vL.T @ (M @ vR))

def build_densities(C_H, C_L):
    """
    D_L  = |L><L|  (nao x nao, Hermitian)   -> for J = (HH|LL)
    X    = |H><L|  (nao x nao, non-Hermitian) -> for K = Tr(X^T J[X])
    """
    D_L = C_L @ C_L.T
    X   = C_H @ C_L.T
    return D_L, X

def df_storage_info(dfobj):
    """Reveal DF HDF5 path for disk checks (best-effort)."""
    feri = getattr(dfobj, "_cderi", None)
    path = None
    try:
        path = getattr(feri, "filename", None) or getattr(feri, "name", None)
    except Exception:
        pass
    if path:
        print(">> DF cderi store:", path)
    return path

def get_J_DF(mol, D, auxbasis=DF_AUXBASIS, max_memory_mb=DF_MAX_MEMORY_MB,
             blockdim=DF_BLOCKDIM, keep_in_ram=DF_KEEP_IN_RAM):
    """
    Build Coulomb AO matrix J[D] via density fitting (DF).
    Works for Hermitian or non-Hermitian D (hermi=0).
    """
    dfobj = df.DF(mol).set(auxbasis=auxbasis)
    # Older PySCF prints "Overwritten attributes ..." when setting known attrs; harmless.
    try:
        dfobj.max_memory = int(max_memory_mb)  # MB
    except Exception:
        pass
    try:
        dfobj.blockdim = int(blockdim)
    except Exception:
        pass
    if keep_in_ram:
        dfobj._cderi_to_save = None  # keep 3c integrals in RAM (avoid HDF5)

    # Warn if temp disk space is low (when using HDF5)
    tmpdir = tempfile.gettempdir()
    try:
        free_gb = shutil.disk_usage(tmpdir).free / (1024**3)
        if not keep_in_ram and free_gb < 10:
            print("!! Warning: low free space in {}: {:.1f} GB".format(tmpdir, free_gb))
    except Exception:
        pass

    dfobj.build()
    df_storage_info(dfobj)

    # get_jk returns (vj, vk); we request only J to minimize work
    vj, _ = dfobj.get_jk([D], hermi=0, with_j=True, with_k=False)
    return vj[0]

def get_J_exact(mol, D):
    """
    Build Coulomb AO matrix J[D] via exact (non-DF) streamed AO integrals.
    No 4-index ERIs are stored in RAM.
    """
    vj, _ = hf.get_jk(mol, [D], hermi=0, with_j=True, with_k=False)
    return vj[0]

def get_J(mol, D):
    """
    Try DF first (conservative chunking). If DF fails, fall back to exact.
    Returns (J_AO, method_str, elapsed_seconds).
    """
    try:
        t0 = time(); J_AO = get_J_DF(mol, D); t1 = time()
        return J_AO, "DF", t1 - t0
    except Exception as e:
        print("\n!! DF J-build failed; using exact J. Reason:", repr(e))
        t0 = time(); J_AO = get_J_exact(mol, D); t1 = time()
        return J_AO, "Exact", t1 - t0

# -------------------- Main workflow --------------------
def main():
    print(">> Loading Molden:", MOLDEN_PATH)
    # molden.load parses units/AO order/spin blocks correctly
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(MOLDEN_PATH)

    # Choose spin block explicitly
    mo_energy, mo_coeff, mo_occ = pick_spin_block(mo_energy, mo_coeff, mo_occ, which=SPIN_BLOCK)

    nao, nmo = mo_coeff.shape
    print(">> Molecule built: nao={}, nmo={}".format(nao, nmo))

    # S-orthonormality (verify/fix)
    S = mol.intor('int1e_ovlp')
    mo_coeff, fixed = ensure_S_orthonormal(mo_coeff, S)
    if fixed:
        print(">> Warning: MOs were not S-orthonormal; corrected by symmetric orthonormalization.")

    # Canonical HOMO/LUMO to anchor indices
    homo, lumo = homo_lumo_indices(mo_energy, mo_occ)
    print(">> Canonical HOMO index = {}, LUMO index = {}".format(homo, lumo))

    # --------- Select the requested transition: HOMO-X -> LUMO+Y ----------
    i_from = homo
    i_to   = lumo + 2
    if i_from < 0 or i_to >= nmo:
        raise RuntimeError("Requested HOMO-C -> LUMO+Y is out of bounds (i_from={}, i_to={}, nmo={})"
                           .format(i_from, i_to, nmo))

    # Informative occupation sanity (not fatal)
    if mo_occ[i_from] <= 1e-8:
        print("!! Note: HOMO (index {}) is not marked occupied (mo_occ).".format(i_from))
    if mo_occ[i_to] > 1e-8:
        print("!! Note: LUMO+2 (index {}) appears occupied (fractional?).".format(i_to))

    print(">> Using transition: from index {} (HOMO) to index {} (LUMO+2)".format(i_from, i_to))

    # Column vectors for this pair
    C_H = mo_coeff[:, i_from].reshape(-1, 1)   # “H” side = HOMO-X
    C_L = mo_coeff[:, i_to  ].reshape(-1, 1)   # “L” side = LUMO+Y

    # Densities for J and K
    D_L, X = build_densities(C_H, C_L)

    # ---- Direct Coulomb: J = (HH|LL) = <H | J[D_L] | H> ----
    J_AO, method_JL, tJ = get_J(mol, D_L)
    J = quad_form(C_H, J_AO, C_H)

    # ---- Exchange (singlet): K = Tr( X^T J[X] ), with X = |H><L| ----
    JX_AO, method_X, tX = get_J(mol, X)
    K = float(np.sum(X * JX_AO))  # == trace(X.T @ J[X])

    # ---- Sanity checks ----
    if J < -TOL_NONNEG:
        print("!! Warning: J is negative beyond tolerance:", J)
    if K < -TOL_NONNEG:
        print("!! Warning: K is negative beyond tolerance:", K)

    # Separation sanity diagnostic: <H|L>
    S_HL = float(C_H.T @ (S @ C_L))
    ratio = (K / J) if J != 0 else np.nan

    print("\n=== Excitonic Integrals for HOMO -> LUMO+2 (gas-phase, AO-based) ===")
    print("Direct Coulomb   J = (HH|LL) = {: .10f} Hartree  | {: .6f} eV   [{} J[D_L] in {:.3f}s]"
          .format(J, J*HARTREE_TO_EV, method_JL, tJ))
    print("Exchange (singlet) K = Tr(X^T J[X]) = {: .10f} Hartree  | {: .6f} eV   [{} J[X] in {:.3f}s]"
          .format(K, K*HARTREE_TO_EV, method_X, tX))

    print("\nSanity diagnostics:")
    print("S-orthonormality check: passed" + (" (fixed)" if fixed else ""))
    print("Non-negativity (tol {:g}): J {}  |  K {}"
          .format(TOL_NONNEG, "OK" if J >= -TOL_NONNEG else "FAIL",
                  "OK" if K >= -TOL_NONNEG else "FAIL"))
    print("MO overlap  <H|L> = S_HL = {: .3e}".format(S_HL))
    print("Ratio K/J   = {: .3e}".format(ratio))
    if abs(S_HL) < 1e-3 and (not np.isnan(ratio)) and ratio > 1e-2:
        print("  !! Note: |<H|L>| is small but K/J is not; inspect basis/MOs/DF accuracy.")
    if abs(S_HL) > 1e-1 and K < 1e-5:
        print("  !! Note: <H|L> sizable but K ~ 0; confirm correct spin block and orbitals.")

    print("\nNotes:")
    print("K is computed via Coulomb on the transition density (no exchange builder), i.e., K=(HL|LH).")
    print("DF is approximate; if DF fails, exact J is used (streamed; no 4-index ERIs in RAM).")
    print("Results are unscreened (gas-phase). Screening in condensed media can reduce J and K substantially.")

if __name__ == "__main__":
    main()
```


