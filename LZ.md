# Landau–Zener (LZ) non-adiabatic transition probability

Below is a “from first principles” account of how the Landau–Zener (LZ) non-adiabatic transition probability is derived, how it is mapped onto molecular charge transport, how it interpolates between the classical Marcus picture and the fully adiabatic regime, and what approximations are involved. The treatment follows the original quantum-mechanical derivation by Landau (1932) and Zener (1932), but it is recast in the language used in modern organic-semiconductor transport theory.

## 1 Two–level Hamiltonian along a reaction coordinate

Consider two neighbouring redox sites A and B whose nuclei move collectively along a single reaction coordinate $q$ ($q$ in practice a linear combination of internal vibrations and an intermolecular libration).

The diabatic electronic energies are

$$
E_A(q)=E_{A0}+k\,q,\quad E_B(q)=E_{B0}-k\,q,
$$

so they cross at $q=0$. Coupling $J$ J mixes the electronic states and produces the 2×2 Hamiltonian

$$
H(q)=\begin{pmatrix}
k\,q & J\\
J & -k\,q
\end{pmatrix}.
$$

(1)


Diagonalising (1) gives adiabatic energies

$$
E_\pm(q)=\pm\sqrt{J^2+(k\,q)^2},
$$


which form an avoided crossing with a minimum gap $2J$ at $q=0$.

<img width="1000" height="663" alt="image" src="https://github.com/user-attachments/assets/0920a279-2023-4c25-b09e-b97aa0e6e065" />


## 2 Time-dependent crossing and the original LZ problem

Assume the nuclear subsystem moves at a constant velocity through the crossing:

$$
q(t)=v\,t,
$$

so the time-dependent Schrödinger equation reduces to a linear-sweep problem

<img width="373" height="94" alt="image" src="https://github.com/user-attachments/assets/4d6f6c37-23af-42ea-b399-70ae556ecd3e" />


with diabatic amplitudes $c_A(t),c_B(t)$ cA(t),cB(t). Imposing that the system is purely on state A at $t\to -\infty$ t→−∞, LZ showed that the probability to remain on A (i.e. not make a transition) after the passage is

$$
P_{ad}=\exp\!\Bigl(-\frac{2\pi J^2}{\hbar\,k\,v}\Bigr).
$$


The probability to transfer to B is therefore

$$
P_{LZ}=1-P_{ad}=1-\exp(-2\pi\Gamma),\quad \Gamma\equiv\frac{J^2}{\hbar\,k\,v}.
$$


Interpretation of $Γ$:

**Large $J$ or slow sweep → $Γ\gg1$ Γ≫1 → transition probability $P_{LZ}\to1$ P LZ →1: the crossing is essentially diabatic.

**Small $J$ or fast sweep → $Γ\ll1$ Γ≪1 → $P_{LZ}\simeq2\pi\Gamma\propto J^2$ P LZ ≃2πΓ∝J2: the electron stays adiabatic.

## 3 Connecting the sweep parameters to molecular properties

For a harmonic potential in Marcus theory the driving coordinate obeys

$$
\frac12\,k\,(q\pm q_0)^2,\quad q_0=\frac{2\lambda}{k},
$$


with reorganisation energy $λ$ .

Linearising near $q=0$  gives $k=2\lambda/q_0$ .

The characteristic velocity with which the nuclei traverse the crossing is set by the vibrational frequency $ω_{eff}$ ω eff:

$$
v\approx q_0\,ω_{eff}=\frac{2\lambda}{k}.
$$


Combining, the Landau–Zener adiabaticity parameter becomes

$$
Γ=\frac{J^2}{\hbar ω_{eff} λ}.
$$


Typical organic semiconductors:

λ = 0.2–0.4 eV

ωeff = 40–150 cm⁻¹ (5–20 meV) → 6–25×10¹² s⁻¹

<img width="600" height="85" alt="image" src="https://github.com/user-attachments/assets/fa2382d9-57aa-465f-a23e-e97f227218b4" />


## 4 Landau–Zener-corrected hopping rate

Multiply the golden-rule rate $k_{NA}\propto J^2$ k NA ∝J2 by the LZ transmission factor $\kappa_{LZ}=P_{LZ}$ κ LZ =P LZ:

$$
k_{hop}=k_{NA}\,[1-\exp(-2\pi\,Γ)].
$$

k hop =k NA [1−exp(−2πΓ)] .(7)

<img width="945" height="167" alt="image" src="https://github.com/user-attachments/assets/6713512d-5ea7-4979-ac93-a9b4520e8925" />


## 5 Physical consequences in organic charge transport

| Regime             | Transport Picture               | Experimental Signature                                                              |
|--------------------|----------------------------------|--------------------------------------------------------------------------------------|
| $J \ll 0.1$ eV     | Non-adiabatic hopping (Marcus)   | Arrhenius mobility with $\mu \propto J^2$.                                          |
| $J \sim 0.1–0.3$ eV | Intermediate                   | Mobility deviates from pure Arrhenius; weak field dependence.                       |
| $J \gtrsim 0.3$ eV | Adiabatic transfer               | Rate limited by phonon frequency; mobility plateaus; isotope substitution has small effect. |

**Examples**

Rubrene single crystals: nearest-neighbour $J≈0.14$, $eV$,  $Γ$≈$0.1$ → still non-adiabatic.

Pentacene dimers: $J≈0.25$\,eV, borderline; LZ correction reduces predicted rate by ~3×.

Fullerene PCBM dimer in the “face-on” stack: $J≈0.4$\,$eV$, $Γ≈1–2$ → LZ factor saturates (hops limited by 60 cm⁻¹ lattice phonon).

## 6 Limitations and advanced refinements

- **Multi-mode spectral density**: Real $J$-dependence is integrated over many promoting modes; Eq. (6) uses a single $ω_{eff}$ ω eff.
- **Quantum nuclei**: LZ is semiclassical; quantum tunnel splitting or zero-point motion at cryogenic T require Marcus–Levich–Jortner or full quantum dynamics.
- **Polaron effects**: If $J$ J is large enough to delocalise charge over several sites the underlying Hamiltonian is no longer strictly two-level; one must solve a Holstein–Peierls model or use surface-hopping MD.
- **Disorder**: Energetic and spatial disorder broaden the crossing region; averaging over site pairs often restores the usefulness of the single-parameter $Γ$.

## 7 Take-home rules for using Landau–Zener in simulations

- Compute $Γ$ from Eq. (6) for every pair; store $λ$  and $ω_{eff}$ once.
- Replace Marcus prefactor $J^2$  by $\[1-\exp(-2\pi Γ)]/(\hbar ω_{eff} λ)$
- Ensure units: $J,λ$  in eV, $ω_{eff}$ in eV (1 cm⁻¹ = 1.2398×10⁻⁴ eV).
- Use $Γ$-dependent rate (7) in your kinetic Monte-Carlo or master equation; this prevents the unphysical “infinitely fast” hopping at very large $J$.

## e⁻ / h⁺ Carrier-Specific Effective Frequency

The carrier-specific effective frequency for an electron (anion) or a hole (cation) polaron in an organic semiconductor is defined as:

$$
\hbar \, \Omega_{\text{eff}}^{(e/h)} = \frac{\sum_i S_i^{(e/h)} \, \hbar \omega_i}{\sum_i S_i^{(e/h)}}
$$

---

## 1 Theory Recap — Why This $\Omega_{\text{eff}}$ Works

### Huang–Rhys Factor

$$
S_i = \frac{\Delta Q_i^2 \, \omega_i}{2\hbar}
$$

where $\Delta Q_i$ is the **mass-weighted displacement** between equilibrium geometries along normal mode $i$ of the initial state.

### Intramolecular Reorganisation Energy

$$
\lambda_{\text{intra}} = \sum_i S_i \, \hbar \omega_i
$$

### Single-Mode Mapping

Replacing the full set of modes by a single “effective” Einstein mode $(\tilde{\omega}, \tilde{S})$ must preserve both $\lambda$ and $S$:

$$
\tilde{S} = \sum_i S_i, \quad \lambda = \tilde{S} \, \hbar \tilde{\omega}
$$

Which implies:

$$
\tilde{\omega} = \frac{\sum_i S_i \, \omega_i}{\sum_i S_i}
$$

ℏ$\tilde{\omega}$ expressed in **electron-volts** is what your **KMC code expects as `OMEGA_eff_eV`**. The procedure is identical for an electron (anion) and a hole (cation); the **normal-mode data just come from different charge states**.

---

## 2 ORCA Calculations

Below, `"XYZ"` refers to the Cartesian coordinates section of the ORCA input.

| **Task**                 | **ORCA Input Header**                                  | **Notes**                                                                                  |
|--------------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Neutral geometry + freq  | `! wb97x-d3 def2-TZVP TightSCF Opt Freq`              | Any DFT/dispersion level you trust for π-systems is fine. Keep identical orientation to avoid spurious rotations in the Duschinsky step. |
| Cation (hole)            | `! wb97x-d3 def2-TZVP TightSCF Opt Freq` <br> `* xyz +1 2 …` | Charge = +1, multiplicity = 2 (singlet background). |
| Anion (electron)         | `* xyz -1 2`                                           | Charge = –1. |

Each job writes a `.hess` file (e.g. `neutral.hess`, `cation.hess`, etc.), which contains the normal mode and displacement data required for computing $\Omega_{\text{eff}}$.


## 2 Feed Each Pair of Hessians to the ESD Module

The **Excited-State Dynamics module** (`ORCA_ESD`) produces all **Duschinsky data**, **Huang–Rhys factors** $S_i$, and **per-mode reorganisation energies** in a single run when you set:

```txt
! ESD
%esd
   HESS1  neutral.hess
   HESS2  cation.hess
   USEJ   TRUE          # build J and K
   PRINTLEVEL 3         # list S_i, λ_i etc. :contentReference[oaicite:1]{index=1}
end
* xyzfile 0 1 neutral.xyz   # just to give ESD the equilibrium geom.
```

Repeat with anion.hess for the electron.

Running orca esd_hole.inp (or esd_elec.inp) generates an output block
“Geometry rotation and Duschinsky matrices” and an auxiliary file
esd_duschinsky.dat that contains, for every vibrational mode i of the initial state.

## 3 Compute Ω<sub>eff</sub> from the ESD Output

A tight 15-line Python helper converts that table to Ω<sub>eff</sub> (in eV)  
plus total Huang–Rhys strength ΣS and intramolecular λ:

```python
import numpy as np, re, sys, pathlib

def omega_eff_esd(datfile: pathlib.Path):
    ω_cm, S = [], []
    for line in datfile.read_text().splitlines():
        m = re.match(r'\s*\d+\s+([\d.]+)\s+\S+\s+\S+\s+([\d.]+)', line)
        if m:                       # columns: ω_i  …  S_i
            ω_cm.append(float(m.group(1)))
            S.append(float(m.group(2)))
    ω = np.array(ω_cm) * 2*np.pi*2.99792458e10   # (rad s⁻¹)
    S = np.array(S)
    hbar, e = 1.054571817e-34, 1.602176634e-19
    Ω_eff_eV = hbar * (S*ω).sum()/S.sum() / e
    λ_eV     = hbar * (S*ω).sum() / e           # should equal Σλ_i
    return Ω_eff_eV, S.sum(), λ_eV

for lab in ["hole", "electron"]:
    Ω, ΣS, λ = omega_eff_esd(pathlib.Path(f"esd_{lab}/esd_duschinsky.dat"))
    print(f"{lab:8s}:  Ω_eff = {Ω*1e3:.2f} meV   ΣS = {ΣS:.2f}   λ = {λ*1e3:.1f} meV")
```

#  Marcus × Landau–Zener rate with energy near degeneracy. 



```python
"""
mobility_two_level_uniform_pairs_FAST.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accelerated version of your zero-field mobility KMC with:
  • Numba JIT + SIMD, parallelised over trajectories
  • controlled BLAS thread count (LOKY_MAX_CPU_COUNT = 10)
  • *Optional* single-channel (HOMO–HOMO) mode ― set ONLY_J_HH = True

Apart from the compile-time flag, the scientific model is **identical**
to mobility_two_level_uniform_pairs_RIGOROUS.py.
"""

from __future__ import annotations
from pathlib import Path
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.stats as st
from sklearn.mixture import GaussianMixture
from numba import njit, prange

# ─────────────── User-tunable parameters ──────────────────────────────
FNAME           = Path("e-coupling_donor.txt")  # coupling file
USE_GEOM_FILTER = True                          # distance < 0.9 nm
ONLY_J_HH       = False                     # ← set True for J_HH-only mode
N_GMM_COMPONENTS = 2
N_RUNS          = 500_000
N_HOPS_PER_RUN  = 20_000
T_K             = 300.0
lambda_eV       = 0.05975
sigma_site_eV   = 0.077
OMEGA_eff_eV    = 0.050
DELTA_MEAN      = 0.07347
SEED            = None
# ───────────────────────────────────────────────────────────────────────

# control external BLAS thread explosion (scikit-learn, SciPy)
os.environ["LOKY_MAX_CPU_COUNT"] = "10"

# ─────────── Physical constants (SI or eV units where noted) ──────────
kB_eV_per_K = 8.617_333_262e-5
kB_J_per_K  = 1.380_649e-23
hbar_eV_s   = 6.582_119_569e-16
e_C         = 1.602_176_634e-19

prefactor = 2 * np.pi / hbar_eV_s
kBT_eV    = kB_eV_per_K * T_K
sqrt_term = np.sqrt(4 * np.pi * lambda_eV * kBT_eV)

# ───────── 1. read coupling file (+ optional geometric filter) ────────
cols = ["Mol_i", "Mol_j", "HOMO", "HOMO-1",
        "HOMO-1HOMO", "HOMOHOMO-1", "Distance_nm"]
df = pd.read_csv(FNAME, delim_whitespace=True, names=cols, header=0)
df = df.dropna(subset=cols[2:])

x  = df["Distance_nm"].to_numpy()
# y = df["Angle_deg"].to_numpy()

# expr = 67.7094*(x-0.683595)**2-2*-0.154972*(x-0.683595)*(y-45.2532) +0.00216742*(y-45.2532)**2

if USE_GEOM_FILTER:
    df = df[(x <=0.9)|(x < 0.5)].reset_index(drop=True)
if df.empty:
    raise RuntimeError("No couplings left after filtering.")

# ───────── 2. static disorder bookkeeping (molecule index map) ────────
mol_ids = np.union1d(df["Mol_i"].unique(), df["Mol_j"].unique())
mol2idx = {m: i for i, m in enumerate(mol_ids)}
N_mols  = len(mol_ids)

# =========================  NUMBA-FRIENDLY DATA  ======================
n_rows = len(df)
mol_i_idx = df["Mol_i"].map(mol2idx).astype(np.int32).values
mol_j_idx = df["Mol_j"].map(mol2idx).astype(np.int32).values
J_HH      = df["HOMO"].values.astype(np.float64)
J_LL      = df["HOMO-1"].values.astype(np.float64)
J_LH      = df["HOMO-1HOMO"].values.astype(np.float64)
J_HL      = df["HOMOHOMO-1"].values.astype(np.float64)
dist_m    = (df["Distance_nm"].values * 1e-9).astype(np.float64)

couplings = (mol_i_idx, mol_j_idx, J_HH, J_LL, J_LH, J_HL, dist_m)

rng_global = np.random.default_rng(SEED)
seeds = rng_global.integers(0, 2**32 - 1, size=N_RUNS, dtype=np.uint32)

# =========================  NUMBA kernels  ============================

@njit(fastmath=True, cache=True)
def rate_LZ_numba(J, dG):
    """Marcus × Landau–Zener rate (eV units)."""
    k_na  = prefactor * J * J / sqrt_term * np.exp(-(dG + lambda_eV) ** 2 /
                                                   (4 * lambda_eV * kBT_eV))
    Gamma = J * J / (hbar_eV_s * OMEGA_eff_eV * lambda_eV)
    return k_na * (1.0 - np.exp(-2.0 * np.pi * Gamma))


@njit(parallel=True, fastmath=True, cache=True)
def kmc_ensemble(seeds, N_runs, N_hops, N_mols,
                 sigma_site, delta_mean, couplings,
                 hh_only):

    mol_i_idx, mol_j_idx, J_HH, J_LL, J_LH, J_HL, dist_m = couplings
    n_rows = mol_i_idx.size
    mobilities = np.empty(N_runs, dtype=np.float64)

    # Boltzmann occupancy of L vs H
    p_L = 1.0 / (1.0 + np.exp(-delta_mean / kBT_eV))

    for run in prange(N_runs):

        np.random.seed(seeds[run])                # one seed per trajectory

        # static disorder for this trajectory
        eps_H = np.random.normal(0, sigma_site, N_mols)
        eps_L = np.random.normal(0, sigma_site, N_mols)

        # random initial molecule and electronic level
        posx = posy = posz = 0.0
        t_tot = 0.0
        level = 1 if np.random.random() < p_L else 0   # 1 = L, 0 = H

        for _ in range(N_hops):

            r  = np.random.randint(n_rows)
            i  = mol_i_idx[r]
            j  = mol_j_idx[r]

            # driving forces
            dG_HH = eps_H[j] - eps_H[i]
            dG_LL = eps_L[j] - eps_L[i]
            dG_LH = eps_H[j] - eps_L[i]   # L → H
            # no H → L allowed in this model

            # rates
            k_HH = rate_LZ_numba(J_HH[r], dG_HH)
            if hh_only:
                if level == 1:                          # cannot hop from L
                    continue
                Ktot = k_HH
            else:
                k_LL = rate_LZ_numba(J_LL[r], dG_LL)
                k_LH = rate_LZ_numba(J_LH[r], dG_LH)

                if level == 0:                          # H origin
                    Ktot = k_HH                         # H → H only
                else:                                   # L origin
                    Ktot = k_LL + k_LH                  # L → L/H

            if Ktot <= 0.0 or not np.isfinite(Ktot):
                continue

            # waiting time
            t_tot += np.random.exponential(1.0 / Ktot)

            # choose channel & update level
            u = np.random.random()
            if level == 0:                              # H origin
                # only H → H
                level = 0
            else:                                       # L origin
                if u < k_LL / Ktot:
                    level = 1                           # stay on L
                else:
                    level = 0                           # hop to H

            # isotropic step of length dist_m[r]
            v0, v1, v2 = np.random.normal(), np.random.normal(), np.random.normal()
            norm = 1.0 / np.sqrt(v0*v0 + v1*v1 + v2*v2)
            step = dist_m[r] * norm
            posx += v0 * step
            posy += v1 * step
            posz += v2 * step

        # Einstein relation
        if t_tot == 0.0:
            mobilities[run] = np.nan
        else:
            D = (posx*posx + posy*posy + posz*posz) / (6.0 * t_tot)
            mobilities[run] = e_C * D / (kB_J_per_K * T_K) * 1e4  # cm² V⁻¹ s⁻¹

    return mobilities


# ─────────── 5. ensemble simulation ───────────────────────────────────
mobilities = kmc_ensemble(seeds, N_RUNS, N_HOPS_PER_RUN, N_mols,
                          sigma_site_eV, DELTA_MEAN, couplings,
                          ONLY_J_HH)

mobilities = mobilities[np.isfinite(mobilities)]
np.savetxt("mobilities.txt", mobilities, fmt="%.6e",
           header="zero-field mobilities (cm^2 V^-1 s^-1)")

# ─────────── 6. log-normal (mixture) fit & console summary ────────────
lnμ = np.log(mobilities)[:, None]

if N_GMM_COMPONENTS == 1:
    # analytical MLE for a single log-normal
    μ, σ = lnμ.mean(), lnμ.std(ddof=0)
    p = 1.0                                        # weight of the (only) component
    print("-----------------------------------------------------------")
    print(f"Valid simulations        : {len(mobilities):,} / {N_RUNS:,}")
    print("\nSingle-component fit:")
    print(f"  μ, σ                   = {μ:.4f}, {σ:.4f}")
    print(f"  mode  = {np.exp(μ-σ**2):.3e}")
    print(f"  median= {np.exp(μ):.3e}")
    print(f"\nMean ⟨μ⟩                 = {np.exp(μ+0.5*σ**2):.3e} cm² V⁻¹ s⁻¹")
    print("-----------------------------------------------------------")
else:
    gmm = GaussianMixture(2, covariance_type='full', random_state=0).fit(lnμ)
    w, mus = gmm.weights_.ravel(), gmm.means_.ravel()
    sigmas = np.sqrt(gmm.covariances_.ravel())
    order  = np.argsort(mus)               # enforce μ₁<μ₂
    p      = w[order][0]
    μ1, μ2 = mus[order];  σ1, σ2 = sigmas[order]
    mode1, mode2   = np.exp(μ1-σ1**2), np.exp(μ2-σ2**2)
    median1,median2= np.exp(μ1),       np.exp(μ2)
    mean_mix = (w * np.exp(mus + 0.5*sigmas**2)).sum()

    print("-----------------------------------------------------------")
    print(f"Valid simulations        : {len(mobilities):,} / {N_RUNS:,}")
    print("\nMixture parameters (μ₁<μ₂):")
    print(f"  weight p               = {p:.3f}")
    print(f"  μ₁, σ₁                 = {μ1:.4f}, {σ1:.4f}")
    print(f"  μ₂, σ₂                 = {μ2:.4f}, {σ2:.4f}")
    print("\nPer-component stats:")
    print(f"  mode₁ = {mode1:.3e}   median₁ = {median1:.3e}")
    print(f"  mode₂ = {mode2:.3e}   median₂ = {median2:.3e}")
    print(f"\nMixture mean ⟨μ⟩         = {mean_mix:.3e} cm² V⁻¹ s⁻¹")
    print("-----------------------------------------------------------")

# ─────────── 7. histogram + fitted PDF(s) (log–log) ───────────────────
fig, ax = plt.subplots(figsize=(8, 6))
xlims = (np.log10(mobilities.min()), np.log10(mobilities.max()))
bins  = np.logspace(*xlims, 200)
n, edges, _ = ax.hist(mobilities, bins=bins, density=True, alpha=0.65)

x = np.logspace(*xlims, 600)

if N_GMM_COMPONENTS == 1:
    pdf = st.lognorm.pdf(x, s=σ, loc=0, scale=np.exp(μ))
    ax.plot(x, pdf, lw=2, label="single log-normal")
else:
    pdf1 = p    * st.lognorm.pdf(x, s=σ1, loc=0, scale=np.exp(μ1))
    pdf2 = (1-p)* st.lognorm.pdf(x, s=σ2, loc=0, scale=np.exp(μ2))
    ax.plot(x, pdf1, lw=2, label="component 1")
    ax.plot(x, pdf2, lw=2, label="component 2")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Zero-field mobility μ₀ (cm² V⁻¹ s⁻¹)", fontsize=14)
ax.set_ylabel("Probability density",            fontsize=14)
title = "Histogram + " + ("single log-normal fit"
                          if N_GMM_COMPONENTS == 1
                          else "two-log-normal fit")
ax.set_title(title, fontsize=16, pad=8)
ax.set_ylim(1e-5, None)
ax.legend(fontsize=11)
plt.tight_layout(); plt.show()

# Save histogram data (bin centres + densities)
np.savetxt("mobility_histogram.txt",
           np.column_stack([0.5*(edges[:-1]+edges[1:]), n]),
           header="bin_center  density", fmt="%.6e")
```
