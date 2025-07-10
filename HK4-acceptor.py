
"""
Zero-field mobility via 3-D kinetic Monte-Carlo
  • neighbour-resolved hops with Landau–Zener correction
  • two-log-normal mixture fit (printed to console only)
  • histogram plotted on log–log axes with component fits
  • raw mobilities saved to mobilities.txt
"""
from __future__ import annotations
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "10"        # silence joblib warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.stats as st
from pathlib import Path
from sklearn.mixture import GaussianMixture
from numba import njit, prange

# ─────────────── user parameters ──────────────────────────────────────
FNAME             = Path("e-coupling_with_nn_distance.txt")
USE_GEOM_FILTER   = True
N_RUNS            = 500_000
N_HOPS_PER_RUN    = 20_000
T_K               = 300.0
lambda_eV         = 0.411
sigma_site_eV     = 0.098
OMEGA_eff_eV      = 0.015          # promoting phonon
SEED              = None
# ───────────────────────────────────────────────────────────────────────

# physical constants
kB_eV_per_K = 8.617_333_262e-5
kB_J_per_K  = 1.380_649e-23
hbar_eV_s   = 6.582_119_569e-16
e_C         = 1.602_176_634e-19

prefactor = 2*np.pi/hbar_eV_s
kBT_eV    = kB_eV_per_K * T_K
sqrt_term = np.sqrt(4*np.pi*lambda_eV*kBT_eV)

# ───────── 1. read coupling file (+ optional filter) ──────────────────
cols = ["Mol_i", "Mol_j", "J_eV", "distance_nm", "angle_deg"]
df = pd.read_csv(FNAME, delim_whitespace=True, header=0, names=cols)
df = df.dropna(subset=["J_eV", "distance_nm", "angle_deg"])

if USE_GEOM_FILTER:
    x, y = df["distance_nm"].to_numpy(), df["angle_deg"].to_numpy()
    expr = (74.7613*(x-0.652847)**2
            + 2*(-0.193896)*(x-0.652847)*(y-42.3702)
            + 0.0025642*(y-42.3702)**2)
    df = df[expr <= 4].reset_index(drop=True)

if df.empty:
    raise RuntimeError("No couplings left after filtering.")

# ───────── 2. build neighbour lists ───────────────────────────────────
site_ids = pd.unique(df[["Mol_i", "Mol_j"]].values.ravel())
site2idx = {m: i for i, m in enumerate(site_ids)}
N_sites  = len(site_ids)

nbr_of, J_of, dist_of = [[] for _ in range(N_sites)], [[] for _ in range(N_sites)], [[] for _ in range(N_sites)]
for row in df.itertuples(index=False):
    i, j = site2idx[row.Mol_i], site2idx[row.Mol_j]
    for a, b in ((i, j), (j, i)):
        nbr_of[a].append(b)
        J_of[a].append(row.J_eV)
        dist_of[a].append(row.distance_nm * 1e-9)

deg = np.array([len(l) for l in nbr_of], dtype=np.int32)
max_deg = deg.max()
def pad(lists, val):
    out = np.full((N_sites, max_deg), val)
    for s, lst in enumerate(lists):
        out[s, :len(lst)] = lst
    return out

nbr_arr  = pad([[j for j in lst] for lst in nbr_of], -1).astype(np.int32)
J_arr    = pad(J_of,   0.).astype(np.float64)
dist_arr = pad(dist_of,0.).astype(np.float64)

# ───────── 3. Landau–Zener kinetic Monte-Carlo kernel ─────────────────
@njit(parallel=True, fastmath=True)
def kmc_LZ(nbr,Jmat,dist,deg,n_sites,
           n_runs,n_hops,lam,kBT,pref,sqrt_trm,
           omega_eff,sigma_site,kB_J,T,e):
    mob = np.empty(n_runs, np.float64)
    eps = np.random.normal(0.0, sigma_site, size=n_sites)
    for r in prange(n_runs):
        s = np.random.randint(n_sites)
        x = y = z = t_tot = 0.0
        for _ in range(n_hops):
            d = deg[s]
            if d == 0: break
            nbrs  = nbr[s, :d]
            Jloc  = Jmat[s, :d]
            distm = dist[s, :d]
            dG    = eps[nbrs] - eps[s]

            k_na = pref*Jloc**2 / sqrt_trm * np.exp(-(dG + lam)**2 / (4*lam*kBT))
            Gamma = Jloc**2 / (hbar_eV_s * omega_eff * lam)
            kvec  = k_na * (1 - np.exp(-2*np.pi*Gamma))

            Ktot = kvec.sum()
            if Ktot <= 0: break

            t_tot += np.random.exponential(1.0 / Ktot)
            rnd, sumn, idx = np.random.rand()*Ktot, 0.0, 0
            while sumn < rnd:
                sumn += kvec[idx]; idx += 1
            idx -= 1
            s_new = nbrs[idx]

            vx, vy, vz = np.random.normal(), np.random.normal(), np.random.normal()
            step = distm[idx] / np.sqrt(vx*vx + vy*vy + vz*vz)
            x += vx*step; y += vy*step; z += vz*step
            s = s_new

        if t_tot > 0:
            D = (x*x + y*y + z*z) / (6*t_tot)
            mob[r] = 1e4 * e * D / (kB_J * T)
        else:
            mob[r] = np.nan
    return mob

# ───────── 4. run simulation & save raw mobilities ────────────────────
np.random.seed(SEED)
mobilities = kmc_LZ(nbr_arr, J_arr, dist_arr, deg, N_sites,
                    N_RUNS, N_HOPS_PER_RUN,
                    lambda_eV, kBT_eV, prefactor, sqrt_term,
                    OMEGA_eff_eV, sigma_site_eV,
                    kB_J_per_K, T_K, e_C)
mobilities = mobilities[np.isfinite(mobilities)]
np.savetxt("mobilities.txt", mobilities, fmt="%.6e",
           header="zero-field mobilities (cm^2 V^-1 s^-1)")

# ───────── 5. two-log-normal mixture fit (console only) ──────────────
lnμ = np.log(mobilities).reshape(-1,1)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(lnμ)
w, mus = gmm.weights_.ravel(), gmm.means_.ravel()
sigmas = np.sqrt(gmm.covariances_.ravel())
order  = np.argsort(mus)
p      = w[order][0]
μ1, μ2 = mus[order]
σ1, σ2 = sigmas[order]
mode1, mode2     = np.exp(μ1-σ1**2), np.exp(μ2-σ2**2)
median1, median2 = np.exp(μ1),        np.exp(μ2)
mean_mix         = (w * np.exp(mus + 0.5*sigmas**2)).sum()

print("-----------------------------------------------------------")
print(f"Simulations (valid)      : {len(mobilities):,} of {N_RUNS:,}")
print("\nMixture parameters (μ₁<μ₂):")
print(f"  weight p               = {p:.3f}")
print(f"  μ₁, σ₁                 = {μ1:.4f}, {σ1:.4f}")
print(f"  μ₂, σ₂                 = {μ2:.4f}, {σ2:.4f}")
print("\nPer-component statistics:")
print(f"  mode₁   = {mode1:.3e}   median₁ = {median1:.3e}")
print(f"  mode₂   = {mode2:.3e}   median₂ = {median2:.3e}")
print(f"\nMixture arithmetic mean ⟨μ⟩ = {mean_mix:.3e} cm² V⁻¹ s⁻¹")
print("-----------------------------------------------------------")

# ───────── 6. histogram + component fits on log–log axes ──────────────
fig, ax = plt.subplots(figsize=(8, 6))
log_bounds = (np.log10(mobilities.min()), np.log10(mobilities.max()))
bins = np.logspace(*log_bounds, 200)

# histogram
n, bin_edges, patches =ax.hist(mobilities, bins=bins, density=True, alpha=0.65)

# x-values for plotting the component pdfs
xplot = np.logspace(*log_bounds, 600)

# weighted component pdfs
pdf1 = p    * st.lognorm.pdf(xplot, s=σ1, loc=0, scale=np.exp(μ1))
pdf2 = (1-p)* st.lognorm.pdf(xplot, s=σ2, loc=0, scale=np.exp(μ2))

ax.plot(xplot, pdf1, lw=2, color='tab:blue',  label="component 1")
ax.plot(xplot, pdf2, lw=2, color='tab:orange',label="component 2")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Zero-field mobility μ₀ (cm² V⁻¹ s⁻¹)", fontsize=15)
ax.set_ylabel("Probability density",            fontsize=15)
ax.set_title("Histogram + two log-normal components", fontsize=17, pad=10)
ax.tick_params(axis='both', labelsize=13)
ax.set_ylim(10e-6,None)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()

bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
data = np.column_stack([bin_centers, n])
np.savetxt("mobility_histogram.txt",data,header="bin_center(cm^2/Vs)    density(1/cm^2/Vs)",fmt="%.6e")
