"""
Zero-field mobility simulation + single-log-normal fit (Numba accelerated)

Outputs
-------
• Log-normal parameters (μ̂, σ̂)
• Mode, arithmetic mean, median of the fitted pdf
• Histogram + fitted pdf
"""
from __future__ import annotations
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "10"   # silence joblib on Win
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as st
from numba import njit, prange

# ----------------------------------------------------------------------
# User parameters
# ----------------------------------------------------------------------
FNAME             = Path("e-coupling_with_nn_distance.txt")
N_RUNS            = 100_000
N_HOPS_PER_RUN    = 100_000
T_K               = 300.0
lambda_eV         = 0.411
disorder_sigma_eV = 0.098
SEED              = None            # or an int for reproducibility

# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
kB_eV_per_K = 8.617_333_262e-5
kB_J_per_K  = 1.380_649e-23
hbar_eV_s   = 6.582_119_569e-16
e_C         = 1.602_176_634e-19

# ----------------------------------------------------------------------
# Load coupling file
# ----------------------------------------------------------------------
cols = ["Mol_i", "Mol_j", "J_eV", "distance_nm", "angle_deg"]
df = (pd.read_csv(FNAME, delim_whitespace=True, header=0, names=cols)
        .dropna()[["J_eV", "distance_nm"]])
if df.empty:
    raise RuntimeError("No usable rows in the coupling file.")

J_eV_arr   = df["J_eV"].to_numpy(np.float64)
dist_m_arr = df["distance_nm"].to_numpy(np.float64) * 1e-9

# ----------------------------------------------------------------------
# Marcus prefactors
# ----------------------------------------------------------------------
prefactor = 2.0 * np.pi / hbar_eV_s
kBT_eV    = kB_eV_per_K * T_K
sqrt_term = np.sqrt(4.0 * np.pi * lambda_eV * kBT_eV)

# ----------------------------------------------------------------------
# Numba-compiled simulator
# ----------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def run_ensemble(J, d, n_runs, n_hops, lam, kBT, pref, sqrt_trm,
                 sigma_dis, kB_J, T, e):
    mob = np.empty(n_runs, dtype=np.float64)

    for r in prange(n_runs):
        x = y = z = 0.0
        t_tot = 0.0
        for _ in range(n_hops):
            i      = np.random.randint(J.size)
            J_eV   = J[i]
            dist_m = d[i]
            dG_eV  = np.random.normal(0.0, sigma_dis) - np.random.normal(0.0, sigma_dis)

            k_ij = pref * J_eV**2 / sqrt_trm * \
                   np.exp(-(dG_eV + lam)**2 / (4.0 * lam * kBT))
            if k_ij <= 0.0:
                continue

            t_tot += np.random.exponential(1.0 / k_ij)

            vx, vy, vz = np.random.normal(), np.random.normal(), np.random.normal()
            s = dist_m / np.sqrt(vx*vx + vy*vy + vz*vz)
            x += vx*s;  y += vy*s;  z += vz*s

        if t_tot > 0.0:
            r2 = x*x + y*y + z*z
            D  = r2 / (6.0 * t_tot)
            mu_SI = e * D / (kB_J * T)
            mob[r] = mu_SI * 1e4
        else:
            mob[r] = np.nan
    return mob

np.random.seed(SEED)
mobilities = run_ensemble(J_eV_arr, dist_m_arr,
                          N_RUNS, N_HOPS_PER_RUN,
                          lambda_eV, kBT_eV, prefactor, sqrt_term,
                          disorder_sigma_eV, kB_J_per_K, T_K, e_C)

mobilities = mobilities[np.isfinite(mobilities)]
if mobilities.size == 0:
    raise RuntimeError("All simulations failed; no valid mobilities.")

# ----------------------------------------------------------------------
# Single log-normal fit  (loc fixed at 0)
# ----------------------------------------------------------------------
σ_hat, loc, scale = st.lognorm.fit(mobilities, floc=0)
μ_hat = np.log(scale)          # mean of ln μ
σ_hat = float(σ_hat)           # st.lognorm returns shape as σ

# Derived statistics
mode_logn  = np.exp(μ_hat - σ_hat**2)
mean_logn  = np.exp(μ_hat + 0.5*σ_hat**2)
median_logn= np.exp(μ_hat)

# PDF for plotting
def pdf_logn(x):
    return st.lognorm.pdf(x, s=σ_hat, loc=0, scale=np.exp(μ_hat))

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))        
bins = np.logspace(np.log10(mobilities.min()),
                   np.log10(mobilities.max()), 40)
ax.hist(mobilities, bins=bins, density=True, alpha=0.6, label="simulation", stacked=True)
x = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 600)
ax.plot(x, pdf_logn(x), lw=2, label="log-normal fit")
ax.axvline(mode_logn,   ls="--", c="k",label=f"mode   = {mode_logn:.2e}")
ax.axvline(mean_logn,   ls="-.", c="k",label=f"mean   = {mean_logn:.2e}")
ax.axvline(median_logn, ls=":",  c="k", label=f"median = {median_logn:.2e}")
ax.set_xscale("log")
ax.set_xlabel("Zero-field mobility μ₀ (cm² V⁻¹ s⁻¹)", fontsize=16)
ax.set_ylabel("Probability density",             fontsize=16)
ax.set_title("Histogram + single log-normal fit", fontsize=18, pad=12)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(fontsize=14)
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------
print("-----------------------------------------------------------")
print(f"Simulations (valid)    : {len(mobilities):,} of {N_RUNS:,}")
print("\nLog-normal parameters:")
print(f"  μ̂ (mean of ln μ)     = {μ_hat:.4f}")
print(f"  σ̂ (std  of ln μ)     = {σ_hat:.4f}")
print("\nDerived statistics (fitted pdf):")
print(f"  Mode                 = {mode_logn:.3e} cm² V⁻¹ s⁻¹")
print(f"  Arithmetic mean ⟨μ⟩   = {mean_logn:.3e} cm² V⁻¹ s⁻¹")
print(f"  Median               = {median_logn:.3e} cm² V⁻¹ s⁻¹")
print("-----------------------------------------------------------")
