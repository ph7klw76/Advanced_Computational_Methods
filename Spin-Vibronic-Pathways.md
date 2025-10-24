# Unraveling Spin-Vibronic Pathways in Organic Emitters: A Deep Dive into Non-Adiabatic Spin-Vibronic Coupling (NA-SVC)

![image](https://github.com/user-attachments/assets/66c3cdf0-d475-44ac-bc58-f6398e9ba1d1)

# Organic TADF and OLED materials

Organic TADF and OLED materials achieve high efficiencies only when forbidden singlet–triplet interconversion becomes fast enough to recycle triplet excitons. For heavy-atom-free chromophores the purely electronic spin–orbit matrix element between the first excited singlet and triplet is typically ≤ 1 cm⁻¹, far too small to match experiment; extra vibrational physics is indispensable.

## 1. From classical Marcus to quantum MLJ

The high-frequency molecular vibrations that accompany charge or exciton transfer were incorporated into electron-transfer theory by Jortner, who showed that a Franck–Condon sum over discrete vibrons smoothly bridges thermally activated hopping at room temperature and pure tunnelling at cryogenic temperatures . The resulting MLJ rate expression

![image](https://github.com/user-attachments/assets/af6ba780-c98b-4ec8-8c66-73cdf544fcb2)


retains its form in rigid films; the only change is that the low-frequency, outer-sphere part of the reorganisation energy becomes largely static, effectively shifting the driving force rather than entering the temperature-dependent denominator. Experiments on donor-acceptor pairs embedded in frozen glycerol : methanol (9 : 1) at 255 K confirm the MLJ prediction of a Marcus inverted region even when the matrix is glassy 

## 2. Why static SOC is never enough

A first-order “Condon” calculation uses only the equilibrium SOC,

$$
\langle S_1 \lvert \hat{H}_{SO} \rvert T_1 \rangle,
$$

and therefore underestimates ISC/rISC by several orders of magnitude in purely organic molecules . Vibrational modulation of SOC (the Herzberg–Teller term) already narrows the gap, but many emitters still require an additional pathway: non-adiabatic spin-vibronic coupling.

## 3. What NA-SVC actually is

In second-order perturbation theory the spin flip is mediated by an energetically proximate triplet $T_n$:

![image](https://github.com/user-attachments/assets/ec91ba63-d250-464e-9ae6-e92d35a7869c)


The effective matrix element

![image](https://github.com/user-attachments/assets/634d860b-f43d-4aef-8d00-8f36ddedaacc)


is typically 10–100 × larger than the direct SOC because  
(i) $T_n$ often has a different orbital parentage (e.g. ππ* vs charge-transfer), and  
(ii) the derivative coupling $\langle T_n \lvert \partial/\partial Q \rvert T_1 \rangle$ is large near points where the two triplet potential-energy surfaces approach one another.

Substituting $\lvert H_{\text{NA−SVC}} \rvert^2$ into the MLJ rate integral gives a term that remains active down to at least 80 K, long after classical solvent reorganisation has frozen out; the temperature dependence now rests almost entirely in the quantum-mechanical Franck–Condon factors.

## 4. Empirical validation across temperature and media

**Zeonex and polyethylene-oxide (PEO) matrices**  
Etherington et al. measured the rISC rate of the donor–acceptor emitter DPTZ-DBTO₂ from 30 K to 300 K and showed that only a model including NA-SVC reproduces both the Arrhenius slope above 200 K and the tunnelling plateau below it

**Carbazole derivatives in crystals**  
Sidat and Crespo-Otero computed ISC and rISC for three dichloro-carbazole crystals over 100–300 K. Their MLJ+NA-SVC treatment aligns with experiment within a factor of two, whereas Marcus rates alone fail by an order of magnitude at 150 K 

**Rigid glasses showing Marcus inversion**  
Fluorescence-quenching experiments in glycerol : methanol glass reveal a clear inverted region at 255 K, quantitatively fitted only when the high-frequency FC ladder and NA-SVC are retained 

**Large data set of MR-TADF emitters**  
Hagai et al. benchmarked 121 multi-resonance TADF molecules and found that NA-SVC (with vibrationally modulated SOC included) accounts for ∼90 % of $k_{rISC}$ in the most efficient exemplars .

## 5. Implementing NA-SVC in practice

A reliable workflow now exists. TD-DFT or multi-reference methods supply vertical S₁, T₁, Tₙ energies, SOC matrix elements, and non-adiabatic derivative couplings. The vibrational spectrum is condensed to an effective high-frequency mode (ℏω ≈ 0.16 eV) plus a low-frequency reorganisation term. Packages such as **pySOC2022** and **MultiModeFC** automatically assemble the four perturbative rate components (first/second order × Condon/HT), allowing direct comparison with transient-luminescence experiments.

## 6. Consequences for molecular design

Because NA-SVC is maximised when a “mediator” triplet lies 0.1–0.3 eV above S₁, rational design now centres on tuning that triplet’s energy and orbital character. Rigidifying the π-framework funnels vibrational density into the few modes that both modulate SOC and drive $T_n \leftrightarrow T_1$ mixing, while still keeping the overall singlet–triplet gap small. The success of this strategy in MR-TADF emitters demonstrates that heavy atoms are unnecessary once NA-SVC is intelligently exploited.

## 7. Conclusion

Re-examining the spin-flip problem through the lens of NA-SVC resolves the long-standing discrepancy between bare SOC theory and the remarkably fast ISC/rISC observed in purely organic emitters. By embedding second-order, mediator-assisted spin flips in the quantum-vibrational MLJ framework, one obtains a single, parameter-consistent description that spans fluid solution, rigid glass, and crystalline films from ambient down to liquid-nitrogen temperature. As corroborated by multiple independent laboratories, this approach now sets the quantitative standard for interpreting and engineering the photophysics of next-generation OLED materials.

---

### Key sources

- J. Jortner, J. Chem. Phys. 64, 4860 (1976) 
- M. K. Etherington et al., Nat. Commun. 7, 13680 (2016) 
- T. J. Penfold et al., Chem. Rev. 118, 6975 (2018)  
- M. Hagai et al., Sci. Adv. 10, eadk3219 (2024)  
- A. Sidat et al., Phys. Chem. Chem. Phys. 24, 29437 (2022) 
- Comment on exothermic rate restrictions in rigid glycerol : methanol matrices, J. Phys. Chem. 115, — (2011)



Based on this paper :(https://www.science.org/doi/epdf/10.1126/sciadv.adk3219)

# Reverse intersystem crossing (RISC)

Reverse intersystem crossing (RISC) is the kinetic bottleneck that decides whether a purely organic OLED can recycle its triplet excitons and reach 100 % internal quantum efficiency. The January 2024 Science Advances article by Hagai et al. expands the state-of-the-art theory for RISC in multi-resonant TADF molecules, unifying vibrationally enhanced spin–orbit coupling and indirect spin-flip pathways in a single perturbative framework. Below, we reconstruct—step by step—the logical chain, mathematical derivations, and numerical procedures that underpin their key findings.

## 1 Starting point: Fermi’s golden rule in a vibronic basis

Hagai et al. describe ISC/RISC within second-order time-dependent perturbation theory. The perturbation contains two commuting pieces: a pure electronic spin–orbit operator $\hat{H}_{SOC}$ and a nuclear-kinetic non-Born–Oppenheimer operator $\hat{H}_{nBO}$. Writing vibronic eigenstates as $\lvert i,v \rangle$ and $\lvert f,u \rangle$, the exact golden-rule rate splits naturally into a first-order (direct) term and a second-order (indirect) term. The second-order channel, which couples $\hat{H}_{SOC}$ and $\hat{H}_{nBO}$, is the formal definition of non-adiabatic spin-vibronic coupling (NA-SVC).

## 2 Thermal-vibration correlation functions (TVCF)

To evaluate the golden-rule double sum efficiently, the authors adopt the TVCF formalism of Peng and Shuai. After assuming harmonic potential-energy surfaces, the multi-mode overlap integrals collapse into an analytically known correlation function $\rho_{fi}(t)$. The total rate is recovered through a Fourier transform

$$
k_{fi}(X)(\Delta E_{ST}) = \frac{1}{\hbar^2} \int_{-\infty}^{\infty} dt\, e^{i \Delta E_{ST} t / \hbar} Z_i^{-1} \rho_{fi}(X)(t),
$$

where $X$ labels one of four successively richer approximations (defined below). The harmonic assumption keeps all quantum Franck–Condon factors intact and therefore preserves tunnelling at cryogenic temperature.

## 3 A four-tier hierarchy of spin-vibronic terms

By expanding $\hat{H}_{SOC}$ to first order in each normal coordinate (Herzberg–Teller, HT) and retaining the explicit NA-SVC term, the golden-rule expression decomposes into four additive contributions:

![image](https://github.com/user-attachments/assets/6482399c-c537-41f8-a2b4-c4960f32706b)



The full rate at the 2nd+HT level is the algebraic sum of these four pieces.

## 4 Deriving the 2nd + HT correlation function

Section S7 of the Supplementary Information provides the explicit derivation. The vibronic matrix elements are first Taylor-expanded (Condon + HT), then re-inserted into the golden-rule kernel, yielding four distinct correlation functions $\rho_{fi}^{(X)}(t)$. Each takes the generic form

$$
\rho_{fi}^{(X)}(t) = \rho_{fi}^{core}(t) \times \rho_{fi}^{(X)}(t),
$$

where $\rho^{core}$ encodes Duschinsky mixing and Huang–Rhys displacements, and the $X$-dependent factor inserts the appropriate electronic couplings. Analytical expressions for the deterministic matrices A, B, C, D and displacement vector E appear in Eqs. (11)–(16) of the main text.

## 5 Linking to (and surpassing) Marcus theory

If one  
(i) discards Duschinsky rotations,  
(ii) forces identical mode frequencies in the two electronic states,  
(iii) expands trigonometric factors to second order in time (short-time), and  
(iv) assumes $k_B T \gg \hbar\omega$ (high-T),  

the TVCF integral collapses to the classical Marcus Gaussian. Hagai et al. show algebraically how each approximation shifts $\rho(t)$ and hence $k$ (Eqs. 19–23). In practice these successive degradations can fortuitously cancel, which explains why the simple Marcus rate sometimes agrees with experiment—even though it omits HT-SVC and NA-SVC entirely.

## 6 Practical computation of the couplings

Electronic geometries for S₁ and T₁ are optimised at the PBE0-D3BJ/def2-SV(P) TD-DFT level. Triplet–triplet and singlet–triplet SOC matrix elements employ a ZORA-PBE0 treatment in ORCA, while non-adiabatic couplings are obtained with Q-Chem. Numerical differentiation provides SOC derivatives because analytic HT gradients are unavailable. Vibrational normal modes at both electronic minima feed the TVCF integrator, written in-house, which propagates $\rho(t)$ up to 10 ps with a 0.1 fs step.

## 7 Calibrating the singlet–triplet gap: the ARPSfit scheme

Because $k$ is exponentially sensitive to $\Delta E_{ST}$, the authors introduce an Arrhenius-plot slope fitting (ARPSfit) procedure. They numerically generate $\ln k(1/T)$ curves for trial $\Delta E_{ST}$ values, compute the simulated activation energy $E_a^{Calc}$, and iterate until it matches the experimental slope $E_a^{Exp}$. The resulting $\Delta E_{ST}^{fit}$ is often 0.05–0.07 eV smaller than $E_a^{Exp}$, correcting an over-barrier bias that plagued earlier TADF literature.

## 8 Benchmark on four prototypical MR-TADF emitters

Using $\Delta E_{ST}^{fit}$, the 2nd+HT model predicts $k_{RISC}$ within a factor ≤ 2 for BOBO-Z, BOBS-Z, BSBS-Z, and ν-DABNA, whereas 1st+Condon underestimates by up to three orders of magnitude. The HT term alone raises the rate 10–10³-fold, and NA-SVC via T₂/T₃ adds a further 6× boost for ν-DABNA.

## 9 Large-scale validation and typology of error cancellation

Extending the test set to 121 MR-TADF molecules, the authors show that 1st+Condon correlates with experiment (Pearson $r = 0.51$) but systematically underestimates; Marcus, by contrast, scatters wildly ($r = 0.33$) because its Gaussian kernel is hypersensitive to $\lambda$ and $\Delta E_{ST}$. Data points cluster into three regimes depending on $(\lambda, \Delta E_{ST})$: in one third of cases Marcus grossly underestimates, in another third it is accidentally accurate, and in the remainder it overshoots but error-cancellation masks the defect.

## 10 Consequences for molecular design

The extended 2nd+HT formalism reveals that a low-lying $T_n$ state of contrasting orbital character ($\Delta E_{T_1T_n} \lesssim 0.2$ eV) and sizable SOC to S₁ is the most effective lever for NA-SVC-driven RISC acceleration. At the same time, Herzberg–Teller modulation demands stiff frameworks that channel vibrational density into a few symmetry-allowed modes. With these criteria, heavy atoms become unnecessary for sub-microsecond RISC.

## Closing remarks

By retaining quantum Franck–Condon sums, explicitly coupling SOC derivatives, and embedding indirect NA-SVC pathways, Hagai et al. achieve a single rate expression that reproduces both the temperature dependence and absolute magnitude of RISC across 100 K–300 K, from rigid polymer films to dilute solutions. Their derivations also demystify why the venerable Marcus equation sometimes “works” and, crucially, when it cannot. The article thus sets a new benchmark for predictive modelling and paves the way for rational, heavy-atom-free OLED emitters with near-unity internal quantum efficiencies.

<img width="521" height="82" alt="image" src="https://github.com/user-attachments/assets/117e33d5-3fb2-4529-8fc8-c63323ac1c43" />
 Equation 2
<img width="885" height="253" alt="image" src="https://github.com/user-attachments/assets/de40b402-d7b4-41f3-9fdc-56d3db33b2c3" />
<img width="762" height="158" alt="image" src="https://github.com/user-attachments/assets/3961fb75-8cc5-4771-92a6-4992abee8c57" />
<img width="830" height="216" alt="image" src="https://github.com/user-attachments/assets/a1132187-0fda-4665-91e6-ad6470e21df4" />
<img width="862" height="309" alt="image" src="https://github.com/user-attachments/assets/6ab03d19-3f64-4ff1-85c6-f816a9467706" />
<img width="864" height="500" alt="image" src="https://github.com/user-attachments/assets/699113c5-ead0-4f2b-81a9-74985fa62fb4" />
<img width="837" height="416" alt="image" src="https://github.com/user-attachments/assets/2f2f900a-6422-41bf-9a0d-585c025ae680" />





To calculate the Hamiltonian of the coupling use the code below
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the Eq. 2 effective (³LE-mediated, HT-dominated) spin–vibronic coupling

Inputs:
  - Turbomole-format .hess at the same geometry R used for NAC/SOC
  - NACME text file containing the ETF-corrected
      "CARTESIAN NON-ADIABATIC COUPLINGS  <GS|d/dx|ES>"
    block (x,y,z per atom; units 1/bohr)
  - ORCA ESD(ISC) output or text listing:
      "Reference SOC (Re and Im): ..."  and lines like
      "<<Displacing mode k (+/-)>>  socme = Re, Im"
  - ΔE (eV) for κ_k = ΔE * d^(k)  (optionally a separate denominator)
  - Herzberg–Teller displacement step ΔQ (mass-weighted; default 0.01 bohr√amu)

Outputs:
  - Prints V_eff in eV, cm^-1, and µeV
  - CSV: *_per_mode.csv with frequencies, d^(k), κ_k, g_k, <Q^2>, contribution
  - CSV: *_top30.csv ranked by |contribution|

Usage example:
  python eq2_effective_coupling.py \
    --hess PI7_T1.hess \
    --nac  NAC.txt \
    --soc  P7-3.out \
    --gap  0.087 \
    --dq   0.01 --dq_unit bohr \
    --soc_units hartree \
    --out_prefix results/my_system
"""

import re
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import sys

HA_TO_EV = 27.211386245988
EV_TO_HA = 1 / HA_TO_EV
BOHR_TO_ANG = 0.529177210903
CM_TO_HZ = 2.997_924_58e10  # cm^-1 -> Hz
HBAR = 1.054_571_817e-34    # J*s
KB = 1.380_649e-23          # J/K
AMU_KG = 1.660_539_06660e-27
A_M = 1e-10

@dataclass
class HessData:
    masses_amu: np.ndarray
    modes: np.ndarray       # (3N, 3N): mass-weighted eigenvectors (columns)
    freqs_cm: np.ndarray    # (3N,)

def parse_hess(path: str) -> HessData:
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()

    def extract_section(lines, start_label):
        start = None
        for i, l in enumerate(lines):
            if l.strip() == start_label:
                start = i + 1
                break
        if start is None:
            return None
        end = len(lines)
        for i in range(start, len(lines)):
            if lines[i].startswith('$') and i != start - 1:
                end = i
                break
        return [l.rstrip('\n') for l in lines[start:end]]

    atoms_lines = extract_section(lines, '$atoms')
    if atoms_lines is None:
        raise RuntimeError("Could not find $atoms section in .hess")
    n_atoms = int(atoms_lines[0].strip())
    masses_amu = []
    for i in range(1, 1 + n_atoms):
        parts = atoms_lines[i].split()
        masses_amu.append(float(parts[1]))
    masses_amu = np.array(masses_amu)

    freq_lines = extract_section(lines, '$vibrational_frequencies')
    if freq_lines is None:
        raise RuntimeError("Could not find $vibrational_frequencies in .hess")
    nfreq = int(freq_lines[0].strip())
    freqs_cm = np.zeros(nfreq)
    for i in range(1, 1 + nfreq):
        parts = freq_lines[i].split()
        idx = int(parts[0])
        freqs_cm[idx] = float(parts[1])

    nm_lines = extract_section(lines, '$normal_modes')
    if nm_lines is None:
        raise RuntimeError("Could not find $normal_modes in .hess")
    nrows, ncols = map(int, nm_lines[0].split())
    U = np.zeros((nrows, ncols))
    i = 1
    while i < len(nm_lines):
        header = nm_lines[i].strip()
        if (not header) or header.startswith('#'):
            break
        cols = [int(s) for s in header.split()]
        i += 1
        rows_read = 0
        while i < len(nm_lines) and rows_read < nrows:
            line = nm_lines[i].strip()
            if (not line) or line.startswith('#'):
                break
            parts = line.split()
            try:
                row_idx = int(parts[0])
            except ValueError:
                break
            for j, c in enumerate(cols):
                if 1 + j < len(parts):
                    val = float(parts[1 + j].replace('D', 'E'))
                    U[row_idx, c] = val
            i += 1
            rows_read += 1
    return HessData(masses_amu=masses_amu, modes=U, freqs_cm=freqs_cm)

def parse_nac_cart(path: str) -> np.ndarray:
    """
    Parse lines like:
      '  1   C   :   -0.289474245    0.115032465   -0.035571352'
    Returns flat array of length 3N (x,y,z for each atom), units 1/bohr.
    """
    vals: List[float] = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if ':' not in line:
                continue
            parts = line.replace(':', ' ').split()
            if len(parts) < 5:
                continue
            try:
                xs = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
                vals.extend(xs)
            except Exception:
                continue
    if not vals:
        raise RuntimeError("No NAC Cartesian components found in NAC file")
    return np.array(vals, dtype=float)

@dataclass
class SOCData:
    soc0_re: float
    soc0_im: float
    plus: Dict[int, float]   # mode_index(1-based) -> Im(SOC(+ΔQ))
    minus: Dict[int, float]  # mode_index(1-based) -> Im(SOC(-ΔQ))

def parse_socme(path: str) -> SOCData:
    """
    Robust line parser for ORCA ESD(ISC) SOCME output.
    Accepts:
      - 'Reference SOC (Re and Im ...): re, im'
      - '<<Displacing mode k (+/-)>>  socme = re, im'
    Stores the *imaginary* part per displaced mode, since that's what couples singlet-triplet.
    """
    soc0_re = 0.0
    soc0_im = 0.0
    plus: Dict[int, float] = {}
    minus: Dict[int, float] = {}

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            L = line.strip()
            # Reference line
            if L.lower().startswith("reference soc") or L.lower().startswith("reference socme"):
                # e.g. 'Reference SOC (Re and Im): 0.000000e+00, -1.392555e-08'
                if ":" in L:
                    right = L.split(":", 1)[1]
                    toks = [t.strip() for t in right.split(",")]
                    if len(toks) >= 2:
                        try:
                            soc0_re = float(toks[0]); soc0_im = float(toks[1])
                        except Exception:
                            pass
            # Displacing mode lines
            if "Displacing mode" in L and "socme" in L.lower():
                # Extract index and sign
                m = re.search(r'Displacing\s+mode\s+(\d+)\s*\(', L)
                if not m:
                    continue
                idx = int(m.group(1))
                sign = '+' if '(+)' in L else ('-' if '(-)' in L else '?')
                # Extract re, im after 'socme ='
                try:
                    after = L.split("socme", 1)[1].split("=")[1]
                    re_im = [t.strip() for t in after.split(",")]
                    re_part = float(re_im[0]); im_part = float(re_im[1])
                except Exception:
                    continue
                if sign == '+':
                    plus[idx] = im_part
                elif sign == '-':
                    minus[idx] = im_part

    if not plus and not minus:
        raise RuntimeError("Could not parse any displaced SOCMEs from file")

    return SOCData(soc0_re=soc0_re, soc0_im=soc0_im, plus=plus, minus=minus)

def thermal_Q2(freq_cm: np.ndarray, T: float) -> np.ndarray:
    """
    <Q^2> for mass-weighted normal coordinate at temperature T.
    Returned in amu*Å^2.
    """
    omega = 2 * math.pi * CM_TO_HZ * freq_cm
    x = HBAR * omega / (2 * KB * T)
    coth = np.cosh(x) / np.sinh(x)
    Q2_SI = (HBAR / (2 * omega)) * coth  # kg*m^2
    return Q2_SI / (AMU_KG * A_M**2)     # amu Å^2

def compute_eq2(hess_path: str, nac_path: str, soc_path: str,
                delta_e_eV: float, delta_e_den_eV: Optional[float] = None,
                delta_Q: Optional[float] = None, delta_Q_unit: str = 'bohr',
                temperature: float = 298.15, soc_units: str = 'hartree'):
    """
    Returns: (V_eff in eV, per-mode DataFrame, ranked DataFrame)
    """
    if delta_e_den_eV is None:
        delta_e_den_eV = delta_e_eV
    if delta_Q is None:
        delta_Q = 0.01
        delta_Q_unit = 'bohr'
    if delta_Q_unit.lower().startswith('bohr'):
        delta_Q_A = delta_Q * BOHR_TO_ANG
    else:
        delta_Q_A = delta_Q

    # Parse Hessian / modes / masses
    hd = parse_hess(hess_path)
    masses = hd.masses_amu
    U = hd.modes
    freqs_cm = hd.freqs_cm

    # Parse NACME (Cartesian)
    nac_cart = parse_nac_cart(nac_path)  # length 3N
    if len(nac_cart) != 3 * len(masses):
        raise RuntimeError(f"NAC length {len(nac_cart)} != 3N ({3*len(masses)})")

    # Mass-weighted NAC
    mass_per_cart = np.repeat(masses, 3)
    d_mw = nac_cart / np.sqrt(mass_per_cart)   # 1/(bohr*sqrt(amu))

    # Project onto vibrational modes (skip 6 TR)
    start_mode = 6
    U_vib = U[:, start_mode:]
    d_k = U_vib.T @ d_mw  # (3N-6,)

    # κ_k in eV/(Å√amu)
    kappa_eV_per_A = (delta_e_eV * d_k) / BOHR_TO_ANG

    # Thermal <Q^2> in amu Å^2
    freq_vib = freqs_cm[start_mode:]
    Q2 = thermal_Q2(freq_vib, temperature)

    # SOCME (+/-) => s_k = g_k * ΔQ
    soc = parse_socme(soc_path)

    def soc_to_eV(x):
        return x * HA_TO_EV if soc_units.lower().startswith('hart') else x

    s_k = np.zeros_like(d_k)
    n_vib = len(d_k)
    for k in range(1, n_vib + 1):
        if k in soc.minus:
            s_k[k - 1] = 0.5 * (soc_to_eV(soc.plus.get(k, soc.soc0_im)) - soc_to_eV(soc.minus[k]))
        else:
            s_k[k - 1] = soc_to_eV(soc.plus.get(k, soc.soc0_im)) - soc_to_eV(soc.soc0_im)

    # g_k = s_k / ΔQ   (units: eV/(Å√amu))
    g_k = s_k / delta_Q_A

    # Per-mode contribution to numerator (eV^2): g_k * κ_k * <Q^2>
    contrib_k_eV2 = g_k * kappa_eV_per_A * Q2

    numerator_eV2 = float(np.sum(contrib_k_eV2))
    V_eff_eV = numerator_eV2 / delta_e_den_eV

    df = pd.DataFrame({
        "vib_index_1based": np.arange(1, n_vib + 1),
        "global_mode_index": np.arange(start_mode, start_mode + n_vib),
        "frequency_cm^-1": freq_vib,
        "d_k_1/(bohr*sqrt(amu))": d_k,
        "kappa_eV_per_A_sqrtamu": kappa_eV_per_A,
        "s_k_eV": s_k,
        "g_k_eV_per_A_sqrtamu": g_k,
        "<Q^2>_amu_A^2_(T)": Q2,
        "contribution_eV^2": contrib_k_eV2
    })
    df_rank = df.reindex(df["contribution_eV^2"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return V_eff_eV, df, df_rank

def main():
    ap = argparse.ArgumentParser(description="Compute Eq. 2 effective coupling from .hess, NACME, and SOCME files.")
    ap.add_argument("--hess", required=True, help="Turbomole-format .hess at geometry R")
    ap.add_argument("--nac", required=True, help="NACME text (CARTESIAN NON-ADIABATIC COUPLINGS block)")
    ap.add_argument("--soc", required=True, help="ORCA ESD(ISC) .out or text with Reference SOC and Displacing mode k")
    ap.add_argument("--gap", type=float, required=True, help="ΔE (eV) for κ_k = ΔE * d^(k)")
    ap.add_argument("--den", type=float, default=None, help="Denominator ΔE_TT (eV) in Eq. 2 (default = --gap)")
    ap.add_argument("--dq", type=float, default=None, help="HT displacement step ΔQ (mass-weighted). Default 0.01 bohr√amu if omitted.")
    ap.add_argument("--dq_unit", choices=["bohr", "ang"], default="bohr", help="Unit for ΔQ (bohr or ang for Å)")
    ap.add_argument("--T", type=float, default=298.15, help="Temperature (K) for <Q^2>")
    ap.add_argument("--soc_units", choices=["hartree", "eV"], default="hartree", help="SOCME unit in file (ORCA prints Hartree)")
    ap.add_argument("--out_prefix", default="eq2_results", help="Prefix for output CSVs")

    args = ap.parse_args()

    Veff, df, df_rank = compute_eq2(args.hess, args.nac, args.soc,
                                    delta_e_eV=args.gap, delta_e_den_eV=args.den,
                                    delta_Q=args.dq, delta_Q_unit=args.dq_unit,
                                    temperature=args.T, soc_units=args.soc_units)
    print(f"Effective coupling V_eff = {Veff:.6e} eV")
    print(f"               V_eff = {Veff*8065.544006:.6e} cm^-1")
    print(f"               V_eff = {Veff*1e6:.6f} µeV")

    df.to_csv(f"{args.out_prefix}_per_mode.csv", index=False)
    df_rank.head(30).to_csv(f"{args.out_prefix}_top30.csv", index=False)
    print(f"Wrote: {args.out_prefix}_per_mode.csv and {args.out_prefix}_top30.csv")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage example:")
        print("  python eq2_effective_coupling.py --hess PI7_T1.hess --nac NAC.txt --soc P7-3.out --gap 0.087 --dq 0.01 --dq_unit bohr --soc_units hartree")
        sys.exit(0)
    main()

```

python eq2_effective_coupling.py --hess PI7_T1.hess --nac NAC.txt --soc P7-3.out --gap 0.087 --dq 0.01 --dq_unit bohr --soc_units hartree --out_prefix results

The gap is the 3LE-3CT gap ontained by relaxing 3LE and get the gap based on struture.

PI7.T1,hess is the T1 opt freq
the soc is the ISC spin-vibronic output per mode using ISC moduie

eg
```text
! def2-SVP ESD(ISC) CPCM(Toluene) RIJCOSX 
%scf
  moinp "PI7.gbw" #relexedT1
end
%TDDFT NROOTS 3
       SROOT 1
       TROOT 2
       TROOTSSL 0
       DOSOC TRUE
END
%basis
        AuxJ  "Def2/J" 
End
%scf
  MaxIter 300
end
%rel
  SOCType 3
  SOCFlags 1,3,3,0     # RI-SOMF(1X): 1e term on; Coulomb via RI; one-center exact exchange; no corr
  SOCMaxCenter 4
end
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
        ExtParamXC "_omega" 0.06924171915036594
END
%ESD
   USEJ TRUE
   DOHT TRUE
   ISCISHESSIAN "PI7-3LE.hess"
   ISCFSHESSIAN "PI7_S1.hess" 
   NPOINTS 327680
   PrintLevel 4
   LINEW      50
   INLINEW    150 
END
%maxcore 4000
%pal nprocs 16 end
* XYZFILE 0 1 PI7.xyz  #Triplet

```

nac.txt is the exyracyed output using

```text
! def2-SVP TIGHTSCF CPCM(Toluene) RIJCOSX 
%TDDFT  NROOTS  2
        IROOT   1 #must be LE
        NACME   TRUE
        ETF     TRUE
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
        ExtParamXC "_omega" 0.06924171915036594
END
%basis
        AuxJ  "Def2/J" 
End
%scf
  MaxIter 300
END
%maxcore 4000
%pal nprocs 16 end
* XYZFILE 0 3 PI7.xyz #Triplet 3CT
```


```text
extract the text of 
"---------------------------------
CARTESIAN NON-ADIABATIC COUPLINGS
          <GS|d/dx|ES>
        with built-in ETFs
---------------------------------

   1   C   :   -0.289474245    0.115032465   -0.035571352
   2   C   :    0.048823373    0.179764159    0.105771721
   3   C   :   -0.689728883    0.137789768   -0.179792337
   4   C   :    0.665269311    0.119418336    0.303792690
   5   C   :   -0.325768797    0.021894936   -0.088295309
   6   C   :    0.203959990   -0.085507589    0.020806385
   7   C   :    0.992782092   -1.871475150   -0.878468100
   8   C   :   -0.508404636   -0.137775144   -0.382451370
   9   C   :    0.387260953    0.341736892    0.434340304
  10   C   :   -0.184520783    0.119487873   -0.027607809
  11   H   :    0.048854056    0.073097287    0.073177837
  12   C   :    0.106606435    0.182938866    0.187693359
  13   C   :   -0.057039524   -0.118552976   -0.112479734
  14   C   :    0.140988376    0.156928691    0.198306559
  15   H   :   -0.018894300    0.009869994    0.005050120
  16   H   :   -0.107099311    0.033027869   -0.025936425
  17   H   :    0.136042709    0.088059961    0.105024430
  18   H   :   -0.037578643   -0.001953270   -0.020410559
  19   H   :    0.002273940    0.017100283    0.012021621
  20   H   :    0.030776093    0.016052524    0.025268256
  21   H   :   -0.173944757    0.000067704   -0.089134639
  22   C   :    0.043792599    0.063178439    0.027106373
  23   C   :    0.108256642   -0.718098521    0.103499284
  24   C   :    0.321475344   -0.410934205   -0.339477982
  25   C   :    0.073657457   -0.145698248   -0.020185820
  26   C   :   -0.223829738    0.177912588    0.224928614
  27   C   :   -0.035291060    0.175907337   -0.046669335
  28   C   :    0.114876632   -0.075455498   -0.137208397
  29   C   :   -0.041223009   -0.232656425    0.165282285
  30   H   :   -0.030585246   -0.029641922    0.057169761
  31   C   :    0.082320491    0.387845423   -0.316961276
  32   C   :   -0.251020403    0.041857226    0.358471058
  33   H   :    0.010802308    0.040728130   -0.038745281
  34   C   :    0.104125329    0.015370978   -0.171162993
  35   C   :   -0.306227966    0.382122996    0.275393668
  36   H   :   -0.001159371   -0.006933714    0.007874395
  37   H   :    0.007830970    0.010339190   -0.016707362
  38   C   :   -0.132158442    0.449985749   -0.017657908
  39   H   :   -0.010053361   -0.005797986    0.018513623
  40   H   :    0.003466383    0.002573203   -0.007170320
  41   H   :   -0.011776465   -0.011262563    0.025790077
  42   H   :    0.012752527    0.011495532   -0.024779567
  43   O   :   -0.513838018    0.936814792    0.436227814
  44   C   :   -0.270627444    0.551033333    0.134530684
  45   C   :   -0.012414031   -0.160441377   -0.085333639
  46   C   :    0.142036326   -0.101087119    0.021049790
  47   C   :    0.055494076   -0.002634241    0.026356676
  48   H   :   -0.021965007    0.007620352   -0.006921469
  49   C   :   -0.034478187   -0.034123594   -0.033967450
  50   H   :    0.008130342    0.019037159    0.013409726
  51   C   :    0.009724104   -0.019093967   -0.004516808
  52   H   :    0.016236918   -0.026004597   -0.004823687
  53   H   :    0.010900111   -0.027189683   -0.008131685
  54   H   :    0.004611606   -0.009151934   -0.002204702
  55   N   :    0.390991121   -0.650351115   -0.242016454

Difference to translation invariance:
           :   -0.0039830144    0.0042691953    0.0020673453

Norm of the NACs                   ...    3.5487975599
RMS NACs                           ...    0.2762735153
MAX NAC                            ...    1.8714751500
```

Triplet state has 3sublevel, wity zero magnetic field, it is degenerate and the marcus equation rate by 3 should be used.




