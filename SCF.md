# ΔSCF Re-Examined  
From the many-electron Schrödinger equation to excited-state density-functional theory

## 1 Prelude: what problem are we solving?

Given the electronic Hamiltonian

<img width="531" height="102" alt="image" src="https://github.com/user-attachments/assets/1d0810a7-2520-4647-b186-920461e24201" />


its exact eigenpairs $\{\Psi_I^N, E_I^N\}$ define vertical observables

$$
I_v = E_0^{N-1} - E_0^N,\quad A_v = E_0^N - E_0^{N+1},\quad \omega_{0 \to K} = E_K^N - E_0^N.
$$

Because Kohn–Sham orbital energies are not eigenvalues of $\hat{H}$, they inherit no variational guarantee.  
The Δ-Self-Consistent-Field (ΔSCF) strategy instead evaluates the exact thermodynamic definition directly: one self-consistent field (SCF) for each state, and a difference of total energies. The rest of this article shows—rigorously and from first principles—why that deceptively simple prescription works.

## 2 Wave-function variational principle for excited states

For any $J$-th excited state, the Hylleraas–Undheim–MacDonald theorem furnishes a constrained stationary principle

$$
\delta \langle \Psi_J^N | \hat{H} | \Psi_J^N \rangle = 0,\quad \langle \Psi_J^N | \Psi_I^N \rangle = \delta_{IJ} \quad (I < J).
$$

Thus a single determinant that is orthogonal to all lower states is legitimate—even exact—if it happens to satisfy the stationary condition.  
ΔSCF enforces the orthogonality indirectly: it locks the occupation pattern (e.g. HOMO → LUMO promotion or ±1 charge) and re-optimises orbitals variationally under that constraint.  
The resulting determinant is a local stationary point of the full-CI functional in the restricted subspace, so its energy is upper-bounded by $E_J^N$.

## 3 Density-functional embedding of the variational idea

### 3.1 Why the Hohenberg–Kohn theorem is insufficient

Electron density $\rho(r)$ uniquely fixes the ground state, but different excited states share the same $\rho(r)$.  
The resolution, proven in 2024 by Yang & Ayers: extend the basic variables to any of three equivalent descriptors of a non-interacting reference system—

- excitation quantum number $n_s$ and potential $w_s(r)$ (nPFT),
- Slater determinant $\Phi$ (ΦFT), or
- one-particle density matrix $\gamma_s(r, r')$ (γ_sFT).

All three yield a universal functional $E[\cdot]$ whose minima reproduce the ground state and whose other stationary points reproduce excited states, thereby giving the formal foundation of ΔSCF inside DFT.  
*arXiv*

### 3.2 Piecewise linearity and derivative discontinuity

Among the exact conditions obeyed by the functional is linear behaviour between integer electron numbers.  
For ground states that is the Perdew–Parr–Levy–Balduz (PPLB) theorem; Yang & Fan (2024) generalised it to every excited state, introducing excited-state chemical potentials $\mu_J^\pm$.  
*arXiv*

ΔSCF evaluates the slope by finite difference, automatically capturing the KS derivative discontinuity $\Delta_{xc}$—the main source of the notorious “band-gap problem”.

## 4 Connecting to photoelectron and absorption spectra

The quantity measured in ultraviolet photoelectron spectroscopy is the Dyson orbital

$$
\phi_D^{(J)}(r) = \sqrt{N} \int dx_2 \dots dx_N\, \Psi_0^{N*}(r, x_2, \dots)\, \Psi_J^{N-1}(x_2, \dots),
$$

whose pole energy is $I_v$.  
Because ΔSCF relaxes every orbital in the N and N–1 determinants separately, it reproduces not only the pole but also the shake-up redistribution in the remaining electrons—an effect absent from frozen-orbital Koopmans estimates.  
The same logic holds for electron affinity and neutral excitations, making ΔSCF state-specific and relaxation-exact.

## 5 Algorithmic realisation

| Step                       | Quantum-mechanical rationale                               | Practical control                                                 |
|---------------------------|-------------------------------------------------------------|-------------------------------------------------------------------|
| Choose occupation constraints | Enforces orthogonality to lower determinants            | Define a target occupation vector                                 |
| Initialise orbitals       | Supplies an initial point in the variational landscape      | MOM / IMOM select orbitals with maximum overlap to the previous cycle |
| Unrestricted vs. restricted | Restores proper spin multiplicity and accounts for spin polarisation | UHF/UDFT for open-shell ions; spin-flip ΔSCF for singlets         |
| Convergence               | Stationary condition of the variational principle           | Tight ΔE < 10⁻⁸ Eₕ and Δρ < 10⁻⁶                                 |
| Upper-bound sanity check  | Variational energies must not exceed those from higher-level wave-function methods for the same state | Benchmark small references (EOM-CCSD, CASPT2) |

Periodic solids: place the promoted electron or core hole in a $k$-resolved Wannier function; apply Makov–Payne or extrapolation for finite-size corrections.

## 6 Accuracy landscape

| Application                        | Typical RMS error                 | Dominant physical error                               |
|-----------------------------------|----------------------------------|--------------------------------------------------------|
| Core-level binding energies (XPS) | 0.2–0.4 eV with hybrid/GGA       | Self-interaction & scalar-relativistic effects         |
| Valence IP/EA of molecules        | ≤0.15 eV with tuned range-separated hybrids | Delocalisation error                       |
| Charge-transfer & Rydberg excitations | TDDFT fails; ΔSCF ≤0.2 eV       | Wrong asymptotic kernel in TDDFT                      |
| Excited-state dipole moments      | ΔSCF beats TDDFT in 85 % of a 2025 benchmark set | Orbital relaxation               |

## 7 Error sources and cures

- **Root flipping**: collapse to ground state → MOM, projection operators.
- **Self-interaction error**: convex deviation from linearity → Koopmans-compliant or range-separated functionals; ΔSCF+GW post-correction.
- **Strong correlation**: single determinant insufficient → ensemble ΔSCF or multireference extensions.
- **State tracking in large manifolds**: combine MOM with machine-learning classifiers to pre-screen viable orbital patterns.

## 8 Frontier developments

Exact excited-state KS equations (Ayers, Giarrusso, Herbert 2023–25) supply blueprints for next-generation XC functionals aimed specifically at ΔSCF.  
*American Chemical Society Publications, AIP Publishing*

Fractional-charge linearity conditions expose excited-state delocalisation error, enabling density-corrected DFAs.  
*arXiv*

Machine-learned surrogates trained on thousands of ΔSCF core-level calculations now deliver <0.1 eV prediction at microsecond cost, opening high-throughput XPS screening pipelines.

## 9 Best-practice checklist (pinned to your monitor)

✔︎ Recommendation  
- Always verify that the targeted determinant remains orthogonal to lower states after convergence.  
- For open-shell ions, inspect ⟨S²⟩; <10 % deviation signals acceptable spin contamination.  
- Use uncontracted core basis + scalar-relativistic Hamiltonian (ZORA, DKH) for 1s–2p spectroscopy.  
- Report both vertical (ΔE | geom frozen) and adiabatic (post-relaxation) gaps; the former map to spectroscopy, the latter to thermodynamics.  
- Benchmark a minimal test set (e.g. CO, H₂O, benzene) with EOM-CCSD to calibrate functional and convergence parameters.  

---

## Take-home message

ΔSCF is not a heuristic but the state-specific application of the Rayleigh–Ritz principle, now embedded in a rigorous excited-state density-functional framework.  
When executed with occupancy constraints, tight convergence, and modern functionals, it yields quasiparticle and excitation energies of the same quality as many-body perturbation theory at a fraction of the cost—while providing uniquely relaxed orbitals indispensable for spectra, dipoles, forces, and machine-learning models.


NAME = minS1.inp
```text

DEF2-SVP OPT CPCM(toluene)  # Opt Singlet excited Geo
%TDDFT  NROOTS  1
        IROOT  1
         IROOTMULT Singlet
|  5> END    
|  6> %method
|  7>         method dft
|  8>         functional HYB_GGA_XC_LRC_WPBEH
|  9>         ExtParamXC "_omega" 0.05175
| 10> END
| 11> %maxcore 3000
| 12> %pal nprocs 32 end
| 13> * XYZFILE 0 1 minS1.xyz
| 14> $new_job
| 15> ! DEF2-SVP CPCM(Toluene) OPT DELTASCF UHF
| 16> %method
| 17>         method dft
| 18>         functional HYB_GGA_XC_LRC_WPBEH
| 19>         ExtParamXC "_omega" 0.05175
| 20> END
| 21> %SCF ALPHACONF 0,1 END
| 22> %output
| 23>  	PrintLevel Huge
| 24> END
| 25> %maxcore 3000
| 26> %pal nprocs 32 end
| 27> * XYZFILE 0 1 minS1.xyz

