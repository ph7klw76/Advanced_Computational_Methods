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

| Tier           | Electronic operator(s)                         | Order in perturbation | Vibrational treatment | Symbol in the paper |
|----------------|-----------------------------------------------|------------------------|------------------------|----------------------|
| 1st + Condon   | $\hat{H}_{SOC}$                               | 1st                    | equilibrium (Condon)   | 1st+Condon           |
| 1st + HT       | $\partial \hat{H}_{SOC} / \partial Q$         | 1st                    | HT modulation          | 1st+HT               |
| 2nd + Condon   | ![image](https://github.com/user-attachments/assets/b1880c88-5aee-4dc3-8d2b-05e2850bf85f)
         | 2nd                    | equilibrium            | 2nd+Condon           |
| 2nd + HT       | $\{\hat{H}_{SOC}, \partial \hat{H}_{SOC}/\partial Q\} \times \hat{H}_{nBO}$ | 2nd | HT modulation          | 2nd+HT               |

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
