# 1 | Photophysics refresher (why we need ns → ms time-resolved PL)

TADF emitters relax through two indistinguishable radiative channels that are separated only by time:

| symbol | process | typical lifetime window | O₂ sensitivity |
|--------|---------|--------------------------|----------------|
| PF     | S₁ → S₀ prompt fluorescence | ≈ 1–50 ns       | negligible     |
| DF     | T₁ → S₁ (RISC) → S₀ delayed fluorescence | ≈ 0.1 µs–10 ms | strongly quenched |

Because both channels emit from the same S₁ state, the energy of every photon is identical; only the arrival time carries information about how many triplets were recycled. A log-time measurement that spans 10⁻⁹–10⁻³ s therefore captures the complete singlet and triplet photon budget  


# 2 | Experimental design that really works

| task        | good practice | why it matters |
|-------------|---------------|----------------|
| Excitation  | ≤ 100 ps pulses, rep-rate ≤ 1 MHz | prevents re-exciting long-lived T₁ states |
| Detection   | 0–100 ns: TCSPC MCP-PMT  
100 ns–30 µs: fast digitiser on PMT  
30 µs–10 ms: gated iCCD or Si-PM | three detectors keep S/N high across 7 orders of magnitude |
| Atmosphere  | pump to ≤ 10⁻⁵ mbar, then admit dry air for the “air” scan | O₂ is a diffusion-limited T₁ quencher, a built-in control |
| Geometry    | excite through the substrate, collect at 45° | minimises wave-guiding/self-absorption artefacts |

# 3 | Reducing the time trace

**IRF de-convolution** – retrieve the true PF decay constant $\tau_{\text{PF}}$ from the first ≈50 ns.

**Choose a cut-off** – set $t_{\text{cut}} = 5 \tau_{\text{PF}}$; at this point <1 % of PF photons remain.

**Integrate intensities**

$$
\Phi_{\text{PF}} = \int_0^{t_{\text{cut}}} I(t)\,dt,\quad
\Phi_{\text{DF}} = \int_{t_{\text{cut}}}^{t_{\text{max}}} I(t)\,dt
$$

**Delayed / prompt ratio**

$$
R_{\text{DP}} = \frac{\Phi_{\text{DF}}}{\Phi_{\text{PF}}}
$$

A full three-level kinetic fit (S₁–T₁–S₀) will yield the individual k’s, but for the ratio the area method is model-agnostic and error-tolerant  
[pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279544/).

# 4 | Triplet-harvesting efficiency

All excitons are created in S₁ under photo-excitation, so

![image](https://github.com/user-attachments/assets/b1be1cd7-48e4-4244-8237-f263aba50f99)


Equation (1) is what you feed into device simulations; it tells you what fraction of all excitons were turned into photons via the T₁ reservoir.

# 5 | Cross-check with steady-state PL

## 5.1 If you do have an integrating sphere

Measure absolute PLQY in vacuum ($\Phi_{\text{vac}}$) and in air ($\Phi_{\text{air}}$). Because O₂ kills DF but not PF  


![image](https://github.com/user-attachments/assets/0c9fe010-9a67-4c74-8b5d-13e0fafe8bc6)



A match (±10 %) between Eq. (1) and the sphere result is a powerful sanity check.

## 5.2 What if you don’t have an integrating sphere?

You can still get a relative $\eta_{\text{TH}}$ that is internally consistent:

- Keep the optical train identical for both “vac” and “air” spectra (same grating, detector gain, slit, spot).
- Correct the spectra for the spectrograph’s wavelength response (use a calibrated quartz-halogen lamp).
- Integrate the emission intensity over the whole band in each atmosphere. Let $S_{\text{vac}}$ and $S_{\text{air}}$ be those integrals.

Compute

![image](https://github.com/user-attachments/assets/f9c49e66-9a0d-4274-880c-18dd2c613774)

Because PF and DF share the same spectrum, the collection geometry factor cancels in Eq. (2). You cannot quote an absolute PLQY without an integrating sphere, but Eq. (2) is still a valid measure of triplet harvesting provided the film is optically thin ($A < 0.2$ at the excitation wavelength) so that self-absorption is negligible.

**Why all the fuss about spheres?**  
Integrating spheres are the only way to capture all photons, including those wave-guided in the substrate – without them PF is routinely over-estimated and DF under-estimated  
[pmc.ncbi.nlm.nih.gov]([https://pubs.acs.org/doi/10.1021/ac2021954)  


If your film is thicker or highly scattering, add a front-face reference standard (e.g. quinine bisulfate film) to correct for re-absorption; the uncertainty will still be larger (~±20 %).

# 6 | Worked example (real numbers)

| quantity (405 nm excitation, 3 wt % emitter in CBP) | value    |
|------------------------------------------------------|----------|
| $\tau_{\text{PF}}$ (de-conv.)                        | 8.6 ns   |
| $\tau_{\text{DF}}$ (single-exp tail)                 | 3.9 µs   |
| $\Phi_{\text{PF}}$ (area)                            | 1.00 (norm.) |
| $\Phi_{\text{DF}}$ (area)                            | 0.72     |
| $\eta_{\text{TH}}$ [Eq. (1)]                         | 0.42     |
| $S_{\text{vac}}$ (front-face)                        | 100 a.u. |
| $S_{\text{air}}$                                     | 58 a.u.  |
| $\eta_{\text{TH}}$ [Eq. (2)]                         | 0.42     |

Agreement within measurement noise confirms that the integration windows, O₂ quench, and detector calibration are all self-consistent.

# 7 | Common pitfalls and how to dodge them

| pitfall | fix |
|---------|-----|
| Laser rep-rate too high → re-excites T₁, inflates PF | set repetition period ≥ 10 $\tau_{\text{DF}}$ |
| Detector stitching artefacts | overlap the 3 time windows on a Coumarin-153 reference |
| Residual O₂ in “vacuum” scan | DF shoulder visible in the “air” trace? → re-pump or purge |
| Optically thick film in front-face mode | use an integrating sphere or dilute the film; otherwise $\eta_{\text{TH}}$ is underestimated |

# 8 | Why this matters for devices

Internal quantum efficiency obeys:

$$
\text{IQE} \leq \eta_{\text{TH}} \times \eta_{\text{out}}.
$$

A film with $\eta_{\text{TH}} = 0.42$ and a realistic out-coupling of 0.3 already allows >12 % EQE—consistent with recent EQE-roll-off analyses that couple $\tau_{\text{DF}}$ and $\tau_{\text{PF}}$ to device performance  
[link.aps.org](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.18.054082).

# 9 | Key take-aways

- **Integrate first, fit later** – the PF/DF area method is robust and model-independent.
- **$\eta_{\text{TH}}$ is the headline figure** – report it alongside lifetimes.
- **O₂ quenching is your free cross-check** – it works even without a sphere.
- **Be honest about geometry** – front-face spectra are fine for ratios, never for absolute $\Phi_{\text{PL}}$.

Master these steps and you can teach a newcomer (or a reviewer!) exactly how much of the triplet reservoir your TADF molecule really harvests—and why they should trust the number.

see details of this for rigorious analysis [TADF](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/25935/Tsuchiya_2021_JPCA_Kineticanalysis_AAM.pdf?sequence=1)

the above can be converted into Master equations below:
# 1 Define the state vector

$$
N(t) =
\begin{bmatrix}
S(t) \\
T(t)
\end{bmatrix},
\quad
S(t) = \text{population of the lowest singlet } (S_1),\quad
T(t) = \text{population of the lowest triplet } (T_1).
$$

The ground-state population is not written explicitly because  
$S_0(t) = 1 - S(t) - T(t)$  
by conservation of excitons.

# 2 Enumerate every elementary rate

| Symbol     | Process                                              | Acts on |
|------------|------------------------------------------------------|---------|
| $k_{rS}$   | radiative decay $S_1 \rightarrow S_0$ (prompt fluorescence) | $S$     |
| $k_{nrS}$  | non-radiative decay $S_1 \rightarrow S_0$            | $S$     |
| $k_{ISC}$  | forward intersystem crossing $S_1 \rightarrow T_1$   | $S \rightarrow T$ |
| $k_{RISC}$ | reverse intersystem crossing $T_1 \rightarrow S_1$   | $T \rightarrow S$ |
| $k_{rT}$   | radiative decay $T_1 \rightarrow S_0$ (phosphorescence) | $T$     |
| $k_{nrT}$  | non-radiative decay $T_1 \rightarrow S_0$            | $T$     |

No rate is set to zero a priori.

# 3 Write the coupled differential equations

![image](https://github.com/user-attachments/assets/3f75d973-9d3a-4224-9267-2b32267f8b61)


Equation (1) is the matrix form of Eqs. 10–12 in the paper;  
$K$ is the full rate operator that governs every possible first-order transition between $S_1$ and $T_1$ and out of the excited manifold.  
No hierarchy of magnitudes ![image](https://github.com/user-attachments/assets/972d0e67-ffd4-460f-b913-564007e247a0) is assumed.
(M) is exact for a three-state TADF scheme; it is identical (line-for-line) to the one Tsuchiya uses to derive his cubic characteristic equation (their eq 24) that eventually feeds Table 2. Nothing is missing unless your sample also exhibits second-order channels (TTA, TPA, TADF-exciplexes, etc.) or higher excited states—those are outside the scope of any first-order three-state mode
# 4 Formal solution

With the initial condition $N(0) = [S_0,\, T_0]^T$,

![image](https://github.com/user-attachments/assets/41d14911-1c63-4e04-9747-135a8296f809)


Diagonalising $K$ (or applying Laplace transforms) yields two eigen-decay constants:

$$
k_{\pm} = \frac{1}{2} \left[ \text{tr}K \pm \sqrt{(\text{tr}K)^2 - 4\det K} \right],
$$

which reduce numerically to the familiar prompt ($\tau_{PF} = 1 / k_+$) and delayed ($\tau_{DF} = 1 / k_-$) lifetimes after extra assumptions are introduced.  
But Eq. (1) itself remains exact.


 # Inclusion of non-linear effect
 “bare” three-state TADF operator so that it also covers second-order triplet–triplet annihilation / fusion (TTA, occasionally written TTS when the energy is recycled into another triplet rather than a singlet),excimer formation and decay, and aggregation-induced quenching (AIQ) of both singlet and triplet excitons.

# 1 State vector and first-order processes (linear part)

We extend the population column vector to hold the extra one-body states

$$
N(t) =
\begin{bmatrix}
S(t) \\
T(t) \\
E(t)
\end{bmatrix},
$$

where

- $S$: lowest singlet exciton (monomer),  
- $T$: lowest triplet exciton (monomer),  
- $E$: bound excimer† (spin-allowed $S_1^*S_1^*$ dimer).

† Triplet excimers do exist, but they almost always convert to monomer triplets on a sub-picosecond time-scale and can be folded into $T$.

All first-order (one-exciton) rate constants appear exactly as in the classic three-state model plus two new rows/columns:

| Symbol           | Process |
|------------------|---------|
| $k_{rS}, k_{nrS}$ | $S \rightarrow S_0$ radiative / non-radiative |
| $k_{ISC}, k_{RISC}$ | $S \leftrightarrow T$ intersystem crossing |
| $k_{rT}, k_{nrT}$ | $T \rightarrow S_0$ phosphorescence / NR loss |
| $k_{fE}$          | formation of an excimer: $S + S_0 \rightarrow E$ (pseudo-first order because $[S_0] \gg [S]$) |
| $k_{dE}$          | dissociation of $E \rightarrow S + S_0$ |
| $k_{rE}, k_{nrE}$ | radiative / NR decay of the excimer |
| $k_{AI,S}, k_{AI,T}, k_{AI,E}$ | aggregation-induced NR quenching of $S, T, E$ |

Collecting those terms gives the linear operator

$$
K_1 =
\begin{bmatrix}
-(k_{rS} + k_{nrS} + k_{ISC} + k_{fE} + k_{AI,S}) & k_{RISC} & k_{dE} \\
k_{ISC} & -(k_{rT} + k_{nrT} + k_{RISC} + k_{AI,T}) & 0 \\
k_{fE} & 0 & -(k_{rE} + k_{nrE} + k_{dE} + k_{AI,E})
\end{bmatrix}.
\tag{1}
$$

If only the terms in (1) were present the system would obey

$$
\frac{d}{dt} N(t) = K_1 N(t),
$$

which is strictly linear and therefore still diagonalised by the usual eigen-value trick.

# 2 Introducing the second-order channels (non-linear part)

## 2.1 Triplet-triplet annihilation / fusion (TTA, TTS)

Two triplet excitons can collide:

$$
T + T \xrightarrow{k_{TTF}} S + S_0 \quad \text{(fusion / delayed singlet)}, \\
T + T \xrightarrow{k_{TTS}} T + S_0 \quad \text{(elastic / scattering)},
$$

with a rate proportional to the square of the triplet density.  
Let $k_{TTF}$ and $k_{TTS}$ be the intrinsic bimolecular rate constants (units cm³ s⁻¹).  
If we work with number densities $S, T, E$ (cm⁻³), the TTA contribution to the population derivatives is

![image](https://github.com/user-attachments/assets/81f08525-41da-4d5a-b78d-3c4514b82614)


(Each annihilation event removes two triplets, hence the factor 2.)

## 2.2 Aggregation-induced singlet–triplet quenching (higher-order)

If excitons hop until they encounter an aggregated “trap” region they are lost non-radiatively.  
The simplest coarse-grained description treats the process as pseudo-first order with  
$k_{AI,S}, k_{AI,T}, k_{AI,E}$ already present in (1).

If your morphology evolves under excitation (e.g. growing aggregates ≈ time-dependent quenchers) you may need to promote AIQ to bimolecular form, but that moves beyond the Tsuchiya framework and usually demands a spatially resolved Monte-Carlo.

# 3 Putting everything together – the master equation

Below is the exact set of coupled population-rate equations once all first-order channels (radiative, non-radiative, ISC, RISC, excimer formation / dissociation, aggregation quenching) and the second-order triplet–triplet interaction channels (TTA/TTS) are included. No terms are dropped, re-labelled, or approximated.

## 3.1 Compact vector form

![image](https://github.com/user-attachments/assets/e2ae5757-57a6-4160-8ad3-0765ff27913b)


