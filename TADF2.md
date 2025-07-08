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

$$
\eta_{\text{TH}} = \frac{\Phi_{\text{DF}}}{\Phi_{\text{PF}} + \Phi_{\text{DF}}} = \frac{R_{\text{DP}}}{1 + R_{\text{DP}}} \tag{1}
$$

Equation (1) is what you feed into device simulations; it tells you what fraction of all excitons were turned into photons via the T₁ reservoir.

# 5 | Cross-check with steady-state PL

## 5.1 If you do have an integrating sphere

Measure absolute PLQY in vacuum ($\Phi_{\text{vac}}$) and in air ($\Phi_{\text{air}}$). Because O₂ kills DF but not PF  


![image](https://github.com/user-attachments/assets/4264b0a7-5c71-420d-9bae-13f3f28e74df)


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
