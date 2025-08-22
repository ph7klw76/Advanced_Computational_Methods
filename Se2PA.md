# Why Replace Sulfur with Selenium in Two-Photon Photoinitiators? A Quantum-Mechanical Design Guide

## Abstract

Selenium substitution (S → Se) in two-photon absorption (2PA) photoinitiators (PIs) is not a cosmetic swap. It systematically re-weights key terms in the two-photon sum-over-states amplitude, strengthens spin–orbit coupling (SOC) to accelerate intersystem crossing (ISC), and tunes electron-transfer thermodynamics to raise radical yields. This article lays out the governing equations, identifies which matrix elements are perturbed by Se, quantifies effects where possible, and closes with a verification workflow (computational + experimental) .

## 1) Two-Photon Absorption: the levers you can actually pull

For excitation by two identical photons of frequency $\omega$ into a target excited state $\lvert f \rangle$, the 2PA amplitude is (velocity/length-gauge specifics aside) well captured by the SOS expression

$$
M^{(2)}(\omega) \propto \sum_n \frac{\mu_{0n} \cdot \mu_{nf}}{E_{n0} - \hbar \omega - i \Gamma_n},
$$

so the 2PA cross-section 

$$
\delta(2\omega) \propto \lvert M^{(2)} \rvert^2.
$$

Large $\delta$ arises when (i) transition dipoles $\mu_{0n}$ and $\mu_{nf}$ are large, (ii) intermediate states $E_{n0}$ lie near $\hbar \omega$ (near-resonant enhancement), and (iii) the chromophore is highly polarizable (boosts the numerators broadly). These design rules and their reduction to two-/few-state models are standard in the nonlinear optics literature.  


**Implication for PIs.** Under femtosecond NIR irradiation (e.g., 700–1000 nm), the initiation rate roughly follows

$$
R_{init} \propto I^2 \delta(2\omega) \Phi_{rad},
$$

so any substitution that simultaneously increases $\delta$ and the radical quantum yield $\Phi_{rad}$ will multiply performance.  


## 2) What Se changes in the electronic structure (relative to S)

### 2.1 Gap and resonance placement

Replacing S(3p) with Se(4p) increases orbital diffuseness and donor strength, typically raising the HOMO and modestly stabilizing π* levels within a conjugated framework. Net effect: a red-shift of one-photon bands and movement of important intermediate states $E_{n0}$ closer to $\hbar \omega$ for NIR-driven 2PA—i.e., improved denominators in $M^{(2)}$. Direct comparative data for “chalcogenophene” 2PIs show Se analogs exhibit stronger 2PA and superior microfabrication thresholds versus S congeners.  


### 2.2 Polarizability, hyperpolarizability, and the two-state picture

Atomic polarizability increases down Group 16; embedding a more polarizable Se center in a donor–π–acceptor (D–π–A) scaffold enlarges the difference dipole $\Delta \mu$ and the transition dipole $\mu_{01}$. In the widely used two-state approximation,

$$
\delta(2\omega) \sim \frac{\lvert \Delta \mu \, \mu_{01} \rvert^2}{(E_{01} - \hbar \omega)^2 + \Gamma^2},
$$

so Se tends to raise the numerator while shifting $E_{01}$ favorably. Reviews covering structure–property links for $\delta$ and the role of charge-transfer (CT) character substantiate this mechanistic picture.  


### 2.3 Symmetry mixing and selection rules

Strict 1PA/2PA mutual exclusion (centrosymmetric g/u parity) is deliberately violated in practical 2PIs by using push–pull architectures. Heavier chalcogens further mix nearby $n\pi^*$ and $\pi\pi^*$ manifolds, enriching the density of intermediate states that contribute constructively to the SOS sum. The general design logic and its parity implications are treated in standard 2PA reviews.  


## 3) Selenium and spin–orbit coupling: turning excitation into radicals

### 3.1 Why SOC matters for PIs

Efficient PIs must reach reactive states (often $T_1$ or charge-separated triplets) that fragment or engage in H-abstraction/electron transfer. ISC rates from singlets to triplets follow Fermi’s golden rule,

$$
k_{ISC} \propto \lvert \langle \psi_T \lvert \hat{H}_{SO} \rvert \psi_S \rangle \rvert^2 \rho,
$$

with $\hat{H}_{SO}$ scaling strongly with atomic number $Z$. In organic photophysics, the heavy-atom effect is commonly approximated as 

$$
SOC \propto Z^4.
$$

Thus, moving from S (Z = 16) to Se (Z = 34) raises SOC matrix elements by 

$$
\approx \left(\frac{34}{16}\right)^4 \approx 20,
$$

boosting ISC and triplet yields,exactly what a 2PI wants.  


### 3.2 State-mixing rules (El-Sayed)

ISC is especially fast when the orbital character changes between the initial singlet and accepting triplet (e.g., pi-pi^* $  $\rightarrow n\pi^*$ or vice-versa). This is codified in the El-Sayed rules (IUPAC). Se substitution tends to compress the $n\pi^*–\pi\pi^*$ gap and increase orbital mixing, which, together with higher SOC, can substantially increase $k_{ISC}$.  


### 3.3 Practical evidence in 2PI families

Selenium-containing two-photon initiators (e.g., selenopheno-based D–π–A systems) exhibit superior reactivity and lower dose thresholds in acrylate resins versus widely used benchmarks (Irgacure 369, BAPO; and sensitizer ITX), linking Se to both enhanced 2PA and downstream photochemistry.  
(https://pmc.ncbi.nlm.nih.gov/articles/PMC9009090/)

## 4) Radical generation channels and thermodynamics (Type-I / Type-II / PET)

### 4.1 PET driving force (Rehm–Weller)

When radical generation proceeds via photoinduced electron transfer (PI + co-initiator, typically an amine), the driving force

$$
\Delta G^\circ_{ET} \approx E_{ox}(D) - E_{red}(A) - E_{00} - \Delta E_C
$$

becomes more negative if the PI’s oxidation potential drops (or its excited-state energy rises). Se typically lowers ionization energies/stabilizes radical cations relative to S analogs, nudging $\Delta G^\circ_{ET}$ more exergonic and improving initiation kinetics; the formalism and its use in photopolymer systems are well established.  


### 4.2 Competing deactivation and how Se helps

Fast ISC to productive triplets reduces losses through fluorescence/internal conversion, particularly when triplet manifolds connect efficiently to bond scission or H-abstraction coordinates. Heavy-atom enabled SOC acceleration and El-Sayed-favored coupling pathways underpin this improvement in $\Phi_{rad}$.  


## 5) Matching Se-tuned spectra to lasers and resists

NIR femtosecond sources for two-photon lithography (typically Ti:sapphire or Yb-fiber harmonics) operate near 700–1000 nm; Se-induced red-shifts commonly place strong one-photon bands near 350–450 nm, enabling near-resonant 2PA at ~700–900 nm while avoiding excessive linear absorption at the fundamental—ideal for deep-voxel writing.



## 6) Putting it together: why Se outperforms S in 2PIs

**Bigger $\delta$:** Se increases $\mu_{01}$, $\Delta \mu$, and near-resonant contributions,amplifying the SOS numerator and shrinking energy denominators.  


**Higher $\Phi_{rad}$:** SOC↑ (∝ $Z^4$) + El-Sayed-favorable orbital mixing → faster ISC, better access to reactive triplets/CS states.  


**Better PET alignment:** lower IP/more stable PI$^{\cdot+}$ with Se → $\Delta G^\circ_{ET}$ more negative for common amine partners.  


**Laser matching:** Se red-shifts place 1PA bands to 350–450 nm, enabling efficient 2PA at 700–900 nm without linear losses,ideal for 2PL.  

