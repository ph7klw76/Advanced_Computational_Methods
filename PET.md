
# Organic photoinduced electron transfer (PET)

Organic photoinduced electron transfer (PET) lies at the heart of both organic photovoltaics (OPVs) and photocatalytic systems, serving as the primary step by which absorbed photons are converted into separated charges or reactive redox intermediates. In OPVs, PET governs the separation of tightly bound excitons into free electrons and holes, while in photocatalysis it drives redox cycles that generate radical species for chemical transformations or fuel production. Achieving record efficiencies in both fields requires a deep, quantitative understanding of the interplay between electronic coupling, molecular reorganization energies, interfacial energetics, and dynamic relaxation processes spanning femtoseconds to microseconds  

## Theoretical Framework: Marcus Theory and Beyond

The canonical framework for non‐adiabatic PET is provided by Marcus theory, which relates the electron‐transfer rate constant $k_{ET}$ to key molecular and environmental parameters:

$$
k_{ET} = \frac{2\pi}{\hbar} \, \lvert H_{DA} \rvert^2 \, \frac{1}{\sqrt{4\pi\lambda k_B T}} \exp \left[ -\frac{(\Delta G_0 + \lambda)^2}{4\lambda k_B T} \right].
$$

Here, $H_{DA}$ is the electronic coupling between donor and acceptor states, $\lambda$ is the total reorganization energy (inner‐ and outer‐sphere), and $\Delta G_0$ is the driving force (Gibbs free energy change) for the transfer. The exponential term captures the thermally activated crossing of parabolic free‐energy surfaces, predicting an “inverted region” of decreasing rate at very large driving forces when $\lvert \Delta G_0 \rvert > \lambda$.

Extensions to the Marcus picture including instantaneous Marcus theory for nonequilibrium solvation and vibronic coupling treatments allow quantitative modeling of ultrafast PET in condensed phases.


# Photo-induced Electron Transfer in Organic Photovoltaics  
**A Mechanistic Deep-Dive for the 20 % Efficiency Era**

Organic photovoltaics (OPVs) have just crossed the psychologically important 20 % certified power-conversion‐efficiency (PCE) threshold thanks to high-entropy donor–acceptor blends and next-generation Y-series non-fullerene acceptors (NFAs).  


At these performance levels every millivolt of energy loss and every femtosecond of interfacial dynamics matter. This article unpacks the physics and chemistry that govern photo-induced electron transfer (PET) in OPVs with a level of quantitative rigor suitable for researchers designing the next wave of >20 % devices.

## 1. From Photon to Charge: The Five-Step Sequence

- **Absorption:** π-conjugated donors (D) or NFAs (A) create tightly bound Frenkel excitons (binding energy $E_b \approx 0.3$–$1$ eV; diffusion length $L_D \approx 5$–$20$ nm).
- **Diffusion:** Excitons random-walk to a D:A interface before their 0.5–1 ns radiative/non-radiative decay clock runs out.
- **Interfacial PET:** On arrival the exciton undergoes non-adiabatic electron (or hole) transfer with time constants as short as 30–100 fs, forming a coulombically bound charge-transfer (CT) state.  

- **CT-state Dissociation:** The electron–hole pair must overcome their mutual Coulomb energy (≈300–400 meV in low-ε organic media) to form free carriers.
- **Collection:** Delocalised carriers percolate through donor/acceptor domains to the electrodes, where they are extracted if they avoid nongeminate recombination.

**PET (steps 3–4)** therefore sets the quantum efficiency ceiling—mastering it is the key to simultaneously maximising short-circuit current $J_{sc}$ and open-circuit voltage $V_{oc}$.

## 2. Quantitative Framework: Marcus Theory in the Solid State

For incoherent, non-adiabatic transfer, the electron-transfer rate is captured by the semiclassical Marcus expression:

$$
k_{ET} = \frac{2\pi}{\hbar} \lvert H_{DA} \rvert^2 \frac{1}{\sqrt{4\pi \lambda k_B T}} \exp \left[ -\frac{(\Delta G_0 + \lambda)^2}{4\lambda k_B T} \right],
$$

where:

- $H_{DA}$ — electronic coupling (1–50 meV typical for molecular crystals)  
- $\lambda = \lambda_{\text{intra}} + \lambda_{\text{outer}}$ — reorganisation energy (0.1–0.3 eV for rigid NFAs)  
- $\Delta G_0$ — free-energy driving force (often equated to the interfacial LUMO/HOMO offset)

Thin-film OPVs operate in the intermediate (near-resonant) Marcus regime:  
$|\Delta G_0| \lesssim \lambda$. Here, $k_{ET}$ is extremely sensitive to  
(i) micro-motion that modulates $H_{DA}$ and  
(ii) entropic contributions that effectively lower $\Delta G^\ddagger$.  
Tuning blends to sit at the peak of the Marcus parabola maximises both PET rate and $V_{oc}$.  

Recent multi-molecule “generalised Marcus” models explicitly include delocalised initial and final states—crucial when crystalline NFAs such as Y6 or BTP-eC9 create band-like acceptor aggregates with coherence lengths of 3–5 molecules. Delocalisation boosts $H_{DA}$ and lowers effective $\lambda$, explaining how modern devices achieve near-unity internal quantum yield with <0.15 eV offsets.  


## 3. Hot, Coherent and Entropy-Assisted Charge Separation

Femtosecond transient absorption and time-resolved soft-X-ray photo-electron spectroscopy show that “hot” CT states—born with excess vibrational or electronic energy—can surmount the Coulomb barrier before cooling. Sub-100 fs interfacial transfer observed at CuPc:C₆₀ and contemporary Y6-based blends leaves insufficient time for full nuclear relaxation, favouring prompt separation.  


Where the driving force is minuscule or even endothermic, configurational entropy provides an additional $\sim k_B T \ln W$ (~25–40 meV) of free energy—enough to tip the balance toward spontaneous charge separation. High-entropy blends that incorporate four or more molecularly similar NFAs deliver both broader density-of-states distributions and record 20 % PCEs.  


## 4. Morphology, Coupling and the “Goldilocks” Interface

- **Domain size:** PET demands a finely intermixed bicontinuous morphology with D:A domains $\lesssim$15 nm (exciton diffusion limit) yet percolating pathways for carriers.
- **Crystallinity:** Crystalline acceptor π-stacks provide delocalised states, enhancing $H_{DA}$; however, excessive purity widens the D:A separation and kills coupling.
- **Interfacial Dipoles:** Local electrostatic fields from polar side-chains or quadrupoles can reduce the effective Coulomb well by 50–100 meV.
- **Entropy-driven miscibility:** Alloys of Y6-type NFAs (e.g., L8-BO, L8-ThCl) fine-tune frontier levels and suppress large crystalline domains, hitting the “Goldilocks” balance between coupling and transport.  


Advanced GIWAXS/NEXAFS tomography confirms that the most successful ≥19 % cells exhibit a hierarchical morphology: ~10 nm mixed phase at the interface nested inside ~50 nm pure D or A fibrils. Such architecture keeps exciton-to-interface distance short while offering low-resistance transport highways.

## 5. Loss Channels and How PET Influences Them

| Loss channel                         | PET parameter that controls it                    | Mitigation strategy                                  |
|-------------------------------------|---------------------------------------------------|------------------------------------------------------|
| Geminate recombination (CT → ground) | Off-rate $k_{GR} \propto \exp[-\beta r]$ (β≈0.5 Å⁻¹) | Increase dielectric constant (additive dipoles), hot CT extraction |
| Non-radiative voltage loss ($\Delta V_{nr}$) | Lower reorganisation energy, vibrational overlap | Rigid fused-ring NFAs, deuteration                   |
| Bimolecular (nongeminate) recombination | Carrier mobility vs trap density                 | Reduce energetic disorder via uniform π–π stacking   |

Because PET sets the initial spatial separation and energy of carriers, mastering it suppresses every downstream loss process.

## 6. State-of-the-Art Efficiencies and Why PET Enabled Them

- **20.0 % certified single-junction (PM6:D18:L8-ThCl multi-alloy):** entropy-stabilised morphology and 0.12 eV offset deliver >95 % charge-generation yield.  


- **20 % high-entropy Joule device:** four-component acceptor alloy increases configurational entropy and reduces $\Delta V_{nr}$ to 0.22 V—among the lowest ever reported.  


Version-66 NREL efficiency tables list >50 entries with ≥19 % PCE, all relying on ultra-fast (<150 fs) PET verified spectroscopically.  

The common denominator is efficient yet low-offset PET that preserves voltage while maintaining near-unity exciton-to-carrier conversion.

## 7. Experimental & Computational Tool-Kit

| Technique                               | Temporal window  | What it reveals                               |
|----------------------------------------|------------------|-----------------------------------------------|
| Femtosecond transient absorption       | 10 fs–10 ns      | Exciton quenching vs CT-state rise            |
| Time-resolved X-ray photo-electron spectroscopy (TR-XPS) | 100 fs–1 ps       | Element-specific charge migration             |
| Time-resolved soft-X-ray absorption    | 10 fs–100 ps     | Site-specific orbital occupancy               |
| Two-dimensional electronic spectroscopy | 10–200 fs        | Vibronic coherence, delocalisation length     |
| Atomistic non-adiabatic molecular dynamics (NAMD) + instant Marcus | fs–ns | Coupled nuclear/electronic trajectories on realistic morphologies |

These complementary probes allow direct extraction of $H_{DA}$, $\lambda$, and CT binding energies, feeding back into materials-by-design loops.

## 8. Design Rules for >20 % OPVs

- Minimise $|\Delta G_0|$—target 0.1–0.15 eV offsets; compensate with high $H_{DA}$ and entropy.
- Rigid, fused-ring NFAs to lower $\lambda$ and suppress vibrational losses.
- Mixed-phase interface thickness ~10 nm for efficient exciton harvesting, nested within crystalline transport domains.
- High-entropy acceptor alloys to flatten energetic landscape and harvest entropy.
- Dipole-rich side-chains or polymeric interlayers to screen Coulomb attraction and accelerate CT dissociation.
- Process-controlled crystallinity (hot-slot-die, sequential deposition)—lock in “Goldilocks” morphology over ≥1 cm² modules.

## 9. Outlook

Ultrafast x-ray free-electron laser facilities and ML-accelerated NAMD promise atomistic‐to-module-scale understanding of PET within this decade. Coupled with scalable printing, OPVs may soon challenge perovskites in the 25 % efficiency class—provided we continue to squeeze every last femtosecond and millivolt out of the photo-induced electron-transfer step.


# Photo-induced Electron Transfer (PET) in Photocatalytic Systems  
**— A Rigorous Kinetic & Theoretical Framework**

Organic and inorganic photocatalysts—whether molecular ruthenium dyes, organic radicals, plasmonic nanoparticles, metal oxides, or quantum dots—convert photons into redox equivalents through a cascade of photo-induced electron-transfer (PET) events. At the device or reactor scale the macroscopic rate follows directly from a network of microscopic PET steps whose time constants range from tens of femtoseconds to seconds.

Below you will find:

- a universal state-diagram for both homogeneous photoredox and heterogeneous semiconductor photocatalysts,  
- the matrix rate equations that govern their temporal evolution,  
- instructions for building and solving the model in practice, and  
- guidance on how to parameterise every rate constant from experiment or first-principles theory (Marcus/Marcus–Levich–Jortner, Redfield, NAMD, etc.).

## 1. Elementary States & Transitions

### 1.1 Homogeneous photoredox catalyst

```
PC      --hν, Φabs-->   PC*   --k_r-->   PC  (radiative/non-rad.)  
                            |            ↑  
                            | k_q[S1]    | k_red[S2]
                            v            |  
                           PC•+  <--reg--┘  
```

- PC = ground-state photocatalyst  
- PC* = excited state  
- PC•⁺ = oxidised catalyst after oxidative quenching  
- S₁/S₂ = substrates  
- Φ_abs = quantum efficiency of excitation

### 1.2 Heterogeneous semiconductor (e.g. TiO₂, g-C₃N₄, QDs)

```
VB(h⁺)   CB(e⁻)
  ^        |
  | k_rel  | k_trap  
bulk recomb.  |  
  |        v  
surface-trapped e⁻/h⁺  --k_surf-->  adsorbed reactants → products
```

- VB = valence band  
- CB = conduction band  
- k_trap = carrier trapping  
- k_surf = electron transfer to a surface intermediate

Spectroscopy confirms sub-ps carrier trapping and ns–µs recombination times for most oxides  


## 2. Writing the Master Equation

Collect the populations into a state vector:

$$
n(t) = (n_{PC}, n_{PC^*}, n_{PC^{\bullet +}}, n_{S1}, n_{S2}, \dots)^T
$$

The continuous-time Markov (master) equation is:

![image](https://github.com/user-attachments/assets/22e7c198-2aef-42b5-9dbb-501a8a6f6636)


where **K** is the rate matrix and **G(t)** is an inhomogeneous generation term that injects excited states at the photon absorption rate Φ_abs I(t). For the 3-level photoredox core:

![image](https://github.com/user-attachments/assets/5178b7ea-ce30-42c1-b8eb-e4cbe8a20e98)


All column sums are zero, ensuring probability (or mass) conservation  

An analogous but larger **K** is written for semiconductor systems with CB, VB, trap and surface states; inter-band recombination enters through off-diagonal elements $k_{rec}$.

## 3. Connecting K to Molecular/Materials Parameters

### 3.1 Photo-induced ET rate constants

For weak electronic coupling, the semi-classical Marcus expression applies:

![image](https://github.com/user-attachments/assets/ea04aa54-357e-4120-8ed8-7f91c9dd8037)


where:  
- $H_{DA}$ = electronic coupling  
- $\lambda$ = total reorganisation energy  
- $\Delta G_0$ = driving force  

Recent work shows that both symmetric and asymmetric Marcus variants reproduce photoredox ET barriers within ~0.05 eV for >100 catalyst–substrate pairs  

### 3.2 Surface or trap-mediated ET

At semiconductor interfaces the same expression holds if the driving force is referenced to band edges and $\lambda$ includes solvent polarisation. DFT+PCM or GW-BSE methods yield the parameters. Carrier trapping and detrapping can be treated as ET between extended (band) and localised (trap) states with their own $H_{DA}, \lambda$.

## 4. Solving and Using the Model

### 4.1 Analytical steady-state

At steady state ($\dot{n} = 0$), Eq. (1) gives:

$$
n_{ss} = -K^{-1} G
$$

provided **K** is non-singular. In homogeneous photoredox, the steady-state approximation reproduces the Stern–Volmer and quantum-yield expressions  

### 4.2 Time-dependent spectroscopy

For transient-absorption or time-resolved photoluminescence experiments, integrate Eq. (1):

![image](https://github.com/user-attachments/assets/f3f13d8d-9b73-4bea-b63c-c24cde92ce0a)


Convert populations to observables through a measurement matrix **M** (Beer–Lambert or emissive coefficients). Singular-value decomposition of the resulting data matrix directly yields the eigenvalues of **K**, i.e., the kinetic time constants.

### 4.3 Numerical example (semiconductor)

For a TiO₂ nanoparticle with states  
$\{ e_{CB}, h_{VB}, e_{trap}, h_{trap}, O_2^{\cdot-}, \cdot OH \}$, one might write:

```
K =
[ 
  -k_{e→t} - k_{rec}     k_{e→t}        0       0       0       0
  0                    -k_{h→t} - k_{rec} k_{h→t}    0       0       0
  k_{t→e}              0            -k_{t→e} - k_{et surf}   0       k_{et surf}  0
  0                    k_{t→h}        0     -k_{t→h} - k_{ht surf}  0    k_{ht surf}
  0                    0            0       0      -k_{O2}      0
  0                    0            0       0       0     -k_{\cdot OH}
]
```

Photon absorption appears in **G** as  
$(G_e, G_h, 0, \dots)^T$ where $G_{e,h} = \Phi_{abs} I_0(t)$

The matrix is readily coded in Python/Julia/MATLAB and fitted to TAS kinetics, giving recombination and surface-ET rates consistent with femtosecond diffuse-reflectance data  

## 5. Parameterisation Workflow

| Step     | Homogeneous system                 | Heterogeneous system                   | Main tools                        |
|----------|------------------------------------|----------------------------------------|-----------------------------------|
| $H_{DA}$ | TD-DFT or CASSCF on PC/S complex   | DFT on slab + non-orthogonal NEGF      | Quantum-chem codes                |
| $\lambda$| Normal-mode analysis in solvent    | Implicit/explicit solvent + surface relax | PCM, AIMD                      |
| $\Delta G_0$ | Redox potentials (cyclic voltammetry) | Band-edge alignment (UPS/XPS)       | Electrochem., UPS                 |
| $k_r$, $\tau$ | Time-resolved PL / TCSPC       | TR-PL of band-edge states              | Ultrafast optics                  |
| $k_q$    | Stern-Volmer slope                 | Surface‐adsorbate photovoltage         | TAS, TRMC                         |
| $\Phi_{abs}$ | Actinometry                    | Scattering-corrected radiometry        | Integrating sphere                |

Combining these inputs yields a fully populated **K**, closing the loop between ab initio chemistry and reactor-scale simulation.

## 6. Why the Matrix View Matters

- **Coupled photophysics & catalysis** – Off-diagonal elements capture competition between productive ET and recombination, rationalising why interfacial charge transfer dominates photoredox yields  

- **Design sensitivity** – Eigen-analysis pinpoints which rate constants bottleneck turnover; in TiO₂, improving $k_{et surf}$ by an order of magnitude has the same impact as doubling light absorption.

- **Scalability** – The same formalism extends from single-electron events to multi-electron, proton-coupled or cascade ET by simply enlarging **K**.

## 7. Outlook

Future PET modelling will couple matrix rate equations with machine-learned Marcus parameters and reactor-scale CFD, enabling closed-loop optimisation of large-area photoreactors under realistic solar spectra  

Ultrafast X-ray and electron diffraction will supply real-time **K** matrices for next-gen catalysts, while asymmetric Marcus frameworks continue to improve the predictive accuracy for both ET and energy-transfer photocatalysis  

The rigorous yet flexible approach presented here positions researchers to push quantum efficiencies toward their thermodynamic limits.


# Seeing Electrons Move: Experimental & Computational Probes of Photo-induced Electron-Transfer (PET) Dynamics

Photo-induced electron transfer underlies the performance of organic photovoltaics, photoredox catalysis, photosynthesis and emerging optoelectronics. The fundamental act—promotion of an electron (or hole) across a donor/acceptor gap—unfolds on the attosecond–microsecond window and over Å-to-µm length scales, demanding a toolbox that spans 18 orders of magnitude in time and six in space. This article surveys, with rigorous detail, the state-of-the-art experimental observables and the computational formalisms that together let us watch, time-stamp and ultimately engineer PET.

## 1 Temporal Landscape

Excited-state wave-packets acquire phase on attosecond (<1 fs) scales; electronic coherence is typically lost within 10–50 fs, while nuclear reorganisation and solvent equilibration proceed over 100 fs–10 ps. Charge separation or recombination may complete within the first picosecond (OPVs), extend into the ns–µs range (trap-limited photocatalysts) or persist to ms (redox-mediated catalysis). Techniques and models must therefore be matched to the process of interest; no single probe suffices.

| Window | Main observables | Representative probes |
|--------|------------------|------------------------|
| 0.1–10 fs | Coherent e-migration, core-level shifts | Attosecond XUV pump/IR probe TA, streaked photoelectron spectroscopy |
| 10–500 fs | CT-state birth, polarisable solvent response | fs-TA, 2DES, FSRS, TR-XPS/XAS |
| 0.5–100 ps | Vibronic cooling, CT dissociation, hot-carrier quenching | TR-PL, TR-THz, TRMC, UED |
| 100 ps–µs | Trapping, recombination, catalytic turnover | Microwave/THz conductivity, transient EPR, spectroelectrochemistry |

## 2 Ultrafast Optical Probes

**Femtosecond / attosecond transient absorption (TA)**  
Broadband TA delivers ΔA(λ,t) maps with <10 fs resolution; attosecond XUV pump–IR probe variants now record charge motion in donor–spacer–acceptor cations in <10 fs, directly revealing coherent electron–nuclear coupling  

**Two-dimensional electronic spectroscopy (2DES)**  
Phase-locked triplets of fs pulses encode frequency–frequency maps that isolate cross-peaks from CT-state formation and vibronic coherences. Recent 2DES measurements resolved 30 fs charge separation in PM6/Y6 solar cells and have been simulated from first principles with real-time TD-DFT  

**Femtosecond stimulated Raman spectroscopy (FSRS)**  
FSRS tags structural fingerprints of transient species with ~100 fs gate. Mode-specific shifts quantify inner-sphere reorganisation energy and coupling between high-frequency vibrations and the ET coordinate  

**Time-resolved photoluminescence & TCSPC**  
Sub-ns emission decays complement TA by reporting bright-state depopulation; global fits with target kinetic schemes yield radiative, non-radiative and PET rate constants  

**Carrier-mobility probes (TR-THz, TRMC)**  
Photo-conductivity transients measure free-carrier density × mobility without electrical contacts. TRMC studies show charge-yield–vs-driving-force trends that mirror Marcus theory in both sensitised films and bulk heterojunctions  

## 3 X-ray & Electron Probes

**Time-resolved X-ray photoelectron (TR-XPS) and absorption (TR-XAS)**  
Core-hole clock methodologies clock electron escape from chromophore to semiconductor in <100 fs by observing ultrafast Auger decay  

**XFEL-based RIXS & diffraction**  
Femtosecond pulses freeze charge density and lattice motion simultaneously  

**Ultrafast electron diffraction / microscopy (UED/UEM)**  
Å-resolution movies capture the structural leg of PET, e.g. π-stack flattening that increases electronic coupling; combined with TA one links ΔA(t) to specific molecular distortions

## 4 Spin & Electrochemical Techniques

Transient EPR isolates radical pair spin states, confirming spin-selective CT routes; spectroelectrochemistry assigns TA features by in-situ redox titration. Ultrafast scanning electrochemical microscopy adds spatial mapping of surface ET heterogeneity (30 nm).

## 5 Extracting Rates: Global Kinetics & Matrix Formalism

Population vector n(t) evolves via

$$
\\frac{d\\mathbf{n}}{dt} = K\\mathbf{n}(t) + G(t),
$$

where $K$ contains all photo-physical and ET rate constants (diagonal: outflow; off-diagonal: inflow) and $G$ is the impulsive pump source. Singular-value decomposition of TA cubes provides the experimental eigenvalues of $K$, while least-squares fitting (or Bayesian evidence optimisation) delivers element-resolved $k_{ij}$. State-of-the-art software (PyASAP, Glotaran) supports constrained fitting with Kramers–Kronig consistency.

## 6 First-Principles & Data-Driven Simulations

**Electronic-structure baseline**  
TD-DFT, GW-BSE and multi-reference wave-function methods yield excitation energies and transition densities; constrained DFT or ΔSCF constructs diabatic donor–acceptor states and electronic couplings.

**Non-adiabatic molecular dynamics (NAMD)**  
Surface-hopping, Ehrenfest and ab-initio multiple spawning propagate coupled electron–nuclear wave-packets; recent extensions treat periodic solids and capture band-to-defect ET in quantum dots with fs accuracy  

**Quantum master equations**  
Redfield, modified-Redfield, Förster and Hierarchical Equations of Motion (HEOM) propagate reduced density matrices

$$
\\dot{\\rho} = -\\frac{i}{\\hbar}[H, \\rho] - R\\rho,
$$

linking bath spectral density to decoherence and hopping rates; 2024 perturbative/HEOM cross-benchmarks quantify the domains of validity  


**Machine-learning acceleration**  
Graph-neural networks trained on >10⁵ molecular geometries predict reorganisation energies within 20 meV and supply $\\lambda$ for thousands of chromophores per hour; transfer-learning slices the chemical space of π-systems and perovskites  


**Quantum simulation**  
Probe-qubit protocols implemented on NISQ hardware already mock up 2DES response functions for three-level PET models, offering Heisenberg-scaling frequency resolution  


## 7 Bridging Experiment & Theory

Attosecond TA on donor-spacer–acceptor triads captured <10 fs charge motion that matches many-body quantum-chemistry predictions once vibronic coupling is included  
Hard-X-ray core-hole clock times of 25 fs at oxide interfaces align with NAMD-derived hopping integrals and solvent reorganisation energies  
In bulk heterojunctions, TRMC carrier yields versus driving force mirror Marcus-Hush rates computed from TD-DFT-derived $H_{DA}$ and ML-predicted $\\lambda$  


## 8 Outlook

The confluence of attosecond light sources, exascale NAMD, and self-learning kinetic models promises “on-the-fly” extraction of $K$ matrices during an experiment, with feedback to pulse shaping or catalyst synthesis in real time. Open-data standards for TA/2DES cubes and for diabatic Hamiltonians will enable community-wide benchmarking and, ultimately, rational design of donor–acceptor assemblies whose PET quantum yields approach unity. Mastering the experimental and computational probes surveyed here is thus not merely academic—it is the route to the next leap in solar energy conversion, chemical synthesis and quantum information science.
