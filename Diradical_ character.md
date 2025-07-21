# Diradical character as a materials design knob

Open‑shell singlet π‑systems have emerged in NIR OLEDs, photon up‑conversion (singlet fission), organic qubits and spin caloritronics. Their unifying descriptor is the singlet diradical character $y_0$ – a dimensionless number between 0 (closed shell) and 1 (pure diradical) that can be extracted from either theory or experiment.


Controlling $y_0$ tunes not only redox and magnetic properties but also the radiative/non‑radiative balance that dominates deep‑red to telecom‑band photophysics.

---

## 2· The exact two‑orbital Hamiltonian

<img width="588" height="390" alt="image" src="https://github.com/user-attachments/assets/d068d5fa-843b-4a1e-8e02-2b07b179f4ec" />


Label two orthogonal 1s‑like orbitals $\phi_A$ and $\phi_B$. In an orthonormal basis the valence singlet space of H₂ (and, by diabatic projection, any Kekulé‑type diradicaloid) is spanned by three configuration‑state functions (CSFs):

$$
\Phi_{\text{CS}} = \frac{1}{\sqrt{2}} \left| \phi_A \bar{\phi}_B + \phi_B \bar{\phi}_A \right\rangle \quad \text{(covalent, “H–H”)}
$$

$$
\Phi_{\text{CT}} = \frac{1}{\sqrt{2}} \left| \phi_A \bar{\phi}_A + \phi_B \bar{\phi}_B \right\rangle \quad \text{(ionic, “–H H⁺”)}
$$

$$
\Phi_{\text{OS}} = \frac{1}{\sqrt{2}} \left| \phi_A \bar{\phi}_B - \phi_B \bar{\phi}_A \right\rangle \quad \text{(open‑shell, “H•H”)}
$$

Because the CSFs are orthonormal, any singlet ground state is a normalised linear combination

$$
\Psi_{\text{GS}}(\lambda) = \sqrt{1 - \lambda^2} \, \Phi_{\text{CS}} + \left( \sqrt{1 - \lambda^2} - \lambda \right) \Phi_{\text{CT}} + \lambda \, \Phi_{\text{OS}} \quad (0 \leq \lambda \leq \tfrac{1}{\sqrt{2}})
$$

The single real parameter $\lambda$ encapsulates the interplay of the one‑electron gap and the electron‑correlation matrix elements; it will become the tuning knob for $y_0$.

---

## 3· Diradical index $y_0$: definition and connection to observables

<img width="633" height="445" alt="image" src="https://github.com/user-attachments/assets/6b6fbefc-db76-4a92-b563-212933d1f17e" />


The singlet diradical character is defined as twice the weight of the open‑shell CSF:

$$
y_0 = 2\lambda^2, \quad 0 \leq y_0 \leq 1 ................(2)
$$

$y_0 = 0$ ($\lambda = 0$) ⟹ $\Psi_{\text{GS}} = \Phi_{\text{CS}} + \Phi_{\text{CT}}$  
(pure closed‑shell + ionic mixture, no open‑shell contribution).

$y_0 = 1$ ($\lambda = \tfrac{1}{\sqrt{2}}$) ⟹ $\Psi_{\text{GS}} = \tfrac{1}{\sqrt{2}}(\Phi_{\text{CS}} + \Phi_{\text{OS}})$  
– ionic amplitude vanishes, leaving a 50 % covalent / 50 % open‑shell superposition. This corrects the common misconception that the $y_0 = 1$ limit is “purely ionic”.

Because broken‑symmetry DFT, CASSCF or RAS‑CI yield the occupations $n_1$ (HONO) and $n_2$ (LUNO):

$$
y_0 = n_2 = 2 - n_1 ................(3)
$$

making $y_0$ a routine diagnostic in computational screening. Experimentally it can be inferred from vibronic envelopes, ESR‑silent/para‑magnetic thermodynamics or temperature‑dependent magnetic susceptibility.

---

## 4· Weight analysis & the true limits

Inserting eq (2) into eq (1) and collecting squared moduli yields:

$$
w_{\text{OS}} = \frac{y_0}{2}, \quad w_{\text{CS}} = 1 - \frac{y_0}{2}, \quad w_{\text{CT}} = \left(1 - \frac{y_0}{2} - \frac{y_0}{2}\right)^2 ................(4)
$$

| $y_0$ | $w_{\text{OS}}$ | $w_{\text{CT}}$ | qualitative picture |
|------:|----------------:|----------------:|----------------------|
| 0     | 0               | 1               | pure ionic VB limit  |
| 0.5   | 0.25            | 0.25            | strong three‑way resonance → bright |
| 1     | 0.5             | 0               | pure diradicaloid (ionic shut off) |

These trends reproduce the coloured dashed curves on the workshop slides and, as we will see, the behaviour of π‑extended molecules.

---

## 5· Optical selection rules in the 2e/2o CI

<img width="601" height="357" alt="image" src="https://github.com/user-attachments/assets/630a7291-e2af-46dd-9e7f-458ecd9ad5e3" />


Let the lowest bright singlet be dominated by the covalent CSF $\Phi_{\text{CS}}$  
(a HOMO→LUMO excitation termed $\Phi_{\text{HL}}$ on the slides).  
Because the electric‑dipole operator $\hat{\mu}$ changes charge parity, the oscillator strength follows:

$$
f \propto \left| \langle \Psi_{\text{GS}} | \hat{\mu} | \Phi_{\text{CS}} \rangle \right|^2 \propto w_{\text{CS}} \cdot w_{\text{CT}} = \left(1 - \frac{y_0}{2}\right) \left(1 - \frac{y_0}{2} - \frac{y_0}{2} \right)^2 ................(5)
$$

Differentiating gives:

$$
\frac{\partial f}{\partial y_0} = 0 \Longrightarrow y_0(\max f) \approx 0.46
$$

the universal “sweet spot” where NIR diradicaloids reach quantum yields rivaling those of closed‑shell chromophores.  
For $y_0 \to 1$ the ionic weight $w_{\text{CT}} \to 0$; $f$ therefore collapses and fluorescence switches off –  
the “dark diradical” regime exploited in singlet‑fission materials.

---

## 6· Case studies – bringing theory to the lab

### 6.1 Difluorenothiophene diradicaloids

<img width="759" height="542" alt="image" src="https://github.com/user-attachments/assets/71228e58-e934-419d-89f8-97e769bee5f7" />


| compound         | donor group | $y_0$ | 0–0 energy / nm | photoluminescence trend |
|------------------|-------------|-------|------------------|--------------------------|
| DFTh–PhOMe       | –OCH₃       | 0.51  | 910              | moderate                 |
| DFTh–PhNPh₂      | –NHPh₂      | 0.56  | 960              | strong                   |
| DFTh–PhNMe₂      | –NMe₂       | 0.57  | 1000+            | strongest                |

Progressively stronger p‑donors raise the HOMO, shrinking $\Delta E_{\text{HL}}$, thereby increasing $\lambda$ and $y_0$.  
All three reside close to the optimum in eq (5), hence their large oscillator strengths despite sub‑eV optical gaps.

### 6.2 Quinoidal QDT $n$ oligomers

| $n$ (QDTₙ) | skeleton     | $y_0$ | behaviour                                     |
|-----------:|--------------|-------|-----------------------------------------------|
| 1 (QDT₁)   | one CPDT     | 0.00  | visible emission, normal Kasha                |
| 2 (QDT₂)   | dimer        | 0.54  | anomalous NIR luminescence 1050–1180 nm       |
| 3 (QDT₃)   | trimer       | ≈ 1   | no emission – ionic CSF extinct, $f \to 0$    |

Systematically lengthening the quinoidal bridge cools the HOMO‑LUMO gap and walks $y_0$ across the entire 0–1 span in textbook accord with eqs (4)–(5).  
Transient‑absorption spectra for QDT₂ reveal sub‑ps equilibration between $S_1$ and a vibronically hot triplet manifold, corroborating the high diradical character.

---

## 7· Guidelines for molecular architects

<img width="628" height="459" alt="image" src="https://github.com/user-attachments/assets/4e3f2453-aba6-47e0-b9ef-4f89c4129d1a" />


| Target property                        | Desired $y_0$ window | Synthetic levers                                                              |
|----------------------------------------|-----------------------|--------------------------------------------------------------------------------|
| Bright NIR/OLED                        | 0.4 – 0.6             | moderate donors/acceptors, modest quinoidal extension, fused heteroacenes     |
| Singlet fission / spintronics         | ≥ 0.8                 | strong quinoid enforcement, multi‑CPDT scaffolds, push‑pull extremes           |
| Redox‑switchable diradical ↔ tetraradical | tune 0 → 1 electrochemically | redox active bridges; avoid bulky sterics to maintain planarity         |

Because $y_0$ is accessible from natural‑orbital occupations (eq 3) at the semi‑empirical level, thousands of candidates can be triaged before synthesis, drastically compressing the design loop.


useful papers
[https://pubs.acs.org/doi/10.1021/acs.jpclett.2c01325]

[https://pubs.acs.org/doi/abs/10.1021/jz100155s]



# 1 · Why radicaloids break Kasha’s rule

Kasha’s rule says that radiation comes from the lowest excited state of a given multiplicity.  
Diradicaloids challenge this because:

- their HOMO/LUMO gap is small → near‑degenerate CSFs,  
- they contain a latent ionic (charge‑transfer) CSF which mixes strongly with covalent ones,  
- vibronic or spin‑orbit couplings connecting $S_n \leftrightarrow S_1$ (or $D_n \leftrightarrow D_1$) can be symmetry‑blocked or energetically suppressed.

The working variables are (i) orbital topology that fixes the electronic dipole, (ii) the singlet (or doublet) diradical index $y_0$, and (iii) the rate ratio $k_r / k_{IC}$.  
Quantitative control of those three dials is now enabling telecom‑band OLEDs, singlet‑fission photovoltaics and spin‑photon interfaces.

---

# 2 · Exact two‑electron/two‑orbital model

Choose an orthonormal pair $g, u$ spanning the active space. The complete singlet CSF basis is:

| label     | occupation | physical picture           |
|-----------|------------|----------------------------|
| $\Phi_C$  | $g^2$      | closed‑shell covalent      |
| $\Phi_O$  | $g^1 u^1$  | (spin‑coupled) open‑shell diradical |
| $\Phi_D$  | $u^2$      | doubly‑excited / ionic     |

The (normalized) ground state is:

$$
\Psi_{\text{GS}} = a \, \Phi_C + b \, \Phi_O + c \, \Phi_D, \quad a^2 + b^2 + c^2 = 1 \..................................(1)
$$

Natural‑orbital occupations follow from diagonalising the 3×3 CI Hamiltonian built from the one‑electron gap $\Delta$ and the Coulomb (K) and coupling (B) integrals.

---

# 3 · Rigorous definition of the diradical index

Occupation of the LUNO gives:

$$
y_0 = n_u = b^2 + 2c^2, \quad 0 \leq y_0 \leq 1 \ ...................................(2)
$$

$y_0 = 0 \Rightarrow b = c = 0$ (pure closed‑shell + ionic mix).  
$y_0 = 1 \Rightarrow c = 0$, $b^2 = \tfrac{1}{2}$ – ionic weight vanishes, leaving a 50 % covalent / 50 % open‑shell superposition.  
Earlier drafts claiming “pure ionic at $y_0 = 1$” are therefore incorrect.

---

# 4 · Why oscillator strength peaks at $y_0 \approx 0.5$

The lowest bright singlet $S_{\text{bright}}$ is dominated by $\Phi_O$.  
Electric‑dipole operator $\hat{\mu}$ couples ionic ↔ covalent sectors only, so:

$$
f\propto |ac \langle \Phi_C | \hat{\mu} | \Phi_D \rangle|^2 \Rightarrow f(y_0) \propto (1 - y_0) y_0 \ ............................(3)
$$

Equation (3) is maximised at $y_0 = 0.5$.  
Multi‑reference TD‑DFT screening of >70 Kekulé diradicaloids confirms this optimum within ±0.05.  


---

# 5 · Competing non‑radiative channels

For internal conversion between two singlets (or two doublets):

<img width="348" height="47" alt="image" src="https://github.com/user-attachments/assets/3ec671c9-a71c-404d-b67f-b73a752830f7" />...................(4)


The radiation/non‑radiation competition becomes:

$$
\frac{k_r(n)}{k_{IC}(n \rightarrow n - 1)} = \frac{\omega_{n0}^3 |\mu_{n0}|^2}{3 \pi \varepsilon_0 c^3 |V^{\text{vib}}_{n,n-1}|^2 \rho(\Delta E)} \.................(5)
$$

Anti‑Kasha emission therefore needs large $|\mu_{n0}|$ (⇒ $y_0 \approx 0.5$)  
and a mechanism that keeps $|V_{nn-1}|^2 \rho$ small.  
Three scenarios dominate:

| Type | what suppresses $k_{IC}$           | examples here        |
|------|------------------------------------|----------------------|
| I    | energy gap ≈ 0.4–0.6 eV            | azuacenes (4Azu)     |
| II   | vibronic‑symmetry blocking         | DFTh‑diradicaloids   |
| III  | spin‑orbit selection rules         | PT20/PT30 radicals   |

---

# 6 · Topology controls $y_0$ and the blocking mechanism

## 6.1 Azuacenes & Super‑Azulenes (6‑7‑5 vs 6‑5‑7)

Collinear 5|7 dipoles in the 6‑7‑5 “super‑azulene” raise the ground‑state dipole to 2.06 D and open an S₂–S₁ gap of 0.50 eV;  
$V^{\text{vib}}_{21}$ is small because the modes that drive S₂ → S₁ belong to an orthogonal representation.  
Time‑correlated single‑photon counting gives τ(S₂→S₀)=0.36 ns while IC is <1 ps (4Azu) – classic Type‑I anti‑Kasha.  
*ACS Publications*

Extending to 6Azu narrows the S₁–T₁ gap below 0.1 eV and funnels the population into singlet fission (1.5 ps).

## 6.2 Difluorenothiophene diradicaloids

DFTh‑PhOMe → DFTh‑PhNPh₂ → DFTh‑PhNMe₂ increment $y_0 = 0.51 \rightarrow 0.57$.

Rigid C₂ symmetry forces $V^{\text{vib}}_{31} = 0$ for the dominant modes,  
so the S₃ → S₁ pathway is symmetry forbidden; radiation from S₃ at 480–500 nm survives with τ ≈ 150 ps (Type‑II).  
Electrochemical scans show reversible 1e⁻ and 2e⁻ oxidations mapping neatly onto the covalent → diradical → tetraradical ladder.

## 6.3 QDT₂ diradicaloids

The quinoidal bridge reduces $\Delta E_{\text{HL}}$ enough to set $y_0 = 0.54$.  
Emission appears at 1050–1180 nm, with a 790 ps → 6 ns vibrational cascade that matches eq (5)  
when $\rho(\Delta E)$ is modelled by a tri‑exponential phonon DOS.

## 6.4 Doublet radicals PT20 / PT30

Replacing one electron creates a SOMO; the D₂–D₁ gap (0.6 eV) + localized spin on the N‑bridge yields Type‑III anti‑Kasha.  
Nanosecond D₂ fluorescence at 850–950 nm is followed by weak D₁ emission (>1 µs)  
only when heavy‑atom solvents restore SOC.

---

# 7 · Putting it all together

| system       | μ / D | $y_0$ | $\Delta E_{n,n−1}$ / eV | blocker      | dominant emission   |
|--------------|-------|-------|--------------------------|--------------|---------------------|
| 4Azu         | 1.72  | 0.32  | 0.50                     | gap (I)      | S₂, 400 nm          |
| 6Azu         | 1.93  | 0.46  | 0.48                     | SEF          | triplet pair        |
| DFTh‑NMe₂    | 1.20  | 0.57  | 0.45                     | symmetry (II)| S₃, 490 nm          |
| QDT₂         | 0.90  | 0.54  | 0.10 (S₁–T₁)             | —            | S₁, 1080 nm         |
| PT30         | 0.88  | —     | 0.60                     | SOC (III)    | D₂, 880 nm          |



---

# 8 · Design rules (quantitatively benchmarked)

- Target $y_0 = 0.45 − 0.60$ for bright, red‑shifted OLED emitters.
- Exploit Type‑I blocking by stacking local dipoles (super‑azulene, zwitterionic donors).
- Invoke Type‑II with rigid C₂ symmetry or out‑of‑plane bridges that forbid key promoting modes.
- Push $y_0 \rightarrow 1$ and quinoid enforcement when non‑radiative channels (singlet fission, spin filters) are desired.
- For doublet photonics, localise the SOMO and keep $D_2 - D_1 > 0.5$ eV; add controlled heavy‑atom SOC if dual‑band emission is required.

[https://onlinelibrary.wiley.com/doi/10.1002/anie.202413988]

[https://pubs.acs.org/doi/full/10.1021/jacs.3c07625]

[https://pubs.acs.org/doi/10.1021/jacs.4c11186]

# ORCA Protocol for Extracting Diradical Descriptors

Below is a step‑by‑step, ORCA‑specific protocol for extracting the diradical descriptors:

$$
y_0 = n_{\text{LUNO}} \quad \text{and} \quad \lambda = \sqrt{\frac{y_0}{2}}
$$

where $n_{\text{LUNO}}$ is the natural‑orbital occupation of the lowest unoccupied natural orbital (LUNO).  
The same workflow works for any π‑diradicaloid — azuacene, difluorenothiophene, QDT, etc. — provided you adopt a (2e, 2o) active space  
or an unrestricted Kohn–Sham determinant that faithfully localises the two frontier electrons.

---

## 1 · Geometry

If you do not have an X‑ray structure, perform a relaxed optimisation at a semi‑local or hybrid DFT level first;  
the choice will not bias the subsequent occupation analysis so long as the correct spin‑state is used.

```text
! Opt B3LYP D3BJ def2-SV(P) TightSCF TightOpt
* xyz 0 1
  … Cartesian coordinates …
*
```
## 2 · Choose the level of electron correlation

| Strategy                        | When to use              | One‑command keyword                                                  |
|--------------------------------|---------------------------|----------------------------------------------------------------------|
| CASSCF(2,2)                    | < 50 atoms, high accuracy | `! CASSCF(2,2) def2-TZVP TightSCF NoFrozenCore`                      |
| DLPNO‑NEVPT2//CASSCF(2,2)      | benchmarking              | add `! NEVPT2 TightPNO` in a single step                             |
| Broken‑symmetry DFT            | > 100 atoms or screening  | `! UKS ωB97X-D def2-TZVP TightSCF`                                   |

Either path yields natural‑orbital occupations:

- **CASSCF** is formally exact for a (2e, 2o) active space.
- **BS‑DFT** gives a good approximation when the unrestricted solution is genuinely open‑shell.

---

## 3 · Input Templates

### 3.1 CASSCF(2,2) (preferred for molecules < 50 atoms)

Use the same formula as above. A quick sanity check:

- $y_0 \lesssim 0.05$ ⇒ essentially closed‑shell.
- $0.3 \lesssim y_0 \lesssim 0.7$ ⇒ diradicaloid window (strong oscillator strength).
- $y_0 \gtrsim 0.8$ ⇒ near‑pure open‑shell; fluorescence is likely quenched.

If the unrestricted SCF collapses to a symmetric (closed‑shell) solution ($n_u \approx 0$),  
**force broken symmetry** by supplying an ALPHA/BETA charge‑separated guess (`%suspect` block)  
or by **mixing fragment orbitals**.

```text
! CASSCF(2,2) def2-TZVP TightSCF
%casscf
  nel             2        # electrons in the active space
  nact            2        # number of active orbitals
  mult            1        # singlet
  lroot           1        # ground state only
  orbitals        {occupied;  core}   # ORCA guesses, refine if needed
end

%output
  Print[ NaturalOrbitals ] 1
  Print[ Basis ] 0
end

* xyzfile 0 1 final.xyz
```
**Spin multiplicity:**  
When the job finishes, search `*.out` for the **Natural Orbital Occupancies** block.  
Take the highest sub‑unity value $n_{\text{LUNO}}$. Insert into:

$$
y_0 = n_{\text{LUNO}}, \quad \lambda = \sqrt{\frac{y_0}{2}}
$$

### 3.2 Broken‑symmetry DFT (for large systems)

```text
! UKS B3LYP D3 TightSCF
%scf
  UHF         true          # enforce unrestricted
  guess       generate
  maxiter     500
end
%output
  PrintLevel Normal
  Print[ P_NatPop] 1 
  Print[ P_Mulliken ] 1
  Print[ P_OrbPopMO_M ]    1
end
%maxcore 3000
%pal nprocs 32 end
* XYZFILE 0 3 radical.xyz
```

ORCA prints **Natural Orbital Population Analysis** near the end:

- **Singlet diradicaloids**: use a restricted (RKS) guess;  
  if convergence flips to RHF‑like (closed shell), switch to broken‑symmetry in § 3.

- **Neutral radicals / doublets**: multiplicity = 2, unrestricted from the outset.

Save the optimised geometr



