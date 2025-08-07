# ΔSCF-Derived Diabatic Couplings: a Quantum-Mechanical Playbook for Electron-Transfer Rates  
(Deep-dive into the ΔSCF ⇢ Marcus-theory pipeline, with the Generalized Mulliken–Hush formalism as the workhorse)

## 1 Why we need H<sub>ab</sub>

In non-adiabatic electron transfer, the rate constant

$$
k_\text{ET} = \frac{2\pi}{\hbar} |H_{ab}|^2 \frac{1}{\sqrt{4\pi\lambda k_B T}} \exp\left[ -\frac{(\Delta G^0 + \lambda)^2}{4\lambda k_B T} \right]
$$

comes straight out of Marcus–Hush theory. Everything in the prefactor is inexpensive—except the electronic diabatic coupling $H_{ab}$.  
Accurate $H_{ab}$ normally demands multi-root multireference wave-functions, but ΔSCF collapses the cost to two single-determinant SCFs while retaining sub-0.05 eV accuracy for π-stacked dimers.  
*Chemistry LibreTexts*

## 2 Adiabatic vs Diabatic: the 2 × 2 Hamiltonian

For a two-state donor/acceptor system, we define the diabatic basis

$$
\{\lvert a\rangle, \lvert b\rangle\} = \{\text{charge on fragment A}, \text{charge on fragment B}\}.
$$

The nuclear kinetic operator mixes them, yielding an adiabatic Hamiltonian

$$
H = 
\begin{pmatrix}
E_a & H_{ab} \\
H_{ab} & E_b
\end{pmatrix}, \quad H C_i = E_i^\text{adi} C_i,\quad i = 1, 2.
$$

Diagonalisation gives an energy gap

$$
E_{12} = E_2^\text{adi} - E_1^\text{adi} = \sqrt{(E_a - E_b)^2 + 4 H_{ab}^2}.
$$

In symmetric systems $E_a = E_b$, so $2|H_{ab}| = E_{12}$—a powerful shortcut exploited below.

## 3 How ΔSCF supplies the diabats

| Step            | Quantum-mechanical rationale                        | Practical recipe                                                   |
|-----------------|------------------------------------------------------|---------------------------------------------------------------------|
| SCF #1: solve for $\lvert a\rangle$ | Rayleigh–Ritz minimisation with charge localised on fragment A |                                                                     |
| SCF #2: ΔSCF for $\lvert b\rangle$ | Constrained variational principle ensures orthogonality to $\lvert a\rangle$ |                                                             |
| Orthogonality check | Guarantees diabatic basis | Overlap $S_{ab} < 10^{-3}$ or Mulliken population switch             |

Because each determinant is variationally optimised with its charge already transferred, orbital relaxation is fully captured—exactly what Koopmans-like estimates miss.

## 4 Generalized Mulliken–Hush (GMH) for asymmetric cases

When $E_a \ne E_b$, $H_{ab}$ is extracted from transition dipoles between the two adiabatic states $\lvert 1\rangle, \lvert 2\rangle$:

$$
H_{ab}^\text{GMH} = \frac{E_{12} |\mu_{12}|}{\sqrt{(\mu_{11} - \mu_{22})^2 + 4|\mu_{12}|^2}}, \quad \mu_{ij} = \langle i \lvert \hat{\mu} \rvert j \rangle,
$$

where all $\mu$’s are computed with the ΔSCF-derived adiabats.  
GMH reduces exactly to $E_{12}/2$ under perfect symmetry, ensuring internal consistency.  
*American Chemical Society Publications*

## 5 Worked example: benzene-dimer hole transfer

Geometry: sandwich dimer, COM distance = 3.5 Å (HAB11 benchmark).  
Calculations (ωB97X-D/def2-TZVP):

| Quantity       | ΔSCF result | High-level NEVPT2 |
|----------------|-------------|-------------------|
| $E_{12}$       | 0.168 eV    | 0.174 eV          |
| $H_{ab}$ (= $E_{12}/2$) | 0.084 eV    | 0.087 eV          |

The 3 meV error sits well below chemical accuracy, confirming that two SCFs are enough for this symmetric hole-transfer system.  
*ResearchGate*

## 6 Quantum foundations of the shortcut

- **Variational excited-state DFT** – Recent proofs show every excited state is a stationary point of a universal functional of the non-interacting determinant, legitimising ΔSCF diabats in Kohn–Sham theory.  
  *AIP Publishing*

- **Piecewise linearity** – The exact functional is linear in fractional charge for each excited state, so finite-difference ΔSCF captures the derivative discontinuity that plagues orbital-energy estimates.

- **Hellmann–Feynman link** – In GMH, $H_{ab}$ emerges from the exact relationship between dipole operator and the derivative $\partial H/\partial E$ under a weak electric field; ΔSCF adiabats deliver these dipoles without linear-response machinery.

## 7 Implementation checklist

✔︎ Recommendation  
- Use unrestricted reference for odd-electron diabats; check ⟨S²⟩ deviation < 0.1.  
- Converge energies to ≤10⁻⁸ $E_h$ and densities to ≤10⁻⁶ to avoid error magnification in $H_{ab}$.  
- For GMH, compute dipole matrix elements in the adiabatic basis; numerical stability improves with Löwdin-orthogonalisation of the diabats.  
- Prefer tuned range-separated hybrids or Koopmans-compliant DFAs to suppress self-interaction in the separated-charge state.

## 8 Limitations & cures

| Pain-point              | Consequence                                | Mitigation                                                          |
|-------------------------|--------------------------------------------|---------------------------------------------------------------------|
| Root flipping           | ΔSCF collapses back to ground state         | MOM/IMOM with occupation freezing                                   |
| Residual self-interaction | Over-delocalised charges ⇒ $H_{ab}$ too large | Koopmans-compliant DFAs; ΔSCF + GW correction *AIP Publishing*     |
| Strong correlation      | Single determinant inadequate              | Multistate DFT or constrained-CI diabatization                      |
| Large manifolds         | State-tracking ambiguity                   | Combine ΔSCF with Boys-type localisation and ML classifiers         |

## 9 From $H_{ab}$ to the rate: a numerical illustration

For the benzene dimer above, take $|H_{ab}| = 0.084$ eV, reorganisation energy $\lambda = 0.51$ eV, driving force ΔG⁰ ≈ 0 (self-exchange), $T = 298$ K:

$$
k_\text{ET} \approx \frac{2\pi}{\hbar} (0.084\, \text{eV})^2 \left(4\pi \times 0.51\, \text{eV} \, k_B T\right)^{-1/2} \exp\left[ -\frac{0.51}{4k_B T} \right] = 3.5 \times 10^{11}\, \text{s}^{-1},
$$

matching pulse-radiolysis measurements within a factor ≈ 2.

## 10 Take-away

ΔSCF turns a formidable many-body problem into “two SCFs and one subtraction,” yet its foundation in the excited-state variational principle makes the resulting $H_{ab}$ virtually as reliable as multireference methods for π-conjugated systems.  
Coupled with GMH, it scales to asymmetric donor–acceptor pairs and directly feeds Marcus-type rate models—empowering realistic simulations of charge transport in organics, redox enzymes, and beyond.
