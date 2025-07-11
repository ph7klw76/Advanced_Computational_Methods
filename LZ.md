# Landau–Zener (LZ) non-adiabatic transition probability

Below is a “from first principles” account of how the Landau–Zener (LZ) non-adiabatic transition probability is derived, how it is mapped onto molecular charge transport, how it interpolates between the classical Marcus picture and the fully adiabatic regime, and what approximations are involved. The treatment follows the original quantum-mechanical derivation by Landau (1932) and Zener (1932), but it is recast in the language used in modern organic-semiconductor transport theory.

## 1 Two–level Hamiltonian along a reaction coordinate

Consider two neighbouring redox sites A and B whose nuclei move collectively along a single reaction coordinate $q$ ($q$ in practice a linear combination of internal vibrations and an intermolecular libration).

The diabatic electronic energies are

$$
E_A(q)=E_{A0}+k\,q,\quad E_B(q)=E_{B0}-k\,q,
$$

so they cross at $q=0$. Coupling $J$ J mixes the electronic states and produces the 2×2 Hamiltonian

$$
H(q)=\begin{pmatrix}
k\,q & J\\
J & -k\,q
\end{pmatrix}.
$$

(1)


Diagonalising (1) gives adiabatic energies

$$
E_\pm(q)=\pm\sqrt{J^2+(k\,q)^2},
$$


which form an avoided crossing with a minimum gap $2J$ at $q=0$.

<img width="1000" height="663" alt="image" src="https://github.com/user-attachments/assets/0920a279-2023-4c25-b09e-b97aa0e6e065" />


## 2 Time-dependent crossing and the original LZ problem

Assume the nuclear subsystem moves at a constant velocity through the crossing:

$$
q(t)=v\,t,
$$

so the time-dependent Schrödinger equation reduces to a linear-sweep problem

<img width="373" height="94" alt="image" src="https://github.com/user-attachments/assets/4d6f6c37-23af-42ea-b399-70ae556ecd3e" />


with diabatic amplitudes $c_A(t),c_B(t)$ cA(t),cB(t). Imposing that the system is purely on state A at $t\to -\infty$ t→−∞, LZ showed that the probability to remain on A (i.e. not make a transition) after the passage is

$$
P_{ad}=\exp\!\Bigl(-\frac{2\pi J^2}{\hbar\,k\,v}\Bigr).
$$


The probability to transfer to B is therefore

$$
P_{LZ}=1-P_{ad}=1-\exp(-2\pi\Gamma),\quad \Gamma\equiv\frac{J^2}{\hbar\,k\,v}.
$$


Interpretation of $Γ$:

**Large $J$ or slow sweep → $Γ\gg1$ Γ≫1 → transition probability $P_{LZ}\to1$ P LZ →1: the crossing is essentially diabatic.

**Small $J$ or fast sweep → $Γ\ll1$ Γ≪1 → $P_{LZ}\simeq2\pi\Gamma\propto J^2$ P LZ ≃2πΓ∝J2: the electron stays adiabatic.

## 3 Connecting the sweep parameters to molecular properties

For a harmonic potential in Marcus theory the driving coordinate obeys

$$
\frac12\,k\,(q\pm q_0)^2,\quad q_0=\frac{2\lambda}{k},
$$


with reorganisation energy $λ$ .

Linearising near $q=0$  gives $k=2\lambda/q_0$ .

The characteristic velocity with which the nuclei traverse the crossing is set by the vibrational frequency $ω_{eff}$ ω eff:

$$
v\approx q_0\,ω_{eff}=\frac{2\lambda}{k}\,ω_{eff}.
$$


Combining, the Landau–Zener adiabaticity parameter becomes

$$
Γ=\frac{J^2}{\hbar\,ω_{eff}\,λ}.
$$


Typical organic semiconductors:

λ = 0.2–0.4 eV

ωeff = 40–150 cm⁻¹ (5–20 meV) → 6–25×10¹² s⁻¹

<img width="600" height="85" alt="image" src="https://github.com/user-attachments/assets/fa2382d9-57aa-465f-a23e-e97f227218b4" />


## 4 Landau–Zener-corrected hopping rate

Multiply the golden-rule rate $k_{NA}\propto J^2$ k NA ∝J2 by the LZ transmission factor $\kappa_{LZ}=P_{LZ}$ κ LZ =P LZ:

$$
k_{hop}=k_{NA}\,[1-\exp(-2\pi\,Γ)].
$$

k hop =k NA [1−exp(−2πΓ)] .(7)

<img width="945" height="167" alt="image" src="https://github.com/user-attachments/assets/6713512d-5ea7-4979-ac93-a9b4520e8925" />


## 5 Physical consequences in organic charge transport

| Regime             | Transport Picture               | Experimental Signature                                                              |
|--------------------|----------------------------------|--------------------------------------------------------------------------------------|
| $J \ll 0.1$ eV     | Non-adiabatic hopping (Marcus)   | Arrhenius mobility with $\mu \propto J^2$.                                          |
| $J \sim 0.1–0.3$ eV | Intermediate                   | Mobility deviates from pure Arrhenius; weak field dependence.                       |
| $J \gtrsim 0.3$ eV | Adiabatic transfer               | Rate limited by phonon frequency; mobility plateaus; isotope substitution has small effect. |

**Examples**

Rubrene single crystals: nearest-neighbour $J≈0.14$, $eV$,  $Γ$≈$0.1$ → still non-adiabatic.

Pentacene dimers: $J≈0.25$\,eV, borderline; LZ correction reduces predicted rate by ~3×.

Fullerene PCBM dimer in the “face-on” stack: $J≈0.4$\,$eV$, $Γ≈1–2$ → LZ factor saturates (hops limited by 60 cm⁻¹ lattice phonon).

## 6 Limitations and advanced refinements

- **Multi-mode spectral density**: Real $J$-dependence is integrated over many promoting modes; Eq. (6) uses a single $ω_{eff}$ ω eff.
- **Quantum nuclei**: LZ is semiclassical; quantum tunnel splitting or zero-point motion at cryogenic T require Marcus–Levich–Jortner or full quantum dynamics.
- **Polaron effects**: If $J$ J is large enough to delocalise charge over several sites the underlying Hamiltonian is no longer strictly two-level; one must solve a Holstein–Peierls model or use surface-hopping MD.
- **Disorder**: Energetic and spatial disorder broaden the crossing region; averaging over site pairs often restores the usefulness of the single-parameter $Γ$.

## 7 Take-home rules for using Landau–Zener in simulations

- Compute $Γ$ from Eq. (6) for every pair; store $λ$ λ and $ω_{eff}$ ω eff once.
- Replace Marcus prefactor $J^2$ J2 by $J^2\,[1-\exp(-2\pi\,Γ)]/(\hbar\,ω_{eff}\,λ)$ J2 [1−exp(−2πΓ)]/(ℏω eff λ).
- Ensure units: $J,λ$ J,λ in eV, $ω_{eff}$ ω eff in eV (1 cm⁻¹ = 1.2398×10⁻⁴ eV).
- Use $Γ$-dependent rate (7) in your kinetic Monte-Carlo or master equation; this prevents the unphysical “infinitely fast” hopping at very large $J$ J.
