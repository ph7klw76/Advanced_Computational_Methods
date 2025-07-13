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
v\approx q_0\,ω_{eff}=\frac{2\lambda}{k}.
$$


Combining, the Landau–Zener adiabaticity parameter becomes

$$
Γ=\frac{J^2}{\hbar ω_{eff} λ}.
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

## e⁻ / h⁺ Carrier-Specific Effective Frequency

The carrier-specific effective frequency for an electron (anion) or a hole (cation) polaron in an organic semiconductor is defined as:

$$
\hbar \, \Omega_{\text{eff}}^{(e/h)} = \frac{\sum_i S_i^{(e/h)} \, \hbar \omega_i}{\sum_i S_i^{(e/h)}}
$$

---

## 1 Theory Recap — Why This $\Omega_{\text{eff}}$ Works

### Huang–Rhys Factor

$$
S_i = \frac{\Delta Q_i^2 \, \omega_i}{2\hbar}
$$

where $\Delta Q_i$ is the **mass-weighted displacement** between equilibrium geometries along normal mode $i$ of the initial state.

### Intramolecular Reorganisation Energy

$$
\lambda_{\text{intra}} = \sum_i S_i \, \hbar \omega_i
$$

### Single-Mode Mapping

Replacing the full set of modes by a single “effective” Einstein mode $(\tilde{\omega}, \tilde{S})$ must preserve both $\lambda$ and $S$:

$$
\tilde{S} = \sum_i S_i, \quad \lambda = \tilde{S} \, \hbar \tilde{\omega}
$$

Which implies:

$$
\tilde{\omega} = \frac{\sum_i S_i \, \omega_i}{\sum_i S_i}
$$

ℏ$\tilde{\omega}$ expressed in **electron-volts** is what your **KMC code expects as `OMEGA_eff_eV`**. The procedure is identical for an electron (anion) and a hole (cation); the **normal-mode data just come from different charge states**.

---

## 2 ORCA Calculations

Below, `"XYZ"` refers to the Cartesian coordinates section of the ORCA input.

| **Task**                 | **ORCA Input Header**                                  | **Notes**                                                                                  |
|--------------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Neutral geometry + freq  | `! wb97x-d3 def2-TZVP TightSCF Opt Freq`              | Any DFT/dispersion level you trust for π-systems is fine. Keep identical orientation to avoid spurious rotations in the Duschinsky step. |
| Cation (hole)            | `! wb97x-d3 def2-TZVP TightSCF Opt Freq` <br> `* xyz +1 2 …` | Charge = +1, multiplicity = 2 (singlet background). |
| Anion (electron)         | `* xyz -1 2`                                           | Charge = –1. |

Each job writes a `.hess` file (e.g. `neutral.hess`, `cation.hess`, etc.), which contains the normal mode and displacement data required for computing $\Omega_{\text{eff}}$.


## 2 Feed Each Pair of Hessians to the ESD Module

The **Excited-State Dynamics module** (`ORCA_ESD`) produces all **Duschinsky data**, **Huang–Rhys factors** $S_i$, and **per-mode reorganisation energies** in a single run when you set:

```txt
! ESD
%esd
   HESS1  neutral.hess
   HESS2  cation.hess
   USEJ   TRUE          # build J and K
   PRINTLEVEL 3         # list S_i, λ_i etc. :contentReference[oaicite:1]{index=1}
end
* xyzfile 0 1 neutral.xyz   # just to give ESD the equilibrium geom.
```

Repeat with anion.hess for the electron.

Running orca esd_hole.inp (or esd_elec.inp) generates an output block
“Geometry rotation and Duschinsky matrices” and an auxiliary file
esd_duschinsky.dat that contains, for every vibrational mode i of the initial state.

## 3 Compute Ω<sub>eff</sub> from the ESD Output

A tight 15-line Python helper converts that table to Ω<sub>eff</sub> (in eV)  
plus total Huang–Rhys strength ΣS and intramolecular λ:

```python
import numpy as np, re, sys, pathlib

def omega_eff_esd(datfile: pathlib.Path):
    ω_cm, S = [], []
    for line in datfile.read_text().splitlines():
        m = re.match(r'\s*\d+\s+([\d.]+)\s+\S+\s+\S+\s+([\d.]+)', line)
        if m:                       # columns: ω_i  …  S_i
            ω_cm.append(float(m.group(1)))
            S.append(float(m.group(2)))
    ω = np.array(ω_cm) * 2*np.pi*2.99792458e10   # (rad s⁻¹)
    S = np.array(S)
    hbar, e = 1.054571817e-34, 1.602176634e-19
    Ω_eff_eV = hbar * (S*ω).sum()/S.sum() / e
    λ_eV     = hbar * (S*ω).sum() / e           # should equal Σλ_i
    return Ω_eff_eV, S.sum(), λ_eV

for lab in ["hole", "electron"]:
    Ω, ΣS, λ = omega_eff_esd(pathlib.Path(f"esd_{lab}/esd_duschinsky.dat"))
    print(f"{lab:8s}:  Ω_eff = {Ω*1e3:.2f} meV   ΣS = {ΣS:.2f}   λ = {λ*1e3:.1f} meV")
```

