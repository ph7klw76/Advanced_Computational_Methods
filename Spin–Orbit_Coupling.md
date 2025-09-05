## 1. Foundations of Spin–Orbit Coupling  
### 1.1. Spin and Orbital Angular Momentum  

![image](https://github.com/user-attachments/assets/92c31d9f-bb29-4966-8a18-6f69d599c1a7)

In nonrelativistic quantum mechanics, the Hamiltonian of an electron in an atom or molecule typically includes kinetic energy and potential energy due to the Coulomb interaction with the nucleus (or nuclei in a molecule), but neglects any mixing between the electron’s spin ($S$) and its orbital motion ($L$). Consequently, the total wavefunction $\Psi$ separates into a spatial part $\psi(r)$ and a spin part $\chi(s)$, and spin multiplicities often appear as “selection rules” that forbid transitions between states of different spin in the electric dipole approximation.

In relativistic treatments, or simply more accurate quantum chemical treatments, the motion of electrons in the electric field of a heavy nucleus or heavy atom substituent leads to an effective magnetic field in the electron’s rest frame. The electron’s spin interacts with this magnetic field, creating a coupling between $L$ and $S$. This is the spin–orbit interaction, or spin–orbit coupling (SOC).

![image](https://github.com/user-attachments/assets/ecc67ea3-2340-41c0-b911-e7ad0ed45a62)


## 2. Spin–Orbit Hamiltonian and Its Derivation  
### 2.1. One‐Electron Approximation  
A common model Hamiltonian for spin–orbit coupling in a one‐electron system can be written (in atomic units) as:

$$
\hat{H}_{SO} = \frac{\alpha^2}{2} \sum_i \frac{1}{r_i} \frac{dV}{dr_i} \mathbf{L}_i \cdot \mathbf{S}_i,
$$

where:

- $\alpha \approx 1/137$ is the fine‐structure constant (relativistic factor).  
- $V$ is the potential energy felt by the electron due to the nucleus (or effective core potential).  
- $r_i$ is the radial coordinate of electron $i$.  
- $\mathbf{L}_i$ is the orbital angular momentum operator for electron $i$.  
- $\mathbf{S}_i$ is the spin operator for electron $i$.  

In SI units, one commonly sees a form:

$$
\hat{H}_{SO} = \frac{1}{2m_e^2c^2} \sum_i \frac{1}{r_i} \frac{dV}{dr_i} \mathbf{L}_i \cdot \mathbf{S}_i,
$$

reflecting the same physics.

**Key Idea**: Heavier atoms (larger atomic number $Z$) have stronger nuclear potentials, thus larger $\frac{dV}{dr}$, which increases spin–orbit coupling. This is why heavy‐metal complexes (e.g., Ir, Pt) have large SOC and can display efficient phosphorescence.

### 2.2. Many‐Electron Systems  
For molecules, we extend the one‐electron concept to all electrons. Typically, the total spin–orbit operator is a sum over electrons, each interacting with the molecular potential:


![image](https://github.com/user-attachments/assets/366e59c5-c101-47dc-989f-03705807b8c4)



The exact functional form can get complicated (particularly if we incorporate two‐electron spin–other‐orbit terms), but the principle remains: each electron’s spin couples with its orbital angular momentum in the potential produced by the nuclei and by the rest of the electrons (often approximated via an effective Hamiltonian).

## 3. Spin–Orbit Coupling in Electronic Transitions  
In the electric dipole approximation, transitions between electronic states with different spin are formally forbidden. This is because the transition dipole operator $\hat{\mu}$ does not act on spin coordinates:

$$
\langle \psi_f | \hat{\mu} | \psi_i \rangle = \langle \psi_f(\text{space, spin}) | \hat{\mu} | \psi_i(\text{space, spin}) \rangle.
$$

If $\psi_f$ and $\psi_i$ differ by total spin (e.g., singlet vs. triplet), the spin wavefunctions are orthogonal. Spin–orbit coupling effectively mixes states of different spin multiplicities, partially “lifts” the spin selection rule, and enables otherwise forbidden transitions.

### 3.1. Perturbative Picture  
In a perturbative approach, we treat the spin–orbit Hamiltonian $\hat{H}_{SO}$ as a small perturbation to the nonrelativistic Hamiltonian:

![image](https://github.com/user-attachments/assets/908e0293-b422-4f55-87eb-0002aa1fcd58)


where $\hat{H}_0$ is the usual electronic Hamiltonian (kinetic + Coulomb terms, ignoring spin–orbit). We can expand the true eigenstates $\Phi_n$ of $\hat{H}$ in terms of the eigenstates $\Phi_n^{(0)}$ of $\hat{H}_0$, so that:

$$
\Phi_n = \Phi_n^{(0)} + \sum_{m \neq n} \frac{\langle \Phi_m^{(0)} | \hat{H}_{SO} | \Phi_n^{(0)} \rangle}{E_n^{(0)} - E_m^{(0)}} \Phi_m^{(0)} + \dots
$$

If a singlet state $\Phi_n^{(0)}$ lies close in energy to a triplet state $\Phi_m^{(0)}$, the spin–orbit coupling matrix element $\langle \Phi_m^{(0)} | \hat{H}_{SO} | \Phi_n^{(0)} \rangle$ can be nonzero, mixing singlet and triplet character.

### 3.2. Transition Rate Enhancement  
The transition dipole for a nominally spin‐forbidden transition from an initial state $\Psi_i$ to final state $\Psi_f$ can be written as:

$$
\langle \Psi_f | \hat{\mu} | \Psi_i \rangle = \langle \Phi_f^{(0)} + \delta \Phi_f | \hat{\mu} | \Phi_i^{(0)} + \delta \Phi_i \rangle,
$$

where $\delta \Phi$ terms carry spin admixtures. If $\Phi_i^{(0)}$ was purely triplet and $\Phi_f^{(0)}$ purely singlet, the direct overlap would be zero—but with SOC mixing, the triplet state acquires some fraction of singlet character, leading to a “rescued” nonzero overlap. This can drastically enhance radiative rates for “spin‐forbidden” processes such as phosphorescence.

## 4. Phosphorescence in OLEDs  
Phosphorescence is emission from a triplet excited state $T_1$ to the ground state $S_0$. Without SOC, this process would be highly forbidden and thus extremely weak. But:

- In heavy‐metal complexes (e.g., Ir(III), Pt(II) complexes), the presence of a heavy central metal greatly increases SOC.  
- The triplet state is significantly mixed with nearby singlet configurations, partially relaxing spin‐selection rules.  
- Phosphorescence rates can then be high (lifetimes can be microseconds or even sub‐microseconds).  

This principle is heavily exploited in phosphorescent OLED devices: doping organic host materials with an Ir(III) or Pt(II) complex ensures almost all excitons (both singlet and triplet) funnel into the emissive triplet state of the heavy‐metal complex, yielding internal quantum efficiencies (IQE) up to 100%.

### 4.1. Simplified Rate Equation  
The radiative decay rate for phosphorescence ($k_{\text{phos}}$) can be written (in a simplified form derived from Fermi’s golden rule) as:

![image](https://github.com/user-attachments/assets/530fb9d4-87c1-47d8-8186-e6363914c10e)


where:

- $\hat{H}_{SO}$ mixes the triplet $|T_1\rangle$ with singlet character.  
- $\mu_{\text{eff}}$ is the effective dipole moment after mixing.  
- $\rho(E)$ is the density of final states (roughly constant in a discrete molecular system, but the exact factor can vary).  

While the above expression is highly schematic, it captures the idea that the SOC matrix element $\langle T_1 | \hat{H}_{SO} | S_n \rangle$ is key to unlocking radiative decay from the triplet.

## 5. TADF and Reverse Intersystem Crossing  
### 5.1. Thermally Activated Delayed Fluorescence (TADF)  
Thermally Activated Delayed Fluorescence (TADF) is a photophysical phenomenon that enables purely organic molecules to achieve efficient harvesting of triplet excitons, circumventing the need for heavy metal complexes. This is achieved through the following mechanism:

**Exciton Formation and Energy States**: Upon photon absorption, the molecule transitions to the first singlet excited state ($S_1$). Non-radiative intersystem crossing (ISC) may populate the first triplet state ($T_1$) via spin–orbit coupling (SOC).

**Reverse Intersystem Crossing (rISC)**: If the energy gap between $S_1$ and $T_1$ ($\Delta E_{ST}$) is sufficiently small, typically less than 0.05 eV, thermal energy at ambient conditions ($k_BT \approx 0.025 \, \text{eV}$ at 300 K) can activate reverse intersystem crossing (rISC), allowing excitons in $T_1$ to transition back to $S_1$.

**Delayed Fluorescence**: The $S_1$ state subsequently undergoes radiative decay, resulting in delayed fluorescence (DF) with high internal quantum efficiency (IQE). In ideal conditions, TADF molecules can approach 100% IQE by recycling triplet excitons, enabling enhanced performance in organic light-emitting diodes (OLEDs).

### 5.2. Role of Spin–Orbit Coupling and Vibronic Effects in TADF  
The efficiency of TADF is strongly governed by the electronic structure, spin–orbit coupling (SOC), and vibronic interactions:

**Minimizing the Energy Gap ($\Delta E_{ST}$):**  

- $\Delta E_{ST}$, the singlet–triplet energy difference, is minimized in TADF molecules, often by designing molecules with spatially separated highest occupied molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO). This spatial separation weakens exchange interactions, reducing $\Delta E_{ST}$ to approximately 0.02–0.05 eV, making rISC thermally accessible.

**Spin–Orbit Coupling and Vibronic Coupling:**  

- While SOC is inherently weak in organic molecules due to the absence of heavy atoms, the introduction of intramolecular charge-transfer (CT) states enhances triplet–singlet mixing. This is achieved via second-order vibronic coupling mechanisms involving intermediate states such as local triplet $^3$LE and CT triplet ($^3$ CT) states.  
- Vibronic coupling between $^3$LE and $^3$CT states mediates transitions to the singlet CT state ($^1$CT) via SOC. This two-step mechanism aligns with quantum dynamic simulations and experimental photophysical studies.

**Molecular and Environmental Factors:**  

- Donor–acceptor (D–A) molecular architectures are employed to induce strong CT character, facilitating reduced $\Delta E_{ST}$ and enhanced SOC.  
- The relative positioning of $^3$LE, $^3$CT, and $^1$CT energy levels, modulated by host polarity, rigidity, and external perturbations, critically affects TADF efficiency. Resonance between $^3$LE and $^3$CT maximizes rISC and delayed fluorescence.

**Advanced Insights from Vibronic and Spin–Orbit Coupling Models**  
Recent experimental and theoretical studies highlight the importance of second-order vibronic coupling and the relative energy alignments of excited states:

- **Vibronic Coupling Efficiency:** Resonance between $^3$LE and $^3$CT states enhances non-adiabatic coupling, significantly increasing the rISC rate to $^1$CT.  
- **Host Effects:** Environmental factors such as host polarity and rigidity shift energy levels of the CT states relative to $^3$LE, leading to distinct TADF regimes. In the optimal Type II regime, $^3$LE is nearly degenerate with $^3$CT, facilitating efficient triplet harvesting.  
- **Design Strategies:** Molecular designs targeting orthogonal D–A orientations further minimize $\Delta E_{ST}$, while suppressing non-radiative losses, critical for maintaining high TADF efficiency.


ORCA to calculate Spin-orbit coupling with DOSOC TRUE keyword
```text
! DEF2-SVP CPCMC(toluene)
%TDDFT  NROOTS  20
        DOSOC   TRUE         
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.0645
END
%maxcore 2000
%pal nprocs 16 end
* XYZFILE 0 1 30.06454915028125263.xyz
```

To extract out the spin-orbit coupling use the python code below along with singlet and triplet energy

```python
import re
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -------- Parsing --------

SINGLET_HDR = re.compile(r"TD-DFT/TDA\s+Singlet State Energies", re.I)
TRIPLET_HDR = re.compile(r"TD-DFT/TDA\s+Triplet State Energies", re.I)
STATE_RE    = re.compile(r"\bSTATE:\s*([+-]?\d+(?:\.\d+)?)\s*eV\b", re.I)

def extract_energies(filename: str) -> Tuple[List[float], List[float]]:
    singlet, triplet = [], []
    mode = None  # "S", "T", or None

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if SINGLET_HDR.search(line):
                mode = "S"; continue
            if TRIPLET_HDR.search(line):
                mode = "T"; continue

            m = STATE_RE.search(line)
            if m:
                e = float(m.group(1))
                if mode == "S":
                    singlet.append(e)
                elif mode == "T":
                    triplet.append(e)

    return singlet, triplet

# -------- Label placement (non-overlapping) --------

def compute_label_positions(values: List[float], min_gap: float) -> List[float]:
    """Adjust label y-positions so neighbors are at least min_gap apart."""
    items = list(enumerate(values))           # (original_index, y)
    items.sort(key=lambda t: t[1])            # sort by y

    out = [None] * len(values)
    last = float("-inf")
    for idx, y in items:
        ly = y if y >= last + min_gap else last + min_gap
        out[idx] = ly
        last = ly
    return out

# -------- Plotting --------

def draw_energy_levels(
    singlet_levels: List[float],
    triplet_levels: List[float],
    *,
    min_label_gap_eV: Optional[float] = None,
    # Column centers: halved spacing (0.25 → 0.125 gap on each side, centers 0.35 & 0.60)
    left_center_x: float = 0.35,
    right_center_x: float = 0.60,
    line_halfwidth: float = 0.06,
    fontsize: int = 16   # match your original font size everywhere
) -> None:
    """Two-column diagram with auto-nudged, non-overlapping labels + leader lines."""
    singlet = sorted([e for e in singlet_levels if e is not None])
    triplet = sorted([e for e in triplet_levels if e is not None])

    if not singlet and not triplet:
        raise ValueError("No energies to plot.")

    all_vals = singlet + triplet
    ymin0, ymax0 = min(all_vals), max(all_vals)
    span = (ymax0 - ymin0) if ymax0 > ymin0 else 1.0

    # default minimum label gap (~3.5% of span)
    if min_label_gap_eV is None:
        min_label_gap_eV = 0.035 * span

    def plot_group(ax, levels, x_center, color, text_side):
        label_y = compute_label_positions(levels, min_gap=min_label_gap_eV)
        for y, ly in zip(levels, label_y):
            # energy level line
            ax.hlines(y, x_center - line_halfwidth, x_center + line_halfwidth,
                      color=color, linewidth=2)

            # label position and leader line
            if text_side == "left":
                tx = x_center - line_halfwidth - 0.02
                ha = "right"
                x_end = x_center - line_halfwidth
            else:
                tx = x_center + line_halfwidth + 0.02
                ha = "left"
                x_end = x_center + line_halfwidth

            if abs(ly - y) > 1e-12:
                ax.add_patch(FancyArrowPatch((tx, ly), (x_end, y),
                                             arrowstyle='-',
                                             mutation_scale=8,
                                             linewidth=0.8,
                                             color=color))
            ax.text(tx, ly, f"{y:.2f} eV", ha=ha, va="center",
                    fontsize=fontsize, color=color)

    fig, ax = plt.subplots(figsize=(4, 6))
    plot_group(ax, singlet, left_center_x,  color="tab:blue", text_side="left")
    plot_group(ax, triplet, right_center_x, color="tab:red",  text_side="right")

    # axes styling
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(ymin0 - 0.08*span, ymax0 + 0.10*span)
    ax.set_ylabel("Energy (eV)", fontsize=fontsize)     # larger y-axis title
    ax.tick_params(axis='y', labelsize=fontsize)        # larger tick labels
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    # legend (same font size as original)
    ax.plot([], [], color="tab:blue", linewidth=2, label="Singlet")
    ax.plot([], [], color="tab:red",  linewidth=2, label="Triplet")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, fontsize=fontsize)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "singlet_triplet_energies.txt"
    singlet_data, triplet_data = extract_energies(filename)

    limit = 3.80
    singlets = [e for e in singlet_data if e < limit]
    triplets = [e for e in triplet_data if e < limit]

    draw_energy_levels(singlets, triplets)
```
