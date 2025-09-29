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
import math
import os

def calculate_spin_orbit_coupling(filename):
    # Dictionary to store the total spin-orbit coupling for each pair of roots
    total_soc_by_root = {}
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Locate the SOCME section and determine start and end indices dynamically
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if "CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS" in line:
                start_idx = i + 5  # SOCME values start 5 lines below this header
                break

        if start_idx is not None:
            # Dynamically find the end index by looking for non-data lines
            for i in range(start_idx, len(lines)):
                if not lines[i].strip() or not re.match(r'^\s*\d+', lines[i]):  # Non-numeric or empty lines
                    end_idx = i - 1
                    break
            if end_idx is None:  # If no end detected, process till the end of file
                end_idx = len(lines) - 1

            # Process SOCME data between start_idx and end_idx
            for soc_line in lines[start_idx:end_idx + 1]:
                # Split the line and clean up unwanted characters
                values = re.split(r'\s+', soc_line.strip())
                values = [x for x in values if x not in [',', '(', ')']]
                
                if len(values) >= 8:  # Ensure there are enough values for processing
                    try:
                        root0 = values[0]  # Root 0 (T)
                        root1 = values[1]  # Root 1 (S)
                        # Extract real and imaginary parts for Z, X, and Y
                        z = complex(float(values[2]), float(values[3]))
                        x = complex(float(values[4]), float(values[5]))
                        y = complex(float(values[6]), float(values[7]))
                        # Calculate total SOC magnitude
                        total_soc = math.sqrt(abs(z)**2 + abs(x)**2 + abs(y)**2)
                        # Accumulate the SOC value for the root pair
                        total_soc_by_root[(root0, root1)] = total_soc_by_root.get((root0, root1), 0) + total_soc
                    except (ValueError, IndexError):
                        # Handle any parsing errors gracefully
                        continue
        
        return total_soc_by_root

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return {}

if __name__ == "__main__":
    filename = "T1-a-1.out"
    output_filename = "spin_orbit_couplings.txt"
    
    if os.path.exists(filename):
        total_soc_by_root = calculate_spin_orbit_coupling(filename)
        
        # Write results to output file
        with open(output_filename, 'w') as output_file:
            if total_soc_by_root:
                output_file.write("Total Spin-Orbit Coupling for Molecule by Root:\n")
                for (root0, root1), total_soc in total_soc_by_root.items():
                    output_file.write(f"Root pair ({root0}, {root1}): {total_soc:.6f} cm-1\n")
            else:
                output_file.write("No spin-orbit coupling data found in the file.\n")
    else:
        print(f"File {filename} does not exist. Please provide a valid file.")
        

# File paths
input_file_path = filename
output_file_path = 'singlet_triplet_energies.txt'

# Regular expressions to match the start of relevant sections
td_singlet_section_pattern = re.compile(r'TD-DFT/TDA EXCITED STATES \(SINGLETS\)')
td_triplet_section_pattern = re.compile(r'TD-DFT/TDA EXCITED STATES \(TRIPLETS\)')

# Flags to track when within relevant sections
in_singlet_section = False
in_triplet_section = False

# Lists to store extracted energies
singlet_energies = []
triplet_energies = []

# Read the file and process line by line
with open(input_file_path, 'r') as file:
    for line in file:
        if td_singlet_section_pattern.search(line):
            in_singlet_section = True
            in_triplet_section = False
            continue
        elif td_triplet_section_pattern.search(line):
            in_triplet_section = True
            in_singlet_section = False
            continue

        if in_singlet_section or in_triplet_section:
            if line.strip().startswith("STATE"):
                # Extract the fifth element when split by spaces
                parts = line.split()
                if len(parts) > 5:
                    state_energy = float(parts[5])
                    if in_singlet_section:
                        singlet_energies.append((parts[0], state_energy))
                    elif in_triplet_section:
                        triplet_energies.append((parts[0], state_energy))

# Write the filtered results to the output file
with open(output_file_path, 'w') as output_file:
    output_file.write("TD-DFT/TDA Singlet State Energies:\n")
    for state, energy in singlet_energies:
        output_file.write(f"{state}: {energy:.6f} eV\n")
    
    output_file.write("\nTD-DFT/TDA Triplet State Energies:\n")
    for state, energy in triplet_energies:
        output_file.write(f"{state}: {energy:.6f} eV\n")

# Print file path to indicate completion
print(f"Energies extracted and saved to: {output_file_path}")
```

To draw the graph

```python
# -*- coding: utf-8 -*-
"""
Energy-level ladder with spin–orbit coupling arrows (no overlapping arrows or labels).

✔ Quick edits at the USER SETTINGS block:
   - ENERGY & SOC filenames
   - COUPLING_THRESHOLD_CM1
   - ENERGY_LIMIT_EV
   - Label spacing + side alternation (left/right)
   - Arrow curvature (to avoid overlapping arrows)
   - Arrow endpoint inset + per-level fan-out (different start/end points)

Behavior:
- Parses TD-DFT/TDA singlet/triplet energies (eV) and SOC pairs: "Root pair (n, m): xxx cm-1".
- Draws double-headed arrows for SOC > threshold (ignores m=0 if enabled).
- Prevents arrow overlaps by:
    • curved "arc3" paths with alternating curvature,
    • per-level fan-out so multiple arrows attach at different points,
    • slight outward inset so arrowheads don’t touch level lines.
- Labels “xxx cm⁻¹” are bold, strong-colored, 3 significant figures, non-overlapping, and
  clearly connected to their arrow via a colored leader line and a small midpoint marker.
- Labels alternate slightly to the left/right of the center to further improve clarity.
"""

# ========= USER SETTINGS (EDIT THESE) ========================================

ENERGIES_FILE: str = "singlet_triplet_energies.txt"   # TD-DFT/TDA energies file
SOC_FILE: str       = "spin_orbit_couplings1.txt"     # SOC pairs file

COUPLING_THRESHOLD_CM1: float = 0.5    # Only draw arrows if SOC > this value (cm^-1)
ENERGY_LIMIT_EV: float         = 3.80  # Plot only states with E < this limit
IGNORE_M_ZERO: bool            = True  # Ignore SOC entries with m=0

# Label spacing / aesthetics
LABEL_GAP_FRAC: float          = 0.05   # Min vertical gap BETWEEN SOC labels as fraction of span
LABEL_NUDGE_FRAC: float        = 0.015  # Base upward nudge of SOC labels vs arrow midpoint
LABEL_ALTERNATE_SIDES: bool    = True   # Alternate label sides (left/right) to reduce clutter
LABEL_HOFFSET_FRAC: float      = 0.6   # Horizontal offset (fraction of column gap) for labels

# Layout / style
LEFT_CENTER_X: float           = 0.35   # Singlet column x-center
RIGHT_CENTER_X: float          = 0.60   # Triplet column x-center
LINE_HALFWIDTH: float          = 0.06   # Half width of each level's horizontal line
FONTSIZE: int                  = 16     # Base font size

# Arrow curvature control (to prevent overlapping arrows)
CURVE_SNAP_FRAC: float         = 0.005  # Energies within this fraction of span are considered "same"
CURVE_BASE_RAD: float          = 0.18   # Base curvature radius; multiples used: 0, ±rad, ±2rad, ...

# Arrow endpoint control
ARROW_END_INSET_FRAC: float    = 0.1   # Pull arrowheads off level lines (fraction of column gap)
ARROW_ENDPOINT_SPREAD_FRAC: float = 0.2  # Per-level fan-out (fraction of column gap)

# Strong, high-contrast colors (Tableau 10). Cycles if more arrows than colors.
STRONG_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# =============================================================================

import re
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as patheffects


# -------- Parsing energies --------

SINGLET_HDR = re.compile(r"TD-DFT/TDA\s+Singlet State Energies", re.I)
TRIPLET_HDR = re.compile(r"TD-DFT/TDA\s+Triplet State Energies", re.I)
STATE_RE    = re.compile(r"\bSTATE:\s*([+-]?\d+(?:\.\d+)?)\s*eV\b", re.I)

def extract_energies(filename: str) -> Tuple[List[float], List[float]]:
    """Parse singlet/triplet energies (in eV) from a TD-DFT/TDA output text file."""
    singlet, triplet = [], []
    mode = None  # "S", "T", or None

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if SINGLET_HDR.search(line):
                mode = "S"
                continue
            if TRIPLET_HDR.search(line):
                mode = "T"
                continue

            m = STATE_RE.search(line)
            if m:
                e = float(m.group(1))
                if mode == "S":
                    singlet.append(e)
                elif mode == "T":
                    triplet.append(e)

    return singlet, triplet


# -------- Parsing SOC pairs --------
# Accepts integers for n,m; value can be plain or scientific notation; "cm-1" or "cm^-1".
SOC_RE = re.compile(
    r"Root\s+pair\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*cm\^?-?1",
    re.I
)

def parse_soc_pairs(filename: str) -> List[Tuple[int, int, float]]:
    """
    Parse lines like 'Root pair (n, m): 0.331964 cm-1' -> (n, m, value_cm1).
    NOTE: n and m are 1-based indices in the file.
    """
    out: List[Tuple[int, int, float]] = []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = SOC_RE.search(raw)
            if m:
                n = int(m.group(1))
                mm = int(m.group(2))
                val = float(m.group(3))
                out.append((n, mm, val))
    return out


# -------- Utilities for non-overlapping label placement --------

def compute_nonoverlap_positions(targets: List[float], min_gap: float) -> List[float]:
    """
    Given desired y-targets, return adjusted y-positions so neighbors are >= min_gap apart.
    Monotone upward adjustment (never pushes a label below its target).
    """
    items = list(enumerate(targets))          # (original_index, y_target)
    items.sort(key=lambda t: t[1])            # sort by target y

    out = [0.0] * len(targets)
    last = float("-inf")
    for idx, y in items:
        ly = y if y >= last + min_gap else last + min_gap
        out[idx] = ly
        last = ly
    return out


def compute_label_positions(values: List[float], min_gap: float) -> List[float]:
    """For energy level text labels (reused from earlier logic)."""
    items = list(enumerate(values))
    items.sort(key=lambda t: t[1])

    out = [0.0] * len(values)
    last = float("-inf")
    for idx, y in items:
        ly = y if y >= last + min_gap else last + min_gap
        out[idx] = ly
        last = ly
    return out


# -------- Arrow curvature assignment to avoid overlaps --------

def assign_curvatures(
    pairs: List[Tuple[int, int, float]],
    singlet: List[float],
    triplet: List[float],
    span: float,
    snap_frac: float = CURVE_SNAP_FRAC,
    base_rad: float = CURVE_BASE_RAD,
) -> List[float]:
    """
    For arrows that would otherwise overlap (same/very-close endpoints), assign
    different curvature radii so the drawn arcs separate visually.

    - We bin arrows by rounded (yS, yT) using a small snapping tolerance.
    - Within each bin, we cycle radii [0, +r, -r, +2r, -2r, ...].

    Returns a list of 'rad' (connectionstyle arc3 radius) for each pair in order.
    """
    mags = [0.0, base_rad, -base_rad, 2*base_rad, -2*base_rad,
            3*base_rad, -3*base_rad, 4*base_rad, -4*base_rad]

    res = snap_frac * span
    bins: Dict[Tuple[int, int], List[int]] = {}
    yS_list = []
    yT_list = []

    for i, (n, m, _val) in enumerate(pairs):
        yS = singlet[m]
        yT = triplet[n]
        yS_list.append(yS)
        yT_list.append(yT)
        key = (int(round(yS / res)), int(round(yT / res)))
        bins.setdefault(key, []).append(i)

    rads = [0.0] * len(pairs)
    for key, idxs in bins.items():
        # sort by midpoint (optional aesthetic)
        idxs.sort(key=lambda i: 0.5 * (yS_list[i] + yT_list[i]))
        for j, i in enumerate(idxs):
            rads[i] = mags[j % len(mags)]
    return rads


# -------- Plotting --------

def draw_energy_levels(
    singlet_levels: List[float],
    triplet_levels: List[float],
    *,
    couplings: Optional[List[Tuple[int, int, float]]] = None,  # (n, m, value_cm1), 1-based in file -> 0-based here
    coupling_threshold_cm1: float = COUPLING_THRESHOLD_CM1,
    ignore_m_zero: bool = IGNORE_M_ZERO,

    # spacing controls
    min_label_gap_eV: Optional[float] = None,       # for energy level labels
    label_gap_frac: float = LABEL_GAP_FRAC,         # min gap between SOC labels (fraction of span)
    label_base_nudge_frac: float = LABEL_NUDGE_FRAC,# upward nudge of SOC labels vs arrow mid

    # layout parameters
    left_center_x: float = LEFT_CENTER_X,
    right_center_x: float = RIGHT_CENTER_X,
    line_halfwidth: float = LINE_HALFWIDTH,
    fontsize: int = FONTSIZE
) -> None:
    """
    Two-column diagram with energy levels and SOC arrows.
    - Each valid (n, m, val_cm1) gets a strong-colored <-> arrow.
    - Arrows that would overlap are given different curvatures (arc3,rad).
    - Per-level fan-out attaches multiple arrows at different points; endpoints are slightly
      outside the level lines so arrowheads don't touch the lines.
    - Labels are bold, colored, and non-overlapping with leader lines and midpoint markers.
    """

    # Sort ascending to keep indices natural
    singlet = sorted([e for e in singlet_levels if e is not None])
    triplet = sorted([e for e in triplet_levels if e is not None])

    if not singlet and not triplet:
        raise ValueError("No energies to plot.")

    all_vals = singlet + triplet
    ymin0, ymax0 = min(all_vals), max(all_vals)
    span = (ymax0 - ymin0) if ymax0 > ymin0 else 1.0

    # default minimum label gap (~3.5% of span) for energy level labels
    if min_label_gap_eV is None:
        min_label_gap_eV = 0.035 * span

    # Pre-compute energy label positions (non-overlapping)
    singlet_label_y = compute_label_positions(singlet, min_gap=min_label_gap_eV)
    triplet_label_y = compute_label_positions(triplet, min_gap=min_label_gap_eV)

    fig, ax = plt.subplots(figsize=(6, 8))

    # Helper to draw one column of energy levels
    def plot_group(levels, label_y, x_center, color, text_side):
        for y, ly in zip(levels, label_y):
            # energy level line
            ax.hlines(y, x_center - line_halfwidth, x_center + line_halfwidth,
                      color=color, linewidth=2)

            # label and a small leader if nudged
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

    # Draw singlets (left, blue) and triplets (right, red)
    plot_group(singlet, singlet_label_y, left_center_x,  color="tab:blue", text_side="left")
    plot_group(triplet, triplet_label_y, right_center_x, color="tab:red",  text_side="right")

    # Optional: draw SOC double-headed arrows + labels
    max_label_y_seen = ymax0  # track to expand ylim if needed
    if couplings:
        # Filter to valid arrows that are inside plotted lists and above threshold
        valid: List[Tuple[int, int, float]] = []
        for (n1, m1, val) in couplings:
            if ignore_m_zero and m1 == 0:
                continue
            if val <= coupling_threshold_cm1:
                continue
            n = n1 - 1  # triplet index (0-based)
            m = m1 - 1  # singlet index (0-based)
            if 0 <= n < len(triplet) and 0 <= m < len(singlet):
                valid.append((n, m, val))

        if valid:
            # Geometry shared by columns
            x1_edge = left_center_x + line_halfwidth      # singlet right edge
            x2_edge = right_center_x - line_halfwidth     # triplet left edge
            dx_total = x2_edge - x1_edge

            inset = max(0.0, min(ARROW_END_INSET_FRAC * dx_total, 0.45 * dx_total))  # safety clamp
            base_x1 = x1_edge + inset    # base start (right of singlet line)
            base_x2 = x2_edge - inset    # base end   (left  of triplet line)

            # Per-level fan-out: figure out how many arrows touch each singlet/triplet
            singlet_to_is: Dict[int, List[int]] = {}
            triplet_to_is: Dict[int, List[int]] = {}
            for i, (n, m, _v) in enumerate(valid):
                singlet_to_is.setdefault(m, []).append(i)
                triplet_to_is.setdefault(n, []).append(i)

            # Stable ordering within each level to assign slots deterministically
            # Use the energy on the opposite side as the sort key for fan-out ordering
            for m, idxs in singlet_to_is.items():
                idxs.sort(key=lambda i: triplet[valid[i][0]])
            for n, idxs in triplet_to_is.items():
                idxs.sort(key=lambda i: singlet[valid[i][1]])

            # Precompute fan-out slot indices for each arrow index
            s_slot_idx: Dict[int, Tuple[int, int]] = {}  # i -> (k, count)
            t_slot_idx: Dict[int, Tuple[int, int]] = {}
            for m, idxs in singlet_to_is.items():
                count = len(idxs)
                for k, i in enumerate(idxs):
                    s_slot_idx[i] = (k, count)
            for n, idxs in triplet_to_is.items():
                count = len(idxs)
                for k, i in enumerate(idxs):
                    t_slot_idx[i] = (k, count)

            # Spread per slot (keep small relative to gap)
            spread_step = max(0.0, ARROW_ENDPOINT_SPREAD_FRAC * dx_total)

            base_nudge = label_base_nudge_frac * span

            # Assign curvature radii to avoid overlapping arrows (still helpful)
            rads = assign_curvatures(valid, singlet, triplet, span,
                                     snap_frac=CURVE_SNAP_FRAC,
                                     base_rad=CURVE_BASE_RAD)

            # Compute arrow midpoints and base label targets (midpoint + small nudge)
            mids_x = []
            mids_y = []
            lbl_targets = []  # desired y positions before separation

            # Per-arrow start/end x positions with fan-out
            x1_per: List[float] = [base_x1] * len(valid)
            x2_per: List[float] = [base_x2] * len(valid)

            # Assign per-level slot offsets, centered around the base positions
            for i, (n, m, _val) in enumerate(valid):
                # Singlet side: center the k in [-...,+...] around zero
                kS, cS = s_slot_idx[i]
                offsetS = (kS - 0.5 * (cS - 1)) * spread_step
                x1i = base_x1 + offsetS
                # Clamp so we stay outside the singlet line and well before the triplet side
                x1i = max(x1_edge + 0.6 * inset, min(x1i, base_x2 - 0.10 * (base_x2 - base_x1)))
                x1_per[i] = x1i

                # Triplet side
                kT, cT = t_slot_idx[i]
                offsetT = (kT - 0.5 * (cT - 1)) * spread_step
                x2i = base_x2 + offsetT
                # Clamp so we stay outside the triplet line and not too close to singlet side
                x2i = min(x2_edge - 0.6 * inset, max(base_x1 + 0.10 * (base_x2 - base_x1), x2i))
                x2_per[i] = x2i

            # Now compute midpoints/labels based on individualized endpoints
            for i, (n, m, _val) in enumerate(valid):
                yS = singlet[m]
                yT = triplet[n]
                mx = 0.5 * (x1_per[i] + x2_per[i])
                my = 0.5 * (yS + yT)
                mids_x.append(mx)
                mids_y.append(my)
                lbl_targets.append(my + base_nudge)

            # Compute non-overlapping label y-positions
            label_min_gap = label_gap_frac * span
            lbl_y = compute_nonoverlap_positions(lbl_targets, min_gap=label_min_gap)

            # Decide label sides (alternate left/right by vertical order if enabled)
            ordered = sorted(range(len(valid)), key=lambda i: lbl_targets[i])
            side_sign = [1] * len(valid)  # +1 = right of center, -1 = left
            if LABEL_ALTERNATE_SIDES:
                for rank, i in enumerate(ordered):
                    side_sign[i] = 1 if (rank % 2 == 0) else -1

            # Expand ylim later if labels exceed current ymax
            max_label_y_seen = max(max_label_y_seen, max(lbl_y) if lbl_y else ymax0)

            # Draw arrows and labels (each with strong color)
            for i, (n, m, val) in enumerate(valid):
                yS = singlet[m]
                yT = triplet[n]
                mx = mids_x[i]
                my = mids_y[i]
                my_label = lbl_y[i]
                color = STRONG_COLORS[i % len(STRONG_COLORS)]
                rad = rads[i]                     # curvature radius
                sx = side_sign[i]                 # side: +1 right, -1 left
                dx_local = x2_per[i] - x1_per[i]
                hx = LABEL_HOFFSET_FRAC * dx_local  # horizontal offset magnitude

                # double-headed curved arrow in strong color
                ax.add_patch(FancyArrowPatch(
                    (x1_per[i], yS), (x2_per[i], yT),
                    arrowstyle="<->",
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=11,
                    linewidth=1.6,
                    alpha=0.95,
                    color=color,
                    zorder=2
                ))

                # midpoint marker (clarifies which arrow is referenced by the label)
                ax.scatter([mx], [my], s=18, color=color, zorder=3,
                           edgecolors="white", linewidths=0.6)

                # angled leader line from midpoint to label position (same color)
                lx = mx + sx * hx
                ax.plot([mx, lx], [my, my_label],
                        linewidth=1.1, alpha=0.95, color=color, zorder=2)

                # bold label in same strong color, 3 significant figures (+ white halo)
                text_pe = [patheffects.withStroke(linewidth=2.6, foreground="white")]
                ax.text(lx, my_label, f"{val:.3g} cm$^{{-1}}$",
                        ha="center", va="bottom",
                        fontsize=max(11, fontsize - 3),
                        fontweight="bold",
                        color=color,
                        path_effects=text_pe,
                        zorder=3)

    # axes styling and limits (allow room above for labels)
    top_pad = 0.14 * span
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(ymin0 - 0.08 * span, max(max_label_y_seen + top_pad, ymax0 + 0.12 * span))
    ax.set_ylabel("Energy (eV)", fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    # legend
    ax.plot([], [], color="tab:blue", linewidth=2, label="Singlet")
    ax.plot([], [], color="tab:red",  linewidth=2, label="Triplet")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, fontsize=fontsize)

    plt.tight_layout()
    plt.show()


# -------- main --------

def main():
    # Read data
    singlet_data, triplet_data = extract_energies(ENERGIES_FILE)
    soc_pairs = parse_soc_pairs(SOC_FILE)

    # Apply energy cutoff
    singlets = [e for e in singlet_data if e < ENERGY_LIMIT_EV]
    triplets = [e for e in triplet_data if e < ENERGY_LIMIT_EV]

    # Draw
    draw_energy_levels(
        singlets,
        triplets,
        couplings=soc_pairs,
        coupling_threshold_cm1=COUPLING_THRESHOLD_CM1,
        ignore_m_zero=IGNORE_M_ZERO,
        label_gap_frac=LABEL_GAP_FRAC,
        label_base_nudge_frac=LABEL_NUDGE_FRAC
    )


if __name__ == "__main__":
    main()


```
