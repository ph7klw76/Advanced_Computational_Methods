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
    filename = "3.out"
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

extract and draw energy levels
```python
import re

def extract_energies(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    singlet_energies = []
    triplet_energies = []
    
    is_singlet = False
    is_triplet = False
    
    for line in lines:
        line = line.strip()
        if 'TD-DFT/TDA Singlet State Energies' in line:
            is_singlet = True
            is_triplet = False
        elif 'TD-DFT/TDA Triplet State Energies' in line:
            is_singlet = False
            is_triplet = True
        elif 'STATE:' in line:
            match = re.search(r'STATE:\s*([\d\.]+)\s*eV', line)
            if match:
                energy = float(match.group(1))
                if is_singlet:
                    singlet_energies.append(energy)
                elif is_triplet:
                    triplet_energies.append(energy)
    
    return singlet_energies, triplet_energies

filename = 'singlet_triplet_energies.txt'


# Run the function
singlet_data, triplet_data = extract_energies(filename)



import matplotlib.pyplot as plt

def draw_energy_levels(singlet_levels, triplet_levels, offset=0.05):
    fig, ax = plt.subplots()

    # Sort the energy levels
    singlet_levels.sort()
    triplet_levels.sort()

    # Function to add small offset to near-degenerate states
    def add_offset(levels):
        new_levels = []
        for i, level in enumerate(levels):
            count = levels.count(level)
            if count > 1:
                for j in range(count):
                    new_levels.append(level + j * offset - (count - 1) * offset / 2)
            else:
                new_levels.append(level)
        return list(set(new_levels))

    singlet_levels = add_offset(singlet_levels)
    triplet_levels = add_offset(triplet_levels)

    # Draw singlet energy levels
    for i, level in enumerate(singlet_levels):
        ax.hlines(level, xmin=0, xmax=1, color='b', linewidth=2, label='Singlet' if i == 0 else "")
        alignment = 'left' if i % 2 == 0 else 'right'
        ax.text(0.5, level, f"{level:.2f} eV", verticalalignment='bottom', horizontalalignment=alignment, fontsize=16, color='b')

    # Draw triplet energy levels
    for i, level in enumerate(triplet_levels):
        ax.hlines(level, xmin=2, xmax=3, color='r', linewidth=2, label='Triplet' if i == 0 else "")
        alignment = 'left' if i % 2 == 0 else 'right'
        ax.text(2.5, level, f"{level:.2f} eV", verticalalignment='bottom', horizontalalignment=alignment, fontsize=16, color='r')

    # Add labels and title
    ax.set_title('Energy Level Diagram', fontsize=16)
    ax.set_xlabel('Energy Levels', fontsize=16)
    ax.set_ylabel('Energy (eV)', fontsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    
    # Place the legend in the middle
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=16)

    # Show the plot
    plt.show()

# Sample singlet and triplet energy levels in eV
limit=4.0
singlet_levels = [item for item in singlet_data if item < limit]
triplet_levels = [item for item in triplet_data if item < limit]

draw_energy_levels(singlet_levels, triplet_levels)
```
