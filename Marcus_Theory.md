# Marcus Theory: Understanding Electron and Hole Transport Through Hopping in Organic Semiconductors

## Abstract
Marcus Theory, originally formulated to describe electron transfer reactions in polar solvents, is a cornerstone for understanding charge transport in organic materials. Because of their disordered nature and relatively localized electronic states, organic semiconductors often rely on thermally activated hopping mechanisms for electron or hole conduction. In this comprehensive technical blog, we will:

- Introduce the conceptual foundations of Marcus Theory  
- Derive the key equations step by step  
- Discuss how these equations apply to electron and hole transport in organic semiconductors  
- Highlight the roles of reorganization energy, transfer integrals, and energetic disorder  

By the end, you will understand how Marcus Theory quantifies charge transfer rates, how it connects microscopic parameters to macroscopic charge transport, and why it is so crucial for designing high-performance organic electronic devices.

---

## 1. Introduction to Marcus Theory

### 1.1 Historical Context and Relevance
- R. A. Marcus developed a model (awarded the Nobel Prize in Chemistry, 1992) to describe electron transfer in redox reactions.  
- The theory is equally applicable to intra- and intermolecular electron transfer, including electron/hole hopping in organic semiconductors.  
- It incorporates solvent (or lattice) reorganization, electronic coupling, and the thermodynamics of the transfer event into a quantitative rate expression.

### 1.2 Hopping Transport in Organic Semiconductors
- Organic semiconductors often have significant electronic disorder, leading to localized charge carriers (electrons or holes).  
- The conduction proceeds when carriers “hop” between neighboring sites (molecules or polymer segments). This hopping is:
  - **Thermally activated**  
  - **Rate-limited** by the overlap of electronic states and the nuclear (molecular or lattice) reorganization that accompanies the transfer of charge from one site to another.  

Marcus Theory provides a semi-classical framework for calculating the rate of this thermally activated hopping process.

---

## 2. Theoretical Foundations of Marcus Theory

### 2.1 Model System and Assumptions
Consider two sites, donor (D) and acceptor (A). An electron (or hole) resides initially on the donor site. The system is characterized by:

- **Electronic coupling** $V$ (sometimes denoted $J$) between the donor and acceptor states.  
- **Reorganization energy** $\lambda$, the energy required to reorganize the molecular and lattice environment from one electronic configuration to another.  
- **Free energy change** $\Delta G_0$ (often written as $\Delta G$) for the electron transfer.  

In organic semiconductors under equilibrium, $\Delta G_0$ may be small (near zero) if the donor and acceptor sites are energetically similar. If there is an energy offset, $\Delta G_0$ will reflect that difference.

---

## 3. Step-by-Step Derivation of the Marcus Rate Equation

### 3.1 Potential Energy Surfaces and Reaction Coordinate

#### Reaction Coordinate:
Marcus Theory posits that the electron transfer event can be described along a single (collective) reaction coordinate $Q$, representing all nuclear degrees of freedom (internal modes + environmental polarization).

#### Parabolas Representation:
We assume harmonic free-energy surfaces for the initial (donor-occupied) and final (acceptor-occupied) states as a function of $Q$. These two parabolas intersect at the activation point.

#### Key Energies:
- **$\lambda$ (Reorganization Energy):** The energy to deform the environment from its equilibrium state of the reactant to that of the product, without transferring the electron.  
- **$\Delta G_0$:** The free energy difference between the donor and acceptor states at their respective equilibrium coordinates.

#### In a one-dimensional depiction:

![image](https://github.com/user-attachments/assets/2898dd4b-9453-4220-b80f-7e1d767ba3e6)

# Marcus Theory: Understanding Electron and Hole Transport Through Hopping in Organic Semiconductors

## The Parabolas and Key Components
- The parabolas labeled (D) and (A) correspond to the free energy surfaces for donor and acceptor states, respectively.  
- $Q^*$ is the reaction coordinate at the intersection (activation point).  
- (D’) is the displaced parabola for the donor state shifted to the acceptor configuration, emphasizing the reorganization.

---

## 3.2 Activation Energy in Marcus Theory

### 3.2.1 Geometric Relationship for Activation Energy
From the parabolas, one obtains the activation free energy $\Delta G^\ddagger$:

$$
\Delta G^\ddagger = \frac{(\lambda + \Delta G_0)^2}{4\lambda}.
$$

**Derivation Outline:**

- Each parabola can be written (classically) as:

$$
G_i(Q) = \frac{k}{2} (Q - Q_i^0)^2 + G_i^0,
$$

where $k$ is the force constant related to the reorganization, $Q_i^0$ is the equilibrium coordinate for state $i$, and $G_i^0$ is the free energy at that equilibrium point.

- The reorganization energy $\lambda$ relates to the shift between these two parabolas:
  
$$
\lambda = \frac{k}{2} (Q_A^0 - Q_D^0)^2.
$$

- $\Delta G_0$ is the difference between $G_A^0$ and $G_D^0$.  

- By equating $G_D(Q^\ddagger) = G_A(Q^\ddagger)$ at the intersection $Q^\ddagger$, solving for $G_D(Q^\ddagger)$ yields:
  
$$
  \Delta G^\ddagger = \frac{(\lambda + \Delta G_0)^2}{4\lambda}.
$$

This expression indicates how the activation barrier depends quadratically on $(\lambda + \Delta G_0)$.

---

## 3.3 Rate Expression: Marcus Golden Rule

### 3.3.1 Electronic Coupling and the Golden Rule
In semi-classical Marcus Theory, the electron transfer rate $k_{\text{ET}}$ is given by a Fermi’s Golden Rule-type expression:

$$
k_{\text{ET}} = \frac{|V|^2}{\hbar} F(\Delta G^\ddagger),
$$

where:
- $\frac{|V|^2}{\hbar}$ represents the electronic transition probability per unit time in the weak coupling (non-adiabatic) limit, and  
- $F(\Delta G^\ddagger)$ is a thermal factor describing how nuclear motion surmounts the activation barrier $\Delta G^\ddagger$.

### 3.3.2 Marcus Rate Equation
Combining the Arrhenius-like factor for crossing the free-energy barrier $\Delta G^\ddagger$ with the expression for electronic coupling results in the Marcus Rate:

$$
k_{\text{ET}} = \frac{|V|^2}{\hbar} \sqrt{\frac{\pi}{\lambda k_B T}} \exp\left(-\frac{(\lambda + \Delta G_0)^2}{4\lambda k_B T}\right).
$$

**Step-by-Step Reasoning:**
1. The transition state theory factor is approximately:
   $$
   \exp\left(-\frac{\Delta G^\ddagger}{k_B T}\right).
   $$

2. Quantum mechanical treatment of vibrational modes near the crossing point introduces the $\sqrt{\frac{\pi}{\lambda k_B T}}$ factor, reflecting the nuclear density of states at the intersection.

3. In the high-temperature (classical) limit, the final expression simplifies to the form above.

---

## 4. Applying Marcus Theory to Organic Semiconductors
In organic semiconductors, electron (n-type) or hole (p-type) transport often occurs via a polaronic or localized state. Marcus Theory is employed to calculate site-to-site hopping rates:

$$
k_{ij} = \frac{|V_{ij}|^2}{\hbar} \sqrt{\frac{\pi}{\lambda_{ij} k_B T}} \exp\left(-\frac{(\lambda_{ij} + \Delta G_{ij}^0)^2}{4\lambda_{ij} k_B T}\right),
$$

where:
- $i$ and $j$ label neighboring molecules or localized sites,  
- $V_{ij}$ is the transfer integral,  
- $\lambda_{ij}$ is the reorganization energy for the charge to move between sites $i$ and $j$,  
- $\Delta G_{ij}^0$ is the free-energy difference (often related to site energy offsets or disorder).

---

### 4.1 Reorganization Energy in Organic Molecules
The total reorganization energy $\lambda$ can be partitioned into internal and external components:
- **Internal ($\lambda_{\text{int}}$):** Molecular geometry changes upon oxidation or reduction (e.g., bond length changes).  
- **External ($\lambda_{\text{ext}}$):** Environmental polarization (e.g., crystal packing, dielectric screening).  

In organic crystals or thin films, $\lambda_{\text{ext}}$ depends on the local dielectric environment. Smaller $\lambda$ leads to lower activation barriers and faster rates.

---

### 4.2 Transfer Integral ($V$)
The electronic coupling $V_{ij}$ between neighboring sites depends on:
- Orbital overlap (e.g., $\pi$–$\pi$ stacking),  
- Molecular orientation,  
- Distance between the sites,  
- Material-specific wavefunction features.  

Sophisticated quantum-chemical or tight-binding methods can estimate $V_{ij}$.

---

### 4.3 Influence of Disorder
Real organic semiconductors exhibit energetic (diagonal) and off-diagonal disorder:
- **Energetic (diagonal) disorder** influences $\Delta G_{ij}^0$ by shifting site energies.  
- **Off-diagonal disorder** modifies $V_{ij}$.  

Hence, a realistic Marcus-based simulation of charge transport typically integrates the rate expression into kinetic Monte Carlo or master equation frameworks across a lattice of sites with random energetic offsets and variable transfer integrals.

---

## 5. Extended Versions of Marcus Theory

### 5.1 Marcus–Levich–Jortner Theory
In systems where high-frequency intramolecular vibrational modes require a quantum treatment, Marcus–Levich–Jortner Theory refines the expression by including discrete vibrational quantum levels, leading to a Franck–Condon factor:

$$
k_{\text{MLJ}} = \frac{|V|^2}{\hbar} \sum_{m=0}^\infty \frac{S^m}{m!} \exp(-S) \exp\left[-\frac{(\lambda + \Delta G_0 + m\hbar\omega)^2}{4\lambda k_B T}\right],
$$

where:
- $S$ is the Huang–Rhys factor related to electron–phonon coupling, and  
- $\hbar\omega$ is the vibrational quantum of the relevant mode.

---

## 6. Practical Steps to Implement Marcus Theory Calculations

### Compute Reorganization Energy $\lambda$:
- **Internal ($\lambda_{\text{int}}$):** Via quantum-chemical geometry optimization of neutral and charged states (and cross-optimizations).  
- **External ($\lambda_{\text{ext}}$):** Via polarized continuum models or explicit crystal environment.

### Compute Transfer Integral $|V_{ij}|$:
- Fragment orbital approach, extended Hückel, or DFT-based wavefunction analysis.  
- Evaluate for relevant molecular pairs in the crystal packing or polymer chain segments.

### Evaluate Site Energies ($\Delta G_{ij}^0$):
- Extract from partial charges, electrostatic potentials, or classical/quantum mechanical embedding of the local environment.

---

## 7. Summary and Outlook
Marcus Theory offers a powerful, physically transparent framework for describing electron and hole hopping in organic semiconductors. By breaking down the problem into:
- **Electronic coupling**  
- **Reorganization energy**  
- **Thermal activation**  
- **Free-energy driving force**  

One can calculate site-to-site hopping rates that capture the essential physics of charge localization, molecular reorganization, and the role of energetic disorder.

### Key Takeaways:
- Smaller reorganization energy $\lambda$ and larger transfer integral $V$ favor faster charge transport.  
- If $\Delta G_0 \approx 0$, the activation barrier is minimized when $\lambda$ is also small.  

Marcus Theory remains indispensable for guiding synthetic strategies and understanding the interplay between structure and electronic properties in organic semiconductors.

# Marcus Rate

### $K_{ij}$ (or $\nu_{ij}$):

In Marcus theory, the rate for a charge to hop from site $i$ to site $j$ is given by an expression (in its simplest form) like:

$$
K_{ij} = \nu_0 \exp[-\beta \Delta G_{ij}^\dagger],
$$

where:

- $\nu_0$ is a prefactor (attempt frequency),
- $\beta = 1 / (k_B T)$, where $k_B$ is Boltzmann’s constant and $T$ is temperature,
- $\Delta G_{ij}^\dagger$ is the free energy barrier for hopping from site $i$ to $j$.

More detailed forms of the Marcus rate can account for reorganization energy $\lambda_\text{reorg}$ and the energy difference $\Delta E_{ij}$ between sites $i$ and $j$.

---

## Exponential Waiting Time and $\lambda$:

When we perform a kinetic Monte Carlo (KMC) simulation, we often pick the waiting time $t$ for an event (in this case, a hop) from an exponential distribution:

$$
P(t; \lambda) = \lambda e^{-\lambda t},
$$

where $\lambda$ is the total rate of an event occurring. In a KMC simulation of hopping transport, $\lambda$ is effectively the sum of all possible outgoing rates from the current site (or the relevant subset if the model enforces only a single possible hop at a time).

If a site $i$ has possible hops to several neighboring sites $j \in \{1, 2, \dots\}$ with rates $K_{ij}$, then the total rate for a hop out of site $i$ is:

$$
\Lambda_i = \sum_j K_{ij}.
$$

The waiting time for the next hop out of site $i$ is then drawn from:

$$
P(t; \Lambda_i) = \Lambda_i e^{-\Lambda_i t}.
$$

---

## Is $\lambda$ the same as $1 / K_{ij}$?

If we are talking about a single transition $i \to j$ and ignoring any other possible transitions from $i$, then the event “a hop from $i$ to $j$ occurs” has a rate $K_{ij}$. The waiting time for that specific event follows an exponential distribution with parameter $K_{ij}$. Hence, for that single hop event, $\lambda = K_{ij}$. In that case, the mean waiting time for the hop $i \to j$ is:

$$
\langle t \rangle = \frac{1}{K_{ij}}.
$$

If there are multiple possible transitions from the current site $i$, then the KMC algorithm typically considers the total rate $\Lambda_i = \sum_j K_{ij}$. We sample the waiting time from $\Lambda_i$ and then choose a particular $j$ among the possible final sites with probabilities proportional to $K_{ij}$. In that more general case:

$$
\lambda \equiv \Lambda_i = \sum_j K_{ij},
$$

which is not necessarily equal to $1 / K_{ij}$, but rather the sum of all available hopping rates from site $i$. After we pick the waiting time (from $\Lambda_i$), the specific hop $i \to j$ is chosen according to:

$$
P(\text{hop is } i \to j) = \frac{K_{ij}}{\Lambda_i}.
$$

---

## Summary

### For a single hop event:
- **Waiting time:** $t \sim \text{Exp}(K_{ij})$,
- **Rate:** $\lambda = K_{ij}$,
- **Mean waiting time:** $\langle t \rangle = \frac{1}{K_{ij}}$.

### For multiple possible hops from one site:
- **Waiting time:** $t \sim \text{Exp}(\Lambda_i)$,
- **Total rate:** $\Lambda_i = \sum_j K_{ij}$.

In this case, $\lambda \neq 1 / K_{ij}$; it is the sum of all transition rates out of the site.

---

## Conclusion

Therefore, if your question is: “Is $\lambda$ the $1 / K_{ij}$ rate of the Marcus equation?”—the more precise statement is:

- When there is only a single possible hop (rate $K_{ij}$), $\lambda = K_{ij}$, so the mean waiting time $\langle t \rangle$ is $1 / K_{ij}$.
- Generally, in a KMC approach with multiple final states, $\lambda = \sum_j K_{ij}$, i.e., the total rate out of site $i$.

Hence, $\lambda$ in the exponential waiting time distribution usually refers to the total outgoing rate from the current state in your simulation, which may (or may not) equal a single hop rate $K_{ij}$ depending on how many transitions are possible from that state.


# 1. The Marcus Rate Equation (Refresher)

Marcus theory provides the charge-transfer (CT) rate $k$ for an electron (or hole) hopping from one site (molecule) to another. In its simplest form for non-adiabatic reactions:

$$
k = \frac{\pi}{\lambda k_B T} V^2 \exp \left[ - \frac{( \Delta G + \lambda )^2}{4 \lambda k_B T} \right],
$$

where:

- $V$ is the electronic coupling (transfer integral) between initial and final sites,
- $\lambda$ is the reorganization energy (the energy required to reorganize the nuclear coordinates surrounding the charge),
- $\Delta G$ is the free-energy difference between the initial and final electronic states,
- $k_B$ is Boltzmann’s constant,
- $T$ is the temperature.

---

# 2. Why One Often Sets $\Delta G \approx 0$ in Organic Semiconductors

### Homogeneous Material & Symmetry
In a device or film made of the same organic semiconductor (same molecule) with minimal doping gradients or big energetic offsets, each localized site (molecule) can be thought of as having roughly the same electrochemical potential. Thus, the net free-energy difference for moving a charge from one site to an equivalent neighboring site is close to zero.

### Large Positional and Energetic Disorder
- **Energetic Disorder:** Because the local environment (e.g., conformations, slight variations in packing) fluctuates, the local site energies vary in a “Gaussian-like” distribution. The effect of this distribution on $\Delta G$ can often be effectively averaged out, especially if there is no systematic bias.
- **Positional Disorder:** Real organic films are not perfectly crystalline. The structural randomness averages out the site-to-site energy differences over many hops.

In such cases, one focuses on the distribution of transfer integrals $V$ and reorganization energies $\lambda$, rather than $\Delta G$ itself.

### Thermal Fluctuations
If $\Delta G$ is smaller than a few $k_B T$, the exponential factor in Marcus theory is mainly determined by $\lambda$ (the reorganization energy). As a result, ignoring a small $\Delta G$ is practically valid.

---

# 3. When $\Delta G$ Might Matter

### Doped or Strongly Polar Environments
If you introduce doping or if there is an external electric field (e.g., a strong built-in field in a device stack), local site energies can shift significantly, making $\Delta G$ non-negligible.

### Molecular Heterojunctions
In blends or heterostructures (e.g., donor–acceptor systems), the difference in HOMO/LUMO levels between two distinct molecules can produce a substantial $\Delta G$.

### Near Device Interfaces
At an electrode interface or at grain boundaries, local energetic offsets can be large enough that ignoring $\Delta G$ underestimates the barrier for charge injection or extraction.

# Conclusion

In typical bulk organic semiconductor modeling where each hopping site is roughly equivalent, $\Delta G \approx 0$ is a standard and often justified approximation. The rationale hinges on the facts that local energy variations due to positional and energetic disorder average out and that the reorganization energy $\lambda$ plus thermal effects usually dominate the activation barrier.

However, whether you can safely ignore $\Delta G$ depends on the specific physical context—especially in cases with doping gradients, large electric fields, or heterogeneous blends where $\Delta G$ can be substantial. Always compare $\Delta G$ to $\lambda$ and $k_B T$; if it is small by comparison, you can set it aside in your Marcus rate calculations for most practical purposes.


System with  nearly degnerate energy level


```python
import random
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ---------------------------
# Global Constants
# ---------------------------
hbar = 1.0545718e-34   # Reduced Planck's constant (Joule·s)
kB = 1.380649e-23      # Boltzmann constant (Joule/K)
T = 300                # Temperature in Kelvin
q = 1.60218e-19        # Elementary charge (Coulomb)
eV_to_Joule = 1.60218e-19

# ---------------------------
# Problem-Specific Globals
# ---------------------------
DELTA_E_EV = 0.10        # LUMO–LUMO+1 gap, in eV
lambda_ij_eV = 0.044     # Reorganization energy in eV
disorder_std_eV = 0.12   # Disorder standard deviation in eV

# Convert energies to Joules
lambda_ij = lambda_ij_eV * eV_to_Joule

# ---------------------------
# File paths
# ---------------------------
input_file = '5electronmodel.txt'
output_file = '5electronmodelwith_kij.txt'

# ---------------------------
# Utility Functions
# ---------------------------

def occupancy_LUMO_plus_1(T=300):
    """
    Returns the equilibrium probability that an electron
    resides in LUMO+1, given a global DELTA_E_EV between LUMO and LUMO+1.

    Parameters
    ----------
    T : float
        Temperature in Kelvin

    Returns
    -------
    float
        Probability p(LUMO+1) at temperature T
    """
    # Boltzmann constant in eV/K
    kB_eV_per_K = 8.617333262145e-5
    # Occupancy factor using the global DELTA_E_EV
    return 1.0 / (1.0 + np.exp(DELTA_E_EV / (kB_eV_per_K * T)))


def calc_marcus_rate(Vij, Delta_G_ij, T, rate_factor):
    """
    Calculate the Marcus ET (electron transfer) rate k_{i->j} for
    a given coupling Vij (in Joules), free-energy difference Delta_G_ij (J),
    global reorganization energy lambda_ij (J), and temperature T (K).
    'rate_factor' is a precomputed constant = sqrt(pi / (lambda_ij * kB * T)).
    """
    # Pre-factor: (Vij^2 / hbar)
    prefactor = (Vij**2) / hbar

    # Exponential factor from Marcus theory
    exp_factor = np.exp(
        -((lambda_ij + Delta_G_ij)**2) / (4 * lambda_ij * kB * T)
    )

    return prefactor * rate_factor * exp_factor


def generate_Delta_G_list():
    """
    Generate the 4 free-energy differences, using:
      1) N(0.0, disorder_std_eV)
      2) N(+DELTA_E_EV, disorder_std_eV)
      3) N(-DELTA_E_EV, disorder_std_eV)
      4) N(0.0, disorder_std_eV)

    Returns them in Joules.
    """
    return [
        np.random.normal(0.0,        disorder_std_eV) * eV_to_Joule,
        np.random.normal(DELTA_E_EV, disorder_std_eV) * eV_to_Joule,
        np.random.normal(-DELTA_E_EV,disorder_std_eV) * eV_to_Joule,
        np.random.normal(0.0,        disorder_std_eV) * eV_to_Joule
    ]


# ---------------------------
# 1. Compute & Write Rates
# ---------------------------
def first(input_file, output_file):
    """
    Reads each line of input_file, computes 4 Marcus rates,
    and writes them to output_file using the global DELTA_E_EV.
    """
    # Precompute the constant factor: sqrt(pi / (lambda_ij * kB * T))
    rate_factor = np.sqrt(np.pi / (lambda_ij * kB * T))

    output_lines = []
    
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            # Convert the 4 couplings (in eV) to Joules
            Vij_values = [float(parts[i]) * eV_to_Joule for i in [5,6,7,8]]
            
            # Generate random free-energy differences
            Delta_G_list = generate_Delta_G_list()

            # Compute each of the 4 rates via Marcus theory
            k_ij_list = [
                calc_marcus_rate(Vij, dG, T, rate_factor)
                for (Vij, dG) in zip(Vij_values, Delta_G_list)
            ]

            # Build new line with the 4 rates appended
            new_line = line.strip() + "".join(f"\t{val:.6e}" for val in k_ij_list)
            output_lines.append(new_line)

    # Write to output_file
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))


# ---------------------------
# 2. Load Data
# ---------------------------
def load_data(file):
    """
    Load data lines (with the newly added k_{ij} columns) from file
    and parse them. Filter by distance < 0.7 nm, as in original code.
    Returns a list of tuples:
       (distance, rate1, x, y, z, rate2, rate3, rate4)
    """
    data = []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            distance_nm = float(parts[1])
            if distance_nm < 0.7:
                distance = distance_nm * 1e-9   # Convert nm -> m
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                # The 4 rates are in columns [8..11]
                rate1 = float(parts[8])
                rate2 = float(parts[9])
                rate3 = float(parts[10])
                rate4 = float(parts[11])
                
                data.append((distance, rate1, x, y, z, rate2, rate3, rate4))
    return data


# ---------------------------
# 3. Two Direction Functions
# ---------------------------
def random_direction_3d(distance, Nx, Ny, Nz):
    """
    Scale a unit vector (Nx, Ny, Nz) by 'distance'.
    50% chance to mirror-reflect it.
    """
    reflect = random.choice([True, False])  # 50% chance
    direction = np.array([Nx, Ny, Nz])
    if reflect:
        direction = -direction
    return distance * direction


def original_direction(distance, Nx, Ny, Nz):
    """
    'Original' direction: no random reflection, just scale (Nx, Ny, Nz).
    """
    return distance * np.array([Nx, Ny, Nz])


# ---------------------------
# 4. Simulate Hops
# ---------------------------
def simulate_hops(data, num_hops=10000, use_random_direction=False):
    """
    Simulate the hopping process and calculate the random walk
    in 3D space. Occupancy of LUMO+1 vs. LUMO is chosen by the
    global DELTA_E_EV inside occupancy_LUMO_plus_1().

    If use_random_direction=True, apply random_direction_3d().
    Otherwise, apply original_direction().
    """
    positions = [np.zeros(3)]
    times = [0]

    for _ in range(num_hops):
        # Randomly select a row
        distance, rate1, x, y, z, rate2, rate3, rate4 = random.choice(data)
        
        # Draw exponential waiting times from each rate (avoid 1/0)
        wt1 = np.random.exponential(1 / rate1) if rate1 != 0 else np.inf
        wt2 = np.random.exponential(1 / rate2) if rate2 != 0 else np.inf
        wt3 = np.random.exponential(1 / rate3) if rate3 != 0 else np.inf
        wt4 = np.random.exponential(1 / rate4) if rate4 != 0 else np.inf

        # Decide whether LUMO or LUMO+1 is occupied
        if np.random.random() < occupancy_LUMO_plus_1(T):
            waiting_time = min(wt3, wt4)
        else:
            waiting_time = min(wt1, wt2)

        # Pick direction function
        if use_random_direction:
            hop = random_direction_3d(distance, x, y, z)
        else:
            hop = original_direction(distance, x, y, z)

        # Update position, time
        positions.append(positions[-1] + hop)
        times.append(times[-1] + waiting_time)

    return np.array(positions), np.array(times)


# ---------------------------
# 5. Diffusivity & Mobility
# ---------------------------
def calculate_diffusivity(positions, times):
    """
    Calculate diffusivity from the slope of MSD vs. time.
    """
    msd = np.sum((positions - positions[0])**2, axis=1)
    slope, _, _, _, _ = linregress(times, msd)
    n_dim = 3
    diffusivity = slope / (2 * n_dim)
    # Convert from m^2/s to cm^2/s
    diffusivity *= 1e4
    return diffusivity


def calculate_mobility(diffusivity):
    """Calculate zero-field mobility in cm^2/Vs."""
    return q * diffusivity / (kB * T)


# ---------------------------
# 6. Single-run Wrapper
# ---------------------------
def single_run(use_random_direction=True, num_hops=10000):
    """
    Runs one instance of the simulation:
      - Compute k_ij for each line (using global DELTA_E_EV),
      - Load data,
      - Simulate hops,
      - Compute mobility.

    Returns the zero-field mobility (cm^2/Vs).
    """
    # Compute and write rates
    first(input_file, output_file)

    # Load data with newly computed rates
    data = load_data(output_file)

    # Run the hopping simulation
    positions, times = simulate_hops(data, num_hops=num_hops, 
                                     use_random_direction=False)

    # Compute diffusivity & mobility
    D = calculate_diffusivity(positions, times)
    mu_0 = calculate_mobility(D)
    return mu_0


# ---------------------------
# 7. Main
# ---------------------------
if __name__ == "__main__":
    num_runs = 100
    num_hops = 100000
    use_random_reflection = True  # or False

    mobilities = []
    for _ in range(num_runs):
        mu_0 = single_run(use_random_direction=use_random_reflection,
                          num_hops=num_hops)
        mobilities.append(mu_0)

    mobilities = np.array(mobilities)

    # Statistics
    mean_mobility = np.mean(mobilities)
    median_mobility = np.median(mobilities)

    counts, bin_edges = np.histogram(mobilities, bins=15, density=False)
    mode_bin_idx = np.argmax(counts)
    mode_mobility = 0.5 * (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx+1])

    log_mobilities = np.log(mobilities)
    geometric_mean = np.exp(np.mean(log_mobilities))

    # Print results
    print(f"Number of runs:         {num_runs}")
    print(f"Mean μ_0        = {mean_mobility:.6e} cm^2/Vs")
    print(f"Median μ_0      = {median_mobility:.6e} cm^2/Vs")
    print(f"Mode μ_0 (hist) = {mode_mobility:.6e} cm^2/Vs")
    print(f"Geometric Mean μ_0 = {geometric_mean:.6e} cm^2/Vs")

    # Plot histogram
    plt.figure(figsize=(7,5))
    n, bins, patches = plt.hist(mobilities, bins=15, density=True, alpha=0.6, color='g')
    plt.axvline(mean_mobility,   color='r', linestyle='--', 
                label=f"Mean = {mean_mobility:.2e} cm^2/Vs")
    plt.axvline(median_mobility, color='b', linestyle='--',
                label=f"Median = {median_mobility:.2e} cm^2/Vs")
    plt.axvline(mode_mobility,   color='k', linestyle='--',
                label=f"Mode = {mode_mobility:.2e} cm^2/Vs")

    plt.xlabel('Zero-field Mobility (cm^2/Vs)')
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of Zero-field Mobility\n(ΔE={DELTA_E_EV} eV, random={use_random_reflection})')
    plt.legend()
    plt.show()

    # Save mobilities
    np.savetxt("test.csv", mobilities, delimiter=",", 
               header="Mobility (cm^2/Vs)", comments="")
```


# plot the probability density data
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = "3electron-mobilities.csv"
df = pd.read_csv(file_path)

# Extract mobility data
mobility_data = df.iloc[:, 0]  # Assuming the first column contains the mobility values

# Define bin edges using logarithmic scale
bins = np.logspace(np.log10(mobility_data.min()), np.log10(mobility_data.max()), 30)

# Plot the histogram with log-scaled bins
plt.figure(figsize=(10, 6))
plt.hist(mobility_data, bins=bins, density=True, edgecolor='black', alpha=0.7)

# Improve numerical labels on x and y axes
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
# Formatting the graph
plt.xscale("log")
plt.xlabel("Mobility (cm²/Vs)", fontsize=14, fontweight='bold')
plt.ylabel("Probability Density", fontsize=14, fontweight='bold')
plt.title("", fontsize=16, fontweight='bold')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.gca().spines["top"].set_linewidth(2)
plt.gca().spines["right"].set_linewidth(2)
plt.gca().spines["bottom"].set_linewidth(2)
plt.gca().spines["left"].set_linewidth(2)

# Show the plot
plt.show()
```

sometimes we want to quantify the gaussian disorder. This can be done by calculating all pdb molecular structure extracted from MD and run the terachem through the script after list all the pdb files

```txt

# Define TeraChem variables
export TeraChem=/home/user/woon/terachem-1.96p
export PATH=$TeraChem/bin:$PATH
export LD_LIBRARY_PATH=$TeraChem/lib:$LD_LIBRARY_PATH
export NBOEXE=$TeraChem/bin/nbo6.i4.exe

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  tc_homo_lumo_batch.sh
#
#  • reads pdb_files.txt line-by-line
#  • builds 1.ts, runs TeraChem synchronously
#  • extracts HOMO & LUMO (eV) from the output *.molden file
#  • appends   <pdb>  <HOMO_eV>  <LUMO_eV>   to $WORKDIR/homo_lumo.txt
#  • deletes   scr.<N>/   after extraction to free disk space
# ---------------------------------------------------------------------------

set -euo pipefail

PDB_LIST="pdb_files.txt"               # list of .pdb files (one per line)
WORKDIR="$(pwd)"                       # absolute path for safety
HL_TABLE="${WORKDIR}/homo_lumo.txt"    # consolidated output
HARTREE2EV=27.211386245988             # CODATA-2022

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$PDB_LIST" ]] || die "$PDB_LIST not found."

# fresh results file with header row
echo -e "#pdb\tHOMO_eV\tLUMO_eV" > "$HL_TABLE"

# ───────────────────────── Homo/Lumo extractor ────────────────────────────
extract_homo_lumo () (
    # Args: 1 = molden file, 2 = label to print (pdb name)
    molden="$1"
    label="$2"

    awk -v h2e="$HARTREE2EV" -v lab="$label" '
        /Ene[[:space:]]*=/  { sub(/.*Ene[[:space:]]*=[[:space:]]*/, "", $0);  E[n]=$1; next }
        /Occup[[:space:]]*=/ { sub(/.*Occup[[:space:]]*=[[:space:]]*/, "", $0); O[n]=$1; n++ }
        END {
            if (n==0) { print "No orbitals found in " lab > "/dev/stderr"; exit 2 }
            homo=-1e9; lumo=1e9
            for (i=0;i<n;i++) {
                if (O[i]>0 && E[i]>homo) homo=E[i]
                if (O[i]==0 && E[i]<lumo) lumo=E[i]
            }
            printf("%s\t%.6f\t%.6f\n", lab, homo*h2e, lumo*h2e)
        }' "$molden" >> "$HL_TABLE" \
        && echo "  → appended HOMO/LUMO to homo_lumo.txt"
)

# ─────────────────────────── Main processing loop ─────────────────────────
while IFS= read -r pdb || [[ -n "$pdb" ]]; do
    # skip blanks or comment lines
    [[ -z "$pdb" || "$pdb" =~ ^[[:space:]]*# ]] && continue

    echo "=== Processing $pdb ==="

    # 1) build 1.ts
    cat > 1.ts <<EOF
basis          def2-svp
coordinates    $pdb
charge         0
spinmult       1
method         wpbeh
rc_w           0.04975834695659544 #CHANGE THIS
pcm            cosmo
epsilon        2.38
pcm_scale      1
maxit          500
run            energy
end
EOF

    # 2) run TeraChem (blocking)
    if ! terachem "${WORKDIR}/1.ts"; then
        echo " TeraChem failed for $pdb — skipping extraction." >&2
        continue
    fi

    # 3) locate Molden file in scr.<N>/  (N = first integer in pdb name)
    num=$(echo "$pdb" | grep -oE '[0-9]+' | head -n1)
    scrdir="scr.${num}"

    molden=$(find "$scrdir" -maxdepth 1 -type f -name '*.molden' -print -quit 2>/dev/null || true)
    if [[ -z "$molden" ]]; then
        echo " No .molden file found in $scrdir — skipping." >&2
        continue
    fi

    # 4) extract HOMO/LUMO and append to table
    extract_homo_lumo "$molden" "$pdb"

    # 5) remove the scr.<N> directory (including the Molden file)
    rm -rf "$scrdir"
    echo "  → removed directory $scrdir"
done < "$PDB_LIST"

echo "All jobs complete.  Consolidated results in $HL_TABLE"
```

calculation of electronic coupling can be found at (here)[https://github.com/ph7klw76/gaussian_note/blob/main/electronic_coupling.md]
