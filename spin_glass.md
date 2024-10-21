# Simulating Energy Minimization in Disordered Spin Glasses with Python  
*By Kai Lin Woon*

## Introduction  
Spin glasses are fascinating systems in condensed matter physics characterized by disordered magnetic interactions. Unlike conventional ferromagnetic or antiferromagnetic materials, spin glasses exhibit a complex energy landscape due to random interactions between spins. This blog post delves into the mathematical framework of disordered spin glasses, explores energy minimization techniques, and walks through a Python simulation using the Metropolis algorithm.

<!-- Replace with actual image URL -->

## Background on Spin Glasses  
Spin glasses are materials where the magnetic moments (spins) are randomly aligned due to disorder in the system. This randomness leads to frustration, where not all magnetic interactions can be simultaneously minimized, resulting in a highly degenerate ground state and complex dynamics.

### Key Characteristics
- **Disorder**: Random distribution of ferromagnetic and antiferromagnetic interactions.
- **Frustration**: Inability to find a spin configuration that minimizes all interaction energies.
- **Slow Dynamics**: Due to the rugged energy landscape, the system exhibits slow relaxation times.

## Mathematical Model

### The Ising Model  
The Ising model is a mathematical model of ferromagnetism in statistical mechanics. It consists of discrete variables called spins, $s_i$, which can take values of $+1$ or $-1$. The energy of a spin configuration is given by:

$$
E = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j
$$

where:  
- $\langle i,j \rangle$ denotes summation over nearest neighbors.  
- $J_{ij}$ is the interaction strength between spins $s_i$ and $s_j$.  

### Spin Glass Extension  
In a spin glass, the coupling constants $J_{ij}$ are random variables, typically taking values of $+1$ or $-1$, representing ferromagnetic or antiferromagnetic interactions, respectively. This randomness introduces disorder into the system.

## Energy Minimization  
The goal is to find spin configurations that minimize the total energy of the system. However, due to the disordered nature and frustration in spin glasses, traditional optimization methods are ineffective. Instead, we use stochastic methods like the Metropolis algorithm to simulate the dynamics and approach minimal energy states.

## The Metropolis Algorithm  
The Metropolis algorithm is a Monte Carlo method used to obtain a sequence of samples from a probability distribution. It allows us to simulate the thermal fluctuations of the spin system at a given temperature $T$.

### Algorithm Steps  
1. **Initialization**: Start with a random spin configuration.  
2. **Spin Selection**: Randomly select a spin $s_i$ to flip.  
3. **Energy Calculation**: Compute the change in energy $\Delta E$ resulting from flipping $s_i$.  
4. **Acceptance Criterion**:  
    - If $\Delta E \leq 0$, accept the flip.  
    - Else, accept the flip with probability $\exp(-\Delta E / k_B T)$.  
5. **Iteration**: Repeat steps 2-4 for a large number of iterations.

## Python Implementation

Below is the Python code implementing the Metropolis algorithm for simulating a 2D disordered spin glass:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
N = 25  # Lattice size (NxN)
num_iterations = 15000  # Total number of iterations
T = 1.0  # Temperature
record_interval = 100  # Interval to record data for animation

# Initialize spins (+1 or -1) randomly
spins = np.random.choice([-1, 1], size=(N, N))

# Initialize random couplings J_ij (+1 or -1) between neighboring spins
# Horizontal couplings (J_h): interactions in the x-direction
J_h = np.random.choice([-1, 1], size=(N, N - 1))
# Vertical couplings (J_v): interactions in the y-direction
J_v = np.random.choice([-1, 1], size=(N - 1, N))
```
## Explanation
Spin Initialization: Spins are randomly assigned values of $+1$ or $-1$.
Coupling Constants: Random couplings $J_h$ and $J_v$ represent the disordered interactions between neighboring spins.
## Energy Computation
The total energy is calculated using:
```python
def compute_total_energy(spins, J_h, J_v):
    """Compute the total energy of the current spin configuration."""
    energy = 0
    # Horizontal interactions
    energy += -np.sum(J_h * spins[:, :-1] * spins[:, 1:])
    # Vertical interactions
    energy += -np.sum(J_v * spins[:-1, :] * spins[1:, :])
    return energy
```
## Mathematical Representation
The total energy is given by:

$$
E = - \sum_{i=1}^{N} \sum_{j=1}^{N} \left( J_{i,j}^{(h)} s_{i,j} s_{i,j+1} + J_{i,j}^{(v)} s_{i,j} s_{i+1,j} \right)
$$

where:

$J_{i,j}^{(h)}$ and $J_{i,j}^{(v)}$ are the horizontal and vertical coupling constants, respectively.

## Energy Change upon Spin Flip
```python
def delta_energy(spins, J_h, J_v, i, j):
    """Compute the change in energy if spin at (i, j) is flipped."""
    N = spins.shape[0]
    s = spins[i, j]
    dE = 0
    # Interaction with left neighbor
    if j > 0:
        dE += 2 * s * J_h[i, j - 1] * spins[i, j - 1]
    # Interaction with right neighbor
    if j < N - 1:
        dE += 2 * s * J_h[i, j] * spins[i, j + 1]
    # Interaction with top neighbor
    if i > 0:
        dE += 2 * s * J_v[i - 1, j] * spins[i - 1, j]
    # Interaction with bottom neighbor
    if i < N - 1:
        dE += 2 * s * J_v[i, j] * spins[i + 1, j]
    return dE
```
## Mathematical Derivation
When a spin $s_{i,j}$ is flipped, the change in energy $\Delta E$ is:

$$
\Delta E = 2 s_{i,j} \left( J_{i,j}^{(h)} s_{i,j+1} + J_{i,j-1}^{(h)} s_{i,j-1} + J_{i,j}^{(v)} s_{i+1,j} + J_{i-1,j}^{(v)} s_{i-1,j} \right)
$$

This accounts for the interactions with its nearest neighbors.
## Metropolis Step
```python
def metropolis_step(spins, J_h, J_v, T):
    """Perform one Metropolis update step."""
    N = spins.shape[0]
    # Randomly select a spin
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    dE = delta_energy(spins, J_h, J_v, i, j)
    if dE <= 0 or np.random.rand() < np.exp(-dE / T):
        # Accept the flip
        spins[i, j] *= -1
    return spins
```
## Simulation Loop
The simulation runs for a specified number of iterations, recording the energy and spin configurations at intervals.
```python
# Lists to store data for animation
energies = []
spins_list = []
iterations = []

# Initial total energy
initial_energy = compute_total_energy(spins, J_h, J_v)
energies.append(initial_energy)
spins_list.append(spins.copy())
iterations.append(0)

print(f"Initial Energy: {initial_energy}")

# Simulation loop
for n in range(1, num_iterations + 1):
    spins = metropolis_step(spins, J_h, J_v, T)
    # Record data at specified intervals
    if n % record_interval == 0:
        energy = compute_total_energy(spins, J_h, J_v)
        energies.append(energy)
        spins_list.append(spins.copy())
        iterations.append(n)
        print(f"Iteration {n}, Energy {energy}")
```
## Visualization
The evolution of the spin configurations and energy over iterations is visualized using Matplotlib's animation functionality.
```python
# Prepare for animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Set up the heatmap for spin configurations
im = ax1.imshow(spins_list[0], cmap='coolwarm', vmin=-1, vmax=1)
ax1.set_title('Spin Configuration')

# Set up the energy plot
ax2.set_title('Energy vs Iteration')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total Energy')
ax2.grid(True)
line, = ax2.plot([], [], lw=2)

# Initialize the energy plot limits
ax2.set_xlim(0, num_iterations)
ax2.set_ylim(min(energies), max(energies))

def animate(i):
    """Update function for animation."""
    im.set_data(spins_list[i])
    line.set_data(iterations[:i + 1], energies[:i + 1])
    return im, line

# Create the animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(spins_list), interval=200, blit=True, repeat_delay=1000)

plt.tight_layout()
plt.show()
```
## Mathematics Behind the Code

### Random Spin Initialization
- **Purpose**: To start the simulation without any bias.  
- **Mathematics**: Random selection from $\{ -1, +1 \}$ for each spin site.

### Random Coupling Constants
- **Purpose**: Introduce disorder into the system.  
- **Mathematics**: Assign $J_{ij} = \pm 1$ randomly for each pair of neighboring spins.

### Energy Calculation
- **Total Energy**: Sum over all interactions between neighboring spins.  
- **Change in Energy** ($\Delta E$): Calculated for a potential spin flip, considering interactions with nearest neighbors.

### Metropolis Criterion

**Acceptance Probability**:

$$
P =
\begin{cases} 
1, & \text{if } \Delta E \leq 0 \\
\exp\left(-\frac{\Delta E}{k_B T}\right), & \text{if } \Delta E > 0 
\end{cases}
$$

where $k_B$ is the Boltzmann constant (set to 1 in our code for simplicity).

### Temperature Dependence
At higher temperatures, the system is more likely to accept higher energy states, allowing it to escape local minima.

## Visualization of Spin Configuration and Energy vs Iteration
The figure above consists of two subplots that illustrate key aspects of the spin glass simulation:

Left Plot (Spin Configuration): This is a visual representation of the spin configuration on a 2D lattice after a number of iterations. Each pixel in the grid represents a spin, where red corresponds to a spin value of $+1$, and blue corresponds to a spin value of $-1$. As the simulation progresses, the spins evolve, and patterns emerge due to the interaction dynamics of the spin glass model. The randomness and disorder inherent in spin glasses are reflected in the irregular patterns formed by the spins.

Right Plot (Energy vs Iteration): This graph shows the total energy of the system plotted against the number of iterations. Initially, the energy decreases rapidly, indicating that the system is quickly finding lower energy configurations. As the iterations continue, the rate of energy decrease slows down, and the system approaches a more stable, minimal energy state. The small fluctuations toward the end of the simulation reflect the stochastic nature of the Metropolis algorithm and thermal fluctuations.


