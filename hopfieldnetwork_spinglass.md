*By Kai Lin Woon*
# Introduction


Disordered spin glasses are complex systems in statistical physics characterized by randomness and frustration in their interactions. Understanding these systems is crucial for fields ranging from condensed matter physics to optimization problems in computer science. One powerful approach to study spin glasses is through neural networks, specifically Hopfield networks.

In this blog post, we will delve into the mathematical foundations required to understand how Hopfield networks can be used to minimize the energy of a disordered spin glass system. We will explore the connections between spin glasses and neural networks, derive the necessary equations, and provide a Python implementation that visualizes the energy minimization process.

## Background on Spin Glasses

### What is a Spin Glass?

A spin glass is a type of disordered magnetic system where the magnetic moments (spins) are randomly aligned due to irregular interactions between them. Unlike conventional ferromagnets or antiferromagnets, spin glasses exhibit randomness in both the magnitude and sign of their coupling constants. This randomness leads to **frustration**, where it's impossible to find a spin configuration that minimizes the energy of all interactions simultaneously.

### Key Characteristics
- **Disorder**: Random distribution of coupling constants between spins.
- **Frustration**: Competing interactions that prevent simultaneous minimization.
- **Complex Energy Landscape**: Multiple local minima with nearly identical energies.

### The Ising Model

The Ising model provides a mathematical framework to study spin systems:

$$
E = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j
$$

where:  
- $E$ is the total energy.  
- $s_i$ and $s_j$ are spins that can take values $\pm 1$.  
- $J_{ij}$ are the coupling constants between spins.  
- $\langle i,j \rangle$ denotes that the sum is over neighboring spins.  

In spin glasses, $J_{ij}$ are random variables, introducing disorder into the system.

## Hopfield Networks Overview

A Hopfield network is a form of recurrent artificial neural network that serves as a content-addressable memory system with binary threshold nodes. It's particularly useful for solving optimization problems and has direct connections to the Ising model.

### Network Structure
- **Nodes**: Represent spins ($s_i$).  
- **Weights**: Correspond to coupling constants ($J_{ij}$).  
- **State Update**: Based on the local field and an activation function.

### Energy Function

The energy function of a Hopfield network is analogous to that of a spin system:

$$
E = -\frac{1}{2} \sum_{i \neq j} J_{ij} s_i s_j + \sum_i \theta_i s_i
$$

where $\theta_i$ are external thresholds (we can set them to zero for simplicity). The network dynamics aim to minimize this energy function.

## Mathematical Foundations

### Energy Function of Spin Glasses

The total energy of a spin glass system is given by:

$$
E = -\sum_{\langle i,j \rangle} J_{ij} s_i s_j
$$

For a 2D grid, spins are arranged in a lattice, and interactions occur between nearest neighbors.

### Dynamics of Hopfield Networks

In a Hopfield network, the state of each neuron (spin) is updated asynchronously based on the local field:

$$
h_i = \sum_j J_{ij} s_j
$$

The neuron updates its state according to:

$$
s_i^{\text{new}} = \text{sgn}(h_i)
$$

The sign function $\text{sgn}(x)$ is defined as:

$$
\text{sgn}(x) = 
\begin{cases} 
1, & \text{if } x \geq 0 \\
-1, & \text{if } x < 0 
\end{cases}
$$

### Mapping Spin Glasses to Hopfield Networks

By identifying spins with neurons and coupling constants with weights, we can use the Hopfield network dynamics to minimize the energy of a spin glass system.

### Derivation of Update Rules

#### Local Field Calculation

For a spin at position $(i,j)$ in a 2D grid, the local field $h_{i,j}$ is influenced by its neighboring spins:

$$
h_{i,j} = \sum_{(k,l) \in \text{neighbors}} J_{(i,j),(k,l)} s_{k,l}
$$

Neighbors: In a 2D grid, each spin has up to four neighbors (north, south, east, west).

#### Update Rule Derivation

We want to update $s_{i,j}$ to minimize the energy. Consider flipping $s_{i,j}$:

**Energy Change**:

$$
\Delta E = E_{\text{new}} - E_{\text{old}} = -2 s_{i,j} h_{i,j}
$$

**Decision Rule**:  
If $\Delta E < 0$, flipping $s_{i,j}$ decreases the energy.

Therefore, we update $s_{i,j}$ based on the sign of $h_{i,j}$:

$$
s_{i,j}^{\text{new}} = \text{sgn}(h_{i,j})
$$

### Convergence Guarantee

The asynchronous update rule ensures that the energy decreases or remains the same at each step. Since the energy is bounded below, the system will eventually converge to a stable state (local minimum).

## Python Implementation

Let's implement the described model in Python, using NumPy and Matplotlib for computations and visualization.
## Code Explanation

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SpinGlass2D:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
        # Initialize spins randomly to -1 or 1
        self.spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
        
        # Random interaction weights between neighboring spins
        self.weights = {
            'horizontal': np.random.randn(grid_size, grid_size - 1),
            'vertical': np.random.randn(grid_size - 1, grid_size)
        }

    def energy(self):
        E = 0
        # Horizontal interactions
        E -= np.sum(self.weights['horizontal'] * self.spins[:, :-1] * self.spins[:, 1:])
        # Vertical interactions
        E -= np.sum(self.weights['vertical'] * self.spins[:-1, :] * self.spins[1:, :])
        return E

    def sign(self, x):
        return 1 if x >= 0 else -1

    def update_spin(self):
        i, j = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
        local_field = 0
        # Left neighbor
        if j > 0:
            local_field += self.weights['horizontal'][i, j - 1] * self.spins[i, j - 1]
        # Right neighbor
        if j < self.grid_size - 1:
            local_field += self.weights['horizontal'][i, j] * self.spins[i, j + 1]
        # Top neighbor
        if i > 0:
            local_field += self.weights['vertical'][i - 1, j] * self.spins[i - 1, j]
        # Bottom neighbor
        if i < self.grid_size - 1:
            local_field += self.weights['vertical'][i, j] * self.spins[i + 1, j]
        # Update spin based on the local field
        self.spins[i, j] = self.sign(local_field)

    def find_minimum_energy(self, max_steps=1000):
        min_energy = self.energy()
        min_spins = self.spins.copy()
        energies = [min_energy]
        spins_list = [self.spins.copy()]
        steps = [0]

        for step in range(1, max_steps + 1):
            self.update_spin()
            current_energy = self.energy()
            energies.append(current_energy)
            spins_list.append(self.spins.copy())
            steps.append(step)
            if current_energy < min_energy:
                min_energy = current_energy
                min_spins = self.spins.copy()
            if step % 100 == 0:
                print(f"Step {step}, Current Energy: {current_energy}, Minimum Energy so far: {min_energy}")
        return min_spins, min_energy, energies, spins_list, steps

# Initialize the system
grid_size = 10
spin_glass_2d = SpinGlass2D(grid_size=grid_size)

# Find the minimum energy configuration
max_steps = 1000
min_spins, min_energy, energies, spins_list, steps = spin_glass_2d.find_minimum_energy(max_steps=max_steps)
'''

# Initialize the system
grid_size = 10
spin_glass_2d = SpinGlass2D(grid_size=grid_size)

# Find the minimum energy configuration
max_steps = 1000
min_spins, min_energy, energies, spins_list, steps = spin_glass_2d.find_minimum_energy(max_steps=max_steps)
```
### Key Components
- **Initialization**: Random spins and weights represent the disordered system.
- **Energy Calculation**: Computes the total energy based on current spins and weights.
- **Update Rule**: Updates a randomly selected spin using the sign of the local field.
- **Energy Minimization Loop**: Repeats the update process and tracks energy over time.

![fig2](https://github.com/user-attachments/assets/b301844d-1b2d-4050-92fc-0f66ac538c96)


## Conclusion
By leveraging the dynamics of Hopfield networks, we can effectively minimize the energy of disordered spin glass systems. The mathematical derivations show that the asynchronous update rule based on the local field leads the system towards lower energy states.


