# Minimizing Energy in Disordered Spin Glasses using Boltzmann Machines and the Metropolis Algorithm  
*By Kai Lin Woon *

## Introduction  
Disordered spin glasses are fascinating and complex systems in statistical physics that exhibit rich behavior due to the presence of randomness and frustration in their interactions. Understanding these systems is crucial not only in physics but also in fields like neural networks, optimization, and computational neuroscience.

The Sherrington–Kirkpatrick (SK) model, introduced by David Sherrington and Scott Kirkpatrick in 1975, is a foundational model that captures the essence of spin glasses by considering infinite-range interactions with random coupling strengths. In this blog post, we will delve into the mathematical underpinnings of using a Boltzmann machine based on the SK model to minimize the energy of a disordered spin glass system. We will rigorously derive the Metropolis update algorithm used in our simulations.

## 1. Background on Spin Glasses and the SK Model

### What is a Spin Glass?
A spin glass is a type of disordered magnetic system where the spins (magnetic moments of atoms) are randomly aligned due to competing interactions. Unlike conventional magnets, where spins align uniformly (ferromagnetism) or in an alternating pattern (antiferromagnetism), spin glasses have interactions that are both ferromagnetic and antiferromagnetic, distributed randomly throughout the material.

### Key Characteristics:
- **Disorder**: Random distribution of interactions between spins.
- **Frustration**: Inability to satisfy all interaction constraints simultaneously.
- **Slow Dynamics**: The system takes a long time to reach equilibrium due to the complex energy landscape.

### Frustration in Spin Glasses
Frustration arises when the interactions among spins cannot be simultaneously satisfied. Consider a simple example of three spins in a triangle with the following interactions:
- Spin 1 and Spin 2 prefer to align (ferromagnetic interaction).
- Spin 2 and Spin 3 prefer to align.
- Spin 1 and Spin 3 prefer to anti-align (antiferromagnetic interaction).

No arrangement of spins can satisfy all three interactions:
- If Spins 1 and 2 are both up, Spin 3 must be up to satisfy the interaction with Spin 2, but then the antiferromagnetic interaction with Spin 1 is frustrated.

#### Implications:
- Multiple nearly degenerate ground states.
- Highly non-trivial thermodynamic and dynamic properties.

### The Sherrington–Kirkpatrick Model
The Sherrington–Kirkpatrick (SK) model is a mean-field model that extends the Ising model to include infinite-range interactions with random coupling strengths.

#### Hamiltonian of the SK Model
The Hamiltonian (energy function) for the SK model is given by:

$$
E = - \sum_{i < j} J_{ij} s_i s_j
$$

Where:
- $E$: Total energy of the system.
- $s_i = \pm 1$: Spin at site $i$.
- $J_{ij}$: Random coupling constant between spins $i$ and $j$.
- $N$: Total number of spins.

#### Distribution of Coupling Constants
The coupling constants $J_{ij}$ are independent random variables drawn from a Gaussian distribution with zero mean and variance $\frac{1}{N}$:

$$
P(J_{ij}) = \frac{N}{\sqrt{2\pi J^2}} \exp \left( -\frac{N J_{ij}^2}{2 J^2} \right)
$$

Where:
- $J^2$: Variance parameter (often set to 1 for simplicity).

### Properties of the SK Model:
- **Infinite-Range Interactions**: Each spin interacts with every other spin in the system.
- **Mean-Field Approximation**: Due to infinite-range interactions, the model effectively averages over the influence of all other spins.
- **Complex Energy Landscape**: The random couplings lead to a rugged energy landscape with numerous local minima.

---

## 2. Boltzmann Machines

### Overview of Boltzmann Machines
A Boltzmann machine is a type of stochastic recurrent neural network that can model probability distributions over binary variables. It consists of a network of units (neurons) with symmetric connections and operates using stochastic dynamics inspired by statistical mechanics.

#### Components:
- **Units/Neurons**: Binary variables $s_i = \pm 1$ representing the state of each neuron.
- **Weights $W_{ij}$**: Symmetric connection strengths between units.
- **Biases $b_i$**: External bias applied to each unit (often set to zero for simplicity).

#### Operation:
The network evolves by probabilistically updating the state of each neuron based on the states of its neighbors and the connection weights.

### Energy Function in Boltzmann Machines
The energy of a state $s$ in a Boltzmann machine is defined analogously to the Hamiltonian in statistical physics:

$$
E(s) = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j - \sum_i b_i s_i
$$

### Probability Distribution:
The probability of the network being in a particular state $s$ is given by the Boltzmann distribution:

$$
P(s) = \frac{1}{Z} \exp\left( -\frac{E(s)}{k_B T} \right)
$$

Where:
- $Z$: Partition function, ensuring the probabilities sum to 1.
- $k_B$: Boltzmann constant (often set to 1 in simulations).
- $T$: Temperature parameter controlling the level of stochasticity.

### Relation to Spin Glasses
Boltzmann machines are closely related to spin glass models:
- **Spins ↔ Neurons**: The binary spins $s_i$ correspond to the state of neurons in the network.
- **Coupling Constants ↔ Weights**: The random couplings $J_{ij}$ in spin glasses correspond to the weights $W_{ij}$ in the Boltzmann machine.
- **Energy Landscape**: Both systems aim to find configurations that minimize the energy function.

### Advantages of Using Boltzmann Machines:
- Ability to model complex probability distributions.
- Use of stochastic dynamics allows exploration of the energy landscape.
- Suitable for simulating systems with frustration and disorder.

---

## 3. Metropolis Algorithm

### Monte Carlo Methods in Statistical Physics
Monte Carlo methods are computational algorithms that rely on random sampling to obtain numerical results. They are particularly useful in statistical physics for studying systems with a large number of interacting components.

#### Applications:
- Calculating thermodynamic quantities (e.g., average energy, magnetization).
- Simulating time evolution of systems at thermal equilibrium.
- Exploring phase transitions and critical phenomena.

### Detailed Balance and Ergodicity
To ensure that a Monte Carlo simulation converges to the correct equilibrium distribution, two key properties must be satisfied:

- **Detailed Balance**: The probability of transitioning from state $A$ to $B$ times the probability of being in state $A$ must equal the probability of transitioning from $B$ to $A$ times the probability of being in state $B$:

$$
P(A) w(A \to B) = P(B) w(B \to A)
$$

Where $w(A \to B)$ is the transition probability from state $A$ to $B$.

- **Ergodicity**: It must be possible to reach any state from any other state through a series of allowed transitions. This ensures that the simulation explores the entire configuration space.

### Steps of the Metropolis Algorithm
The Metropolis algorithm is a Markov Chain Monte Carlo (MCMC) method that generates a sequence of states according to the Boltzmann distribution.

#### Algorithm Steps:
1. **Initialization**: Start with an arbitrary initial configuration $s$.
2. **Spin Selection**: Randomly select a spin $s_k$ to consider flipping.
3. **Compute Energy Change**: Calculate the change in energy $\Delta E$ resulting from flipping $s_k$:

$$
\Delta E = E_{\text{new}} - E_{\text{old}} = 2 s_k h_k
$$

   Where $h_k = \sum_j J_{kj} s_j$ is the local field acting on spin $s_k$.
   
4. **Acceptance Criterion**:  
   - If $\Delta E \leq 0$, accept the flip (the new state is energetically favorable).
   - If $\Delta E > 0$, accept the flip with probability:

$$
P_{\text{accept}} = \exp\left( -\frac{\Delta E}{k_B T} \right)
$$

5. **Update Configuration**: If the flip is accepted, update $s_k \to -s_k$.
6. **Iteration**: Repeat steps 2–5 for a large number of iterations to ensure convergence to equilibrium.

### Properties:
- **Satisfies Detailed Balance**: Ensures convergence to the correct equilibrium distribution.
- **Ergodic**: Random spin selection allows exploration of the entire state space.
- **Flexible**: Applicable to a wide range of systems and Hamiltonians.

---

## 4. Mathematical Derivation

### Energy Function
In the SK model, the energy (Hamiltonian) is given by:

$$
E = -\sum_{i < j} J_{ij} s_i s_j
$$

For computational purposes, it's convenient to represent the Hamiltonian in matrix form:

$$
E = -\frac{1}{2} \sum_{i,j} J_{ij} s_i s_j
$$

The factor $\frac{1}{2}$ corrects for double-counting since $J_{ij} = J_{ji}$.

Defining the spin vector $s$ and the interaction matrix $J$, we can write:

$$
E = -\frac{1}{2} s^T J s
$$

### Change in Energy
When flipping a single spin $s_k$, the change in energy $\Delta E$ can be derived as follows.

#### Initial Energy:

$$
E_{\text{old}} = -\frac{1}{2} \sum_{i,j} J_{ij} s_i s_j
$$

#### Energy After Flipping $s_k$:

The new spin configuration $s'$ is the same as $s$ except for $s_k' = -s_k$.  
Thus, the new energy is:

$$
E_{\text{new}} = -\frac{1}{2} \sum_{i,j} J_{ij} s_i' s_j'
$$

Since only $s_k$ changes sign:

$$
s_i' = 
\begin{cases} 
-s_k, & \text{if } i = k \\
s_i, & \text{if } i \neq k
\end{cases}
$$

#### Calculating $\Delta E$:

The difference in energy is due to the terms involving $s_k$:

$$
\Delta E = E_{\text{new}} - E_{\text{old}} = -\frac{1}{2} \left( \sum_i J_{ki} s_k' s_i' + \sum_j J_{jk} s_j' s_k' \right) + \frac{1}{2} \left( \sum_i J_{ki} s_k s_i + \sum_j J_{jk} s_j s_k \right)
$$

Simplifying using $J_{ki} = J_{ik}$ and $s_i' = s_i$ for $i \neq k$:

$$
\Delta E = - \sum_i J_{ki} (-s_k) s_i + \sum_i J_{ki} s_k s_i = 2 s_k \sum_i J_{ki} s_i
$$

Thus:

$$
\Delta E = 2 s_k h_k
$$

Where $h_k = \sum_i J_{ki} s_i$ is the local field acting on spin $s_k$.


## Metropolis Acceptance Criterion

The probability of accepting the flip of spin $s_k$ is determined by the Metropolis criterion:

$$
P_{\text{accept}} =
\begin{cases} 
1, & \text{if } \Delta E \leq 0 \\
\exp\left( -\frac{\Delta E}{k_B T} \right), & \text{if } \Delta E > 0
\end{cases}
$$

### Temperature Dependence:
- At low temperatures ($T \to 0$), flips that increase the energy are rarely accepted.
- At high temperatures, the system explores a wider range of configurations, including higher energy states.

---

## Implementation Considerations

### Random Number Generation:
- Use high-quality pseudo-random number generators to ensure proper stochastic behavior.
- The acceptance probability is compared against a uniformly distributed random number in $[0, 1)$.

### Interaction Matrix Storage:
- For large $N$, storing the full $N \times N$ matrix $J$ can be memory-intensive.
- Since $J$ is symmetric with zero diagonal elements ($J_{ii} = 0$), storage can be optimized.

### Energy Calculation Efficiency:
- Instead of computing the total energy after each flip, update the energy incrementally using $\Delta E$. This reduces computational overhead.

### Equilibration and Measurement:
- Allow the system to reach equilibrium before collecting data (e.g., discard initial steps).
- Average quantities over multiple runs or over time to improve statistical accuracy.

### Convergence Criteria:
- Monitor physical quantities (e.g., energy, magnetization) to determine when the system has reached equilibrium.
- Alternatively, run the simulation for a sufficiently large number of steps.

## 5. Python Implementation

### Code Explanation

#### Initialization
- **Spins**: Initialized randomly to $\pm 1$.
- **Weights**: The coupling constants $J_{ij}$ are drawn from a normal distribution with zero mean and variance $\frac{1}{N}$.

```python
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class BoltzmannMachine2D:
    def __init__(self, grid_size, temperature):
        """
        Initialize the 2D Boltzmann Machine based on the Sherrington-Kirkpatrick model.
        :param grid_size: The size of the 2D grid (grid_size x grid_size).
        :param temperature: The temperature controlling the randomness of the updates.
        """
        self.grid_size = grid_size
        self.temperature = temperature
        
        # Initialize the spins randomly to -1 or 1 (2D grid of spins)
        self.spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
        
        # Initialize the interaction weights (J_ij) between all pairs of spins randomly
        # SK model: interactions between all pairs with weights drawn from a normal distribution
        N = grid_size * grid_size
        self.weights = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
        np.fill_diagonal(self.weights, 0)  # No self-interaction

    def energy(self):
        """
        Compute the energy of the 2D spin glass system.
        The energy is computed as the sum of interactions between all pairs of spins.
        :return: The total energy of the system.
        """
        spins_flat = self.spins.flatten()
        E = -0.5 * np.dot(spins_flat, np.dot(self.weights, spins_flat))
        return E

    def metropolis_update(self):
        """
        Perform a Metropolis update to simulate the Boltzmann machine's behavior.
        Randomly pick a spin and flip it with a probability based on the change in energy and temperature.
        :return: None
        """
        # Randomly select a spin index
        i = np.random.randint(0, self.grid_size)
        j = np.random.randint(0, self.grid_size)
        idx = i * self.grid_size + j  # Flattened index

        # Calculate the change in energy if spin (i, j) flips
        spins_flat = self.spins.flatten()
        s = spins_flat[idx]
        delta_E = 2 * s * np.dot(self.weights[idx], spins_flat)

        # Accept the flip based on the Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / self.temperature):
            self.spins[i, j] *= -1

    def find_minimum_energy(self, max_steps=10000):
        """
        Simulate the Boltzmann machine to find the minimum energy configuration.
        :param max_steps: The maximum number of Metropolis updates to perform.
        :return: The final spin configuration and the corresponding minimum energy.
        """
        min_energy = self.energy()
        min_spins = self.spins.copy()

        # Lists to store energies and spin configurations for animation
        energies = []
        spins_list = []
        steps = []

        # Store initial state
        energies.append(min_energy)
        spins_list.append(self.spins.copy())
        steps.append(0)

        for step in range(1, max_steps + 1):
            # Perform Metropolis update
            self.metropolis_update()

            # Calculate the current energy
            current_energy = self.energy()

            # Store current state
            energies.append(current_energy)
            spins_list.append(self.spins.copy())
            steps.append(step)

            # If the new configuration has lower energy, store it
            if current_energy < min_energy:
                min_energy = current_energy
                min_spins = self.spins.copy()

            # Print progress every 1000 steps
            if step % 1000 == 0:
                print(f"Step {step}, Current Energy: {current_energy}, Minimum Energy so far: {min_energy}")
        
        return min_spins, min_energy, energies, spins_list, steps

# Example Usage
if __name__ == "__main__":
    # Define the grid size (e.g., 5x5 grid) and temperature
    grid_size = 25
    temperature = 1.0  # Lower temperatures reduce randomness, higher temperatures increase randomness

    # Initialize the 2D Boltzmann Machine based on the SK model
    boltzmann_machine = BoltzmannMachine2D(grid_size=grid_size, temperature=temperature)

    # Print the initial spin configuration and energy
    print("Initial Spin Configuration:")
    print(boltzmann_machine.spins)
    print("Initial Energy:", boltzmann_machine.energy())

    # Find the minimum energy configuration
    max_steps = 10000
    min_spins, min_energy, energies, spins_list, steps = boltzmann_machine.find_minimum_energy(max_steps=max_steps)

    # Print the final spin configuration and minimum energy
    print("Minimum Energy Spin Configuration:")
    print(min_spins)
    print("Minimum Energy:", min_energy)

    # Visualization
    # Create a figure with two subplots: one for the spin configuration, one for the energy
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
    ax2.set_xlim(0, max_steps)
    ax2.set_ylim(min(energies), max(energies))

    def animate(i):
        """Update function for animation."""
        im.set_data(spins_list[i])
        line.set_data(steps[:i + 1], energies[:i + 1])
        return im, line

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(spins_list), interval=50, blit=True, repeat_delay=1000)

    # Display the animation
    plt.tight_layout()
    plt.show()
```

## Summary of Key Mathematical Differences

| Model                         | Energy Function                                                                                  | Interaction Type                      | Key Property                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------- |
| **Boltzmann Machine (BM)**     | $E(v) = -\sum_{i < j} w_{ij} v_i v_j - \sum_{i} b_i v_i$                                          | Learnable weights $w_{ij}$            | Probabilistic learning of distributions        |
| **Sherrington-Kirkpatrick (SK) Spin Glass** | $H(\sigma) = -\sum_{i < j} J_{ij} \sigma_i \sigma_j$                                           | Random interactions $J_{ij}$          | Models frustration in disordered magnetic systems |
| **Hopfield Network**           | $E(\sigma) = -\sum_{i < j} J_{ij} \sigma_i \sigma_j$                                             | Learned or stored weights $J_{ij}$    | Content-addressable memory, retrieval of patterns |
| **2D Ising Model**             | $H(\sigma) = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_{i} \sigma_i$              | Uniform or fixed interactions $J$     | Phase transitions between ordered and disordered phases |
| **2D Spin Glass**              | $H(\sigma) = -\sum_{\langle i,j \rangle} J_{ij} \sigma_i \sigma_j$                               | Random interactions $J_{ij}$          | Complex energy landscape, frustration           |

Each of these models—Boltzmann Machines, Hopfield Networks, and the Ising model—captures different aspects of spin glass systems, either in their original physics context or as applied to machine learning. The key mathematical differences stem from the nature of the interaction terms: Boltzmann Machines and Hopfield Networks feature learnable interactions, while the SK model and 2D spin glasses introduce randomness and frustration, leading to more complex behavior. The Ising model provides a foundational structure but lacks the disorder that defines spin glasses

## 6. Conclusion
The Boltzmann machine, based on the Sherrington–Kirkpatrick model, provides a powerful framework for studying disordered spin systems. By implementing the Metropolis algorithm, we simulate the thermal fluctuations and observe how the system evolves towards lower energy configurations. The mathematical derivations ensure that our simulation accurately reflects the physics of spin glasses.
