import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SpinGlass2D:
    def __init__(self, grid_size):
        """
        Initialize the 2D Spin Glass System.
        :param grid_size: The size of the 2D grid (grid_size x grid_size).
        """
        self.grid_size = grid_size
        
        # Initialize the spins randomly to -1 or 1 (2D grid of spins)
        self.spins = np.random.choice([-1, 1], size=(grid_size, grid_size))
        
        # Initialize the interaction weights (J_ij) between neighboring spins randomly
        # For simplicity, we consider nearest neighbor interactions (North, South, East, West)
        self.weights = {
            'horizontal': np.random.randn(grid_size, grid_size - 1),  # Horizontal interactions (left-right)
            'vertical': np.random.randn(grid_size - 1, grid_size)     # Vertical interactions (top-bottom)
        }

    def energy(self):
        """
        Compute the energy of the 2D spin glass system.
        The energy is computed as the sum of interactions between neighboring spins.
        :return: The total energy of the system.
        """
        E = 0
        
        # Sum over horizontal interactions (between neighboring spins horizontally)
        E -= np.sum(self.weights['horizontal'] * self.spins[:, :-1] * self.spins[:, 1:])
        
        # Sum over vertical interactions (between neighboring spins vertically)
        E -= np.sum(self.weights['vertical'] * self.spins[:-1, :] * self.spins[1:, :])
        
        return E

    def sign(self, x):
        """
        Sign activation function. Returns -1 for negative inputs and 1 for positive inputs.
        """
        return 1 if x >= 0 else -1

    def update_spin(self):
        """
        Asynchronously update a randomly selected spin based on its neighbors.
        :return: None
        """
        # Randomly select a spin (i, j) in the 2D grid
        i, j = np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)
        
        # Compute the local field acting on spin (i, j) from its neighbors
        local_field = 0
        
        # Horizontal neighbor interaction (left and right)
        if j > 0:  # Left neighbor
            local_field += self.weights['horizontal'][i, j - 1] * self.spins[i, j - 1]
        if j < self.grid_size - 1:  # Right neighbor
            local_field += self.weights['horizontal'][i, j] * self.spins[i, j + 1]
        
        # Vertical neighbor interaction (up and down)
        if i > 0:  # Top neighbor
            local_field += self.weights['vertical'][i - 1, j] * self.spins[i - 1, j]
        if i < self.grid_size - 1:  # Bottom neighbor
            local_field += self.weights['vertical'][i, j] * self.spins[i + 1, j]
        
        # Update the spin at (i, j) based on the sign of the local field
        self.spins[i, j] = self.sign(local_field)

    def find_minimum_energy(self, max_steps=1000):
        """
        Find the minimum energy configuration of the system.
        :param max_steps: The maximum number of update steps.
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
            # Update the spin configuration asynchronously
            self.update_spin()
            
            # Calculate the current energy
            current_energy = self.energy()
            
            # Store the current state
            energies.append(current_energy)
            spins_list.append(self.spins.copy())
            steps.append(step)

            # If the new configuration has lower energy, store it
            if current_energy < min_energy:
                min_energy = current_energy
                min_spins = self.spins.copy()

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}, Current Energy: {current_energy}, Minimum Energy so far: {min_energy}")
        
        return min_spins, min_energy, energies, spins_list, steps

# Example Usage
if __name__ == "__main__":
    # Define the grid size (e.g., 5x5 grid)
    grid_size = 25

    # Initialize the 2D spin-glass system with random spins and random interactions
    spin_glass_2d = SpinGlass2D(grid_size=grid_size)

    # Print the initial spin configuration and energy
    print("Initial Spin Configuration:")
    print(spin_glass_2d.spins)
    print("Initial Energy:", spin_glass_2d.energy())

    # Find the minimum energy configuration
    max_steps = 1000
    min_spins, min_energy, energies, spins_list, steps = spin_glass_2d.find_minimum_energy(max_steps=max_steps)

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
