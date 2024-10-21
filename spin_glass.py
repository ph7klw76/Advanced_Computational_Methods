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

def compute_total_energy(spins, J_h, J_v):
    """Compute the total energy of the current spin configuration."""
    energy = 0
    # Horizontal interactions
    energy += -np.sum(J_h * spins[:, :-1] * spins[:, 1:])
    # Vertical interactions
    energy += -np.sum(J_v * spins[:-1, :] * spins[1:, :])
    return energy

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

# X values for the energy plot
x_vals = iterations

# Initialize the energy plot limits
ax2.set_xlim(0, num_iterations)
ax2.set_ylim(min(energies), max(energies))

def animate(i):
    """Update function for animation."""
    im.set_data(spins_list[i])
    line.set_data(x_vals[:i + 1], energies[:i + 1])
    return im, line

# Create the animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(spins_list), interval=200, blit=True, repeat_delay=1000)

# Display the animation
plt.tight_layout()
plt.show()
