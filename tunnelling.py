import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter  # Import the PillowWriter for saving GIFs

# Constants
hbar = 1.0       # Reduced Planck's constant
m = 1.0          # Particle mass

# Spatial grid
dx = 0.1
x_min = -50.0
x_max = 50.0
x = np.arange(x_min, x_max, dx)
N = len(x)

# Momentum grid
dk = 2 * np.pi / (N * dx)
k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi

# Time grid
dt = 0.05
t_max = 20.0
t = np.arange(0, t_max, dt)
num_t = len(t)

# Initial Gaussian wave packet
x0 = -10.0       # Initial position
p0 = 2.0         # Initial momentum (positive, moving right)
sigma = 1.0      # Width of the wave packet

def psi_initial(x):
    """Initial Gaussian wave packet."""
    return (1 / (np.sqrt(sigma * np.sqrt(np.pi)))) * np.exp(1j * p0 * x / hbar) * np.exp(-((x - x0)**2) / (2 * sigma**2))

# Potential barrier
V0 = 15           # Barrier height
a = 5.0             # Barrier half-width

def V(x):
    """Potential barrier."""
    return np.where(np.abs(x) <= a, V0, 0)

# Initial wavefunction
psi = psi_initial(x)

# Potential energy operator
V_x = V(x)
exp_V = np.exp((-1j * V_x * dt) / (2 * hbar))

# Kinetic energy operator in momentum space
T_k = (hbar * k**2) / (2 * m)
exp_T = np.exp(-1j * T_k * dt)

# Prepare for time evolution
psi_t = np.zeros((num_t, N), dtype=complex)
psi_t[0, :] = psi

# Compute probabilities
def compute_probabilities(psi, x, a):
    """Compute transmission and reflection probabilities."""
    x_barrier_left = -a
    x_barrier_right = a

    # Reflection region: x < x_barrier_left
    prob_reflect = np.trapz(np.abs(psi[x < x_barrier_left])**2, x[x < x_barrier_left])

    # Transmission region: x > x_barrier_right
    prob_transmit = np.trapz(np.abs(psi[x > x_barrier_right])**2, x[x > x_barrier_right])

    # Probability in barrier region
    prob_barrier = np.trapz(np.abs(psi[(x >= x_barrier_left) & (x <= x_barrier_right)])**2, x[(x >= x_barrier_left) & (x <= x_barrier_right)])

    # Total probability
    total_prob = prob_reflect + prob_transmit + prob_barrier

    return prob_reflect, prob_transmit, prob_barrier, total_prob

# Time evolution using the split-operator method
for i in range(1, num_t):
    # Apply half potential operator
    psi = exp_V * psi

    # Fourier transform to momentum space
    psi_k = np.fft.fftshift(np.fft.fft(psi))

    # Apply kinetic operator
    psi_k = exp_T * psi_k

    # Inverse Fourier transform back to position space
    psi = np.fft.ifft(np.fft.ifftshift(psi_k))

    # Apply half potential operator
    psi = exp_V * psi

    # Normalize the wavefunction
    norm = np.trapz(np.abs(psi)**2, x)
    psi /= np.sqrt(norm)

    # Store the wavefunction at this time step
    psi_t[i, :] = psi

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, np.abs(psi_t[0, :])**2, label='|Psi(x,t)|^2')
ax.plot(x, V(x) / np.max(V(x)) * np.max(np.abs(psi_t[0, :])**2) * 0.8, label='Potential Barrier', linestyle='--')
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, np.max(np.abs(psi_t[0, :])**2) * 1.2)
ax.set_xlabel('Position x')
ax.set_ylabel('$|Psi(x, t)|^2$')
ax.set_title('Quantum Tunneling Simulation')

# Add probability annotations
text_reflect = ax.text(0.05, 0.85, '', transform=ax.transAxes)
text_transmit = ax.text(0.05, 0.80, '', transform=ax.transAxes)
text_barrier = ax.text(0.05, 0.75, '', transform=ax.transAxes)

# Animate
def animate(i):
    line.set_ydata(np.abs(psi_t[i, :])**2)

    # Compute and display probabilities
    prob_reflect, prob_transmit, prob_barrier, prob_total = compute_probabilities(psi_t[i, :], x, a)
    text_reflect.set_text(f'Reflection: {prob_reflect:.3f}')
    text_transmit.set_text(f'Transmission: {prob_transmit:.3f}')
    text_barrier.set_text(f'Barrier: {prob_barrier:.3f}')

    return line, text_reflect, text_transmit, text_barrier

anim = FuncAnimation(fig, animate, frames=num_t, interval=50, blit=True)
plt.legend()

# Save the animation as a GIF
gif_writer = PillowWriter(fps=20)  # Set frames per second
anim.save("quantum_tunneling.gif", writer=gif_writer)

plt.show()

# Final transmission and reflection probabilities
psi_final = psi_t[-1, :]
prob_reflect, prob_transmit, prob_barrier, prob_total = compute_probabilities(psi_final, x, a)
print(f"Final Reflection Probability: {prob_reflect:.4f}")
print(f"Final Transmission Probability: {prob_transmit:.4f}")
print(f"Probability in Barrier: {prob_barrier:.4f}")
print(f"Total Probability: {prob_total:.6f}")
