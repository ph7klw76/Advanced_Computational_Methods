#  GPU Computing in Python

**GPU computing in Python is basically**: keep big arrays on the GPU, do as much math there as possible, and avoid bouncing data back and forth to the CPU. The good news is you can do a lot of serious physics-style computing with just a few libraries and a couple of habits:

1. pick a GPU array/tensor library  
2. move data to GPU once  
3. run vectorized kernels (or write your own)  
4. only copy results back when you truly need them

---

## Most GPU work in Python falls into three “feels”:

###  NumPy-like  
**Use `CuPy`** (almost drop-in NumPy on NVIDIA GPUs).

###  Write your own kernels  
**Use `Numba CUDA`** for custom per-element logic.

###  Tensor frameworks  
**Use `PyTorch` or `JAX`** for tensor math, autodiff (great for inverse problems / fitting), and lots of optimized ops.

Examples
### 1) “Do I actually have a GPU available?”
CuPy (NumPy-on-GPU)

```python
import cupy as cp

x = cp.arange(10, dtype=cp.float32)
print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"])
print("x on GPU:", x)
```

## torch

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
x = torch.arange(10, device=device, dtype=torch.float32)
print(x)
```

###2) The #1 rule: avoid CPU↔GPU ping-pong

A common performance mistake is doing this repeatedly:

```python
# slow pattern (copies every loop)
for step in range(1000):
    x_cpu = x_gpu.get()          # GPU -> CPU
    x_cpu = x_cpu * 1.01
    x_gpu = cp.asarray(x_cpu)    # CPU -> GPU
```
Instead, keep arrays on GPU:

```python
import cupy as cp

x = cp.random.rand(10_000_000, dtype=cp.float32)
for step in range(1000):
    x *= 1.01   # stays on GPU
```


How to use torch 

### 3) Example: Projectile ensemble (ballistics / kinematics) as vectorized GPU math

Physics idea: simulate millions of projectiles with different initial velocities/angles and compute range/time-of-flight quickly.

```python
import cupy as cp

N = 5_000_000
g = 9.81

v0 = cp.random.uniform(10, 300, N, dtype=cp.float32)
theta = cp.random.uniform(0, cp.pi/2, N, dtype=cp.float32)

# time of flight (no drag): T = 2 v0 sin(theta) / g
T = 2 * v0 * cp.sin(theta) / g

# range: R = v0 cos(theta) * T = v0^2 sin(2 theta) / g
R = (v0**2) * cp.sin(2*theta) / g

print("mean range (m):", float(R.mean()))
print("max range (m):", float(R.max()))
```
### 4) Example: Monte Carlo estimate of π (or scattering acceptance) at huge scale

Physics idea: Monte Carlo is everywhere (radiation transport, statistical physics, path sampling). GPU shines because trials are independent.

```python
import cupy as cp

N = 200_000_000  # big
x = cp.random.random(N, dtype=cp.float32)
y = cp.random.random(N, dtype=cp.float32)

inside = (x*x + y*y) <= 1.0
pi_est = 4.0 * inside.mean()

print("pi ≈", float(pi_est))
```
Tip: for any Monte Carlo, try to structure it as large vector operations or batches.

### 5) Example: Maxwell–Boltzmann speed distribution sampling (statistical physics)

Physics idea: sample particle speeds at temperature T and compute mean kinetic energy.

For a 3D ideal gas, velocity components are Gaussian with variance 𝜎^2=𝑘𝑇/𝑚

```python
import cupy as cp

kB = 1.380649e-23
T = 300.0                # K
m = 4.65e-26             # kg (approx N2 molecule)

sigma = cp.sqrt(kB*T/m).astype(cp.float64)
N = 20_000_000

vx = sigma * cp.random.randn(N)
vy = sigma * cp.random.randn(N)
vz = sigma * cp.random.randn(N)

v2 = vx*vx + vy*vy + vz*vz
KE = 0.5*m*v2

print("⟨KE⟩ (J):", float(KE.mean()))
print("expected (3/2 kT):", 1.5*kB*T)
```

### 6) Example: FFT for wave physics (spectral methods, optics, diffraction)

Physics idea: FFT is central in wave propagation, diffraction, filtering, and spectral analysis. GPUs often accelerate FFT massively.
2D diffraction-like transform

```python
import cupy as cp

N = 4096
x = cp.linspace(-1, 1, N, dtype=cp.float32)
X, Y = cp.meshgrid(x, x)

# Example: circular aperture (pupil function)
R = cp.sqrt(X*X + Y*Y)
aperture = (R <= 0.25).astype(cp.complex64)

# Fraunhofer diffraction pattern ~ |FFT(aperture)|^2
F = cp.fft.fftshift(cp.fft.fft2(aperture))
I = cp.abs(F)**2
I /= I.max()

print("central intensity:", float(I[N//2, N//2]))
```

### 7) Example: Finite-difference heat equation (PDE) on GPU

Physics idea: solve diffusion/heat equation in 2D:

$$
u_t = \alpha (u_{xx} + u_{yy})
$$

Explicit scheme is stencil-heavy and GPU-friendly.

```python
import cupy as cp

N = 2048
alpha = 1.0
dx = 1.0
dt = 0.1 * dx*dx / (4*alpha)  # stability-ish for explicit 2D
steps = 2000

u = cp.zeros((N, N), dtype=cp.float32)

# initial hot spot
u[N//2-20:N//2+20, N//2-20:N//2+20] = 1.0

for _ in range(steps):
    u_new = u.copy()
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha*dt/dx**2 * (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]
    )
    u = u_new

print("u max:", float(u.max()))
```
This is pure array math—no custom kernels—and it already benefits from GPU parallelism.

8) Example: N-body gravity (or Coulomb) with a custom GPU kernel (Numba)

Physics idea: compute forces between particles. The naive 𝑂(𝑁^2)
is heavy but illustrates custom kernels well. (For large N in real work you’d use Barnes–Hut / PME / FMM, but the kernel approach is still useful.)

Note: Numba CUDA requires an NVIDIA GPU + CUDA setup.

```python
import numpy as np
from numba import cuda
import math

@cuda.jit
def nbody_accel(pos, mass, acc, G, eps2):
    i = cuda.grid(1)
    n = pos.shape[0]
    if i < n:
        ax = 0.0
        ay = 0.0
        az = 0.0
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        for j in range(n):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx*dx + dy*dy + dz*dz + eps2
            invr = 1.0 / math.sqrt(r2)
            invr3 = invr * invr * invr
            s = G * mass[j] * invr3
            ax += s * dx
            ay += s * dy
            az += s * dz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az

# setup
N = 8192
pos_h = np.random.randn(N, 3).astype(np.float64)
mass_h = np.abs(np.random.randn(N)).astype(np.float64) + 1e-3

pos_d = cuda.to_device(pos_h)
mass_d = cuda.to_device(mass_h)
acc_d = cuda.device_array((N, 3), dtype=np.float64)

threads = 256
blocks = (N + threads - 1) // threads
nbody_accel[blocks, threads](pos_d, mass_d, acc_d, 1.0, 1e-6)

acc_h = acc_d.copy_to_host()
print(acc_h[:3])
```

### 9) Example: Lorentz force particle pusher (charged particles in E and B fields)

Physics idea:

$$
\frac{d\mathbf{v}}{dt} = \frac{q}{m} \left( \mathbf{E} + \mathbf{v} \times \mathbf{B} \right)
$$

```python
import cupy as cp

N = 10_000_000
dt = 1e-3
q_over_m = 1.0

v = cp.random.randn(N, 3, dtype=cp.float32)
E = cp.array([0.0, 0.0, 1.0], dtype=cp.float32)
B = cp.array([0.0, 1.0, 0.0], dtype=cp.float32)

for _ in range(1000):
    vxB = cp.cross(v, B)              # vectorized cross product on GPU
    a = q_over_m * (E + vxB)
    v = v + dt * a

print("mean speed:", float(cp.linalg.norm(v, axis=1).mean()))
```

### 10) Example: Linear algebra for quantum mechanics (Hamiltonian eigenproblem)

Physics idea: many quantum problems reduce to building a Hamiltonian matrix and finding eigenvalues/eigenvectors. For large dense matrices, GPU linear algebra can help; for sparse, it depends on structure and library support.

Dense eigenvalues (toy)

```python
import cupy as cp

N = 2048
A = cp.random.randn(N, N, dtype=cp.float32)
H = (A + A.T) * 0.5  # symmetric "Hamiltonian-like"

w = cp.linalg.eigvalsh(H)  # eigenvalues on GPU
print("lowest 5 eigenvalues:", w[:5].get())
```
For real Hamiltonians, you’d build structured matrices (banded, sparse). GPU sparse eigensolvers exist but usage varies by stack.

### quantum tunneling of a Gaussian wavepacket through a finite potential barrier, solved with the split-operator FFT method.

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # workaround to avoid crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- GPU setup (falls back to CPU if no CUDA GPU is available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 1D time-dependent Schrödinger equation: Gaussian wavepacket tunneling through a barrier ---
# Units: ħ = 1, m = 1  (dimensionless "natural" units)

# Grid
N = 4096                 # number of spatial points
L = 200.0                # spatial domain length
dx = L / N
x = torch.linspace(-L/2, L/2 - dx, N, device=device, dtype=torch.float32)

# Wave numbers for FFT
k = 2*np.pi*torch.fft.fftfreq(N, d=dx).to(device=device, dtype=torch.float32)

# Initial Gaussian wavepacket
sigma = 5.0
x0 = -60.0
k0 = 2.0                 # mean momentum (to the +x direction)

norm = (1.0 / (2.0*np.pi*sigma**2))**0.25
psi = norm * torch.exp(-(x - x0)**2 / (4.0*sigma**2)) * torch.exp(1j * k0 * x.to(torch.complex64))
psi = psi.to(torch.complex64)

# Potential barrier: height V0 over width a centered at x=0
V0 = 1.5
a = 10.0
V = torch.where(torch.abs(x) < (a/2), torch.tensor(V0, device=device, dtype=torch.float32), torch.tensor(0.0, device=device, dtype=torch.float32))

# Time step and number of steps
dt = 0.05
steps = 800

# Split-operator precomputations
# psi(t+dt) ≈ e^{-iV dt/2} * F^{-1}[ e^{-i k^2 dt/2} * F[ e^{-iV dt/2} psi(t) ] ]
expV = torch.exp(-0.5j * V.to(torch.complex64) * dt)
expK = torch.exp(-0.5j * (k.to(torch.complex64)**2) * dt)

# Save initial density for plotting
with torch.no_grad():
    dens0 = (psi.abs()**2)

# Time evolution
with torch.no_grad():
    for _ in range(steps):
        psi = expV * psi
        psi_k = torch.fft.fft(psi)
        psi_k = expK * psi_k
        psi = torch.fft.ifft(psi_k)
        psi = expV * psi

densT = (psi.abs()**2)

# Bring data to CPU for plotting
x_cpu = x.detach().cpu().numpy()
dens0_cpu = dens0.detach().cpu().numpy()
densT_cpu = densT.detach().cpu().numpy()
V_cpu = V.detach().cpu().numpy()

# Normalize densities to make the plot scale-friendly (optional for visualization)
dens0_cpu /= dens0_cpu.max()
densT_cpu /= densT_cpu.max()
V_plot = V_cpu / V0  # scaled barrier (0 to 1)

# Plot
plt.figure(figsize=(10, 4.5))
plt.plot(x_cpu, dens0_cpu, label="|ψ(x,0)|² (initial)")
plt.plot(x_cpu, densT_cpu, label="|ψ(x,t)|² (after tunneling)")
plt.plot(x_cpu, V_plot, label="Barrier V(x) (scaled)", linestyle="--")
plt.xlabel("x (arbitrary units)")
plt.ylabel("Normalized probability density / scaled potential")
plt.title("Quantum tunneling: Gaussian wavepacket scattering from a barrier (split-operator FFT)")
plt.legend()
plt.tight_layout()
plt.show()
```
<img width="984" height="556" alt="image" src="https://github.com/user-attachments/assets/99949828-2812-408a-be5f-d8ef8412f35a" />



### Animation

```python

# =========================
# Quantum tunneling animation (GPU if available)
# Split-operator FFT for 1D TDSE
# =========================

import os
# --- Workaround for Windows OpenMP runtime clashes (must be BEFORE numpy/torch imports) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Optional: reduce CPU thread contention (harmless even on GPU)
torch.set_num_threads(1)

# -------------------------
# Physics + numerics setup
# -------------------------
# Natural units: ħ = 1, m = 1
# TDSE: i dψ/dt = [ -1/2 d²/dx² + V(x) ] ψ

N = 2048          # spatial points (increase for accuracy; 4096 is nicer if you have GPU)
L = 200.0         # domain length
dx = L / N
x = torch.linspace(-L/2, L/2 - dx, N, device=device, dtype=torch.float32)

# k grid for FFT
k = (2*np.pi) * torch.fft.fftfreq(N, d=dx).to(device=device, dtype=torch.float32)

# Initial Gaussian wavepacket
sigma = 5.0
x0 = -70.0
k0 = 2.0          # mean momentum to +x

# Complex wavefunction
psi = torch.exp(-(x - x0)**2 / (4.0*sigma**2)).to(torch.complex64)
psi *= torch.exp(1j * (k0 * x).to(torch.complex64))

# Normalize: ∫|ψ|² dx = 1
with torch.no_grad():
    norm = torch.sqrt((psi.abs()**2).sum() * dx)
    psi /= norm

# Finite barrier centered at x=0
V0 = 1.5
a = 10.0
V = torch.where(torch.abs(x) < (a/2),
                torch.tensor(V0, device=device, dtype=torch.float32),
                torch.tensor(0.0, device=device, dtype=torch.float32))

# Time stepping
dt = 0.05
steps = 1400

# Record frames every N steps
record_every = 10
nframes = steps // record_every

# Precompute split-operator factors:
# ψ(t+dt) ≈ e^{-iV dt/2} * FFT^{-1}[ e^{-i k^2 dt/2} * FFT[ e^{-iV dt/2} ψ(t) ] ]
expV = torch.exp(-0.5j * V.to(torch.complex64) * dt)
expK = torch.exp(-0.5j * (k.to(torch.complex64)**2) * dt)

# -------------------------
# Run simulation & collect frames
# -------------------------
frames = []
times = []

with torch.no_grad():
    # save initial frame
    dens = (psi.abs()**2)
    frames.append(dens.detach().cpu().numpy())
    times.append(0.0)

    for n in range(1, steps + 1):
        psi = expV * psi
        psi_k = torch.fft.fft(psi)
        psi_k = expK * psi_k
        psi = torch.fft.ifft(psi_k)
        psi = expV * psi

        # Optional: mild renormalization to counter drift (comment out if you want)
        if (n % 50) == 0:
            norm = torch.sqrt((psi.abs()**2).sum() * dx)
            psi /= norm

        if (n % record_every) == 0:
            dens = (psi.abs()**2)
            frames.append(dens.detach().cpu().numpy())
            times.append(n * dt)

x_cpu = x.detach().cpu().numpy()
V_cpu = V.detach().cpu().numpy()

# Scale barrier for overlay in same plot
V_plot = V_cpu / V0 if V0 != 0 else V_cpu

# Normalize densities for nice visualization (keeps plot readable)
max_d = max(f.max() for f in frames)
frames = [f / max_d for f in frames]

# -------------------------
# Animate with Matplotlib
# -------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))

line_dens, = ax.plot(x_cpu, frames[0], lw=2, label=r"$|\psi(x,t)|^2$")
line_V, = ax.plot(x_cpu, V_plot, ls="--", label="Barrier V(x) (scaled)")

time_text = ax.text(0.02, 0.92, "", transform=ax.transAxes)

ax.set_xlim(x_cpu.min(), x_cpu.max())
ax.set_ylim(0.0, 1.1)
ax.set_xlabel("x (arbitrary units)")
ax.set_ylabel("Normalized probability density / scaled potential")
ax.set_title("Quantum tunneling: Gaussian wavepacket vs finite barrier (split-operator FFT)")
ax.legend(loc="upper right")

def update(i):
    line_dens.set_ydata(frames[i])
    time_text.set_text(f"t = {times[i]:.2f}")
    return line_dens, time_text

anim = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)

# -------------------------
# Save outputs
# -------------------------
# 1) GIF (easy, uses Pillow)
gif_name = "quantum_tunneling.gif"
anim.save(gif_name, writer=PillowWriter(fps=30))
print("Saved GIF:", gif_name)


plt.show()
```

