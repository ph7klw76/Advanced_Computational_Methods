# Speeding up calculation using Numba libraries with examples

# Quantifying Quantum Delocalization: The Role of

$$
\langle x^2 \rangle = \int_{-\infty}^{\infty} x^2 \, |\psi_0(x)|^2 \, dx
$$

In both foundational theory and cutting-edge applications, the second moment of a quantum wavefunction $\langle x^2 \rangle$ serves as a key diagnostic of spatial delocalization, zero-point motion, and coupling strengths. Below, we develop the concept rigorously, survey its real-world impact, and demonstrate how Numba can accelerate Monte Carlo estimates of $\langle x^2 \rangle$ even for complex potentials.

## 1. Theoretical Foundations

Consider a one-dimensional quantum system with normalized ground-state wavefunction $\psi_0(x)$. The variance in position is

$$
\text{Var}(x) = \langle x^2 \rangle - \langle x \rangle^2 = \int x^2 \, |\psi_0(x)|^2 \, dx \quad (\text{for } \langle x \rangle = 0).
$$

This rms displacement:

- Encodes zero-point motion—the irreducible quantum fluctuations even at $T=0$.
- Saturates the uncertainty bound, since $\Delta x = \sqrt{\langle x^2 \rangle}$ pairs with $\Delta p = \hbar / (2 \Delta x)$ in minimum-uncertainty states.
- Determines average potential energy in harmonic or anharmonic traps:

$$
\frac{1}{2} m \omega^2 \langle x^2 \rangle.
$$

Beyond pedagogy, $\langle x^2 \rangle$ appears directly in:

- Debye–Waller factors for X-ray and neutron scattering (solid-state physics).
- Transition dipole moments $\langle 0|x|1 \rangle$ in molecular spectroscopy.
- Nanomechanical resonator amplitudes, where RMS displacement informs force sensitivity.
- Quantum dot confinements, setting electron–phonon coupling constants.

## 2. Real-World Applications

### 2.1 Molecular Vibrations & Spectroscopy

<img width="500" height="433" alt="image" src="https://github.com/user-attachments/assets/074c44bb-8463-4ffe-bcd2-a99468f2c6f3" />

In diatomic molecules modeled by a Morse potential:

$$
V(x) = D_e (1 - e^{-\alpha x})^2,
$$

the ground-state $\langle x^2 \rangle$ predicts bond-length fluctuations, which shift infrared absorption lines and Raman scattering intensities.

### 2.2 Trapped Ion Qubits

<img width="332" height="165" alt="image" src="https://github.com/user-attachments/assets/a9e1bb14-f797-4dd1-9875-bcc63a4c9816" />

Ions in a Paul trap have motional ground states approximated by a harmonic potential. The rms width $\sqrt{\langle x^2 \rangle}$ sets the Lamb-Dicke parameter:

$$
\eta = k \sqrt{\langle x^2 \rangle},
$$

controlling coupling between internal qubit states and motional modes.

### 2.3 Nanomechanical Sensors

<img width="242" height="208" alt="image" src="https://github.com/user-attachments/assets/4af69b08-3048-4511-947b-d3ac31be79e5" />

Cantilever or membrane resonators operating near the quantum regime rely on zero-point displacement noise:

$$
\sqrt{\frac{\hbar}{2m\omega}}, \quad \text{i.e., } \sqrt{\langle x^2 \rangle},
$$


to define their ultimate force or displacement sensitivity.

## 3. Monte Carlo Estimation with Numba

Closed-form integrals exist for simple potentials, but for complex, multidimensional or anharmonic systems, we turn to importance-sampled Monte Carlo:

$$
\langle x^2 \rangle = \int x^2 \, p(x) \, dx = \int x^2 \, \frac{p(x)}{q(x)} \, q(x) \, dx \approx \frac{1}{N} \sum_{i=1}^{N} x_i^2 \, w(x_i),
$$

where $x_i \sim q(x)$, and $w(x) = p/q$. Here’s an accelerated implementation.

```python
import numpy as np
import time
from numba import njit, prange

# --- 3.1 Define target density p(x)=|ψ₀|² for HO or Morse --- #
@njit(fastmath=True)
def psi0_sq_harmonic(x):
    # m=ω=ℏ=1
    return (1/np.sqrt(np.pi)) * np.exp(-x*x)

@njit(fastmath=True)
def psi0_sq_morse(x, D_e=5.0, α=1.0):
    # Approximate ground-state |ψ₀|² via analytic form (for demonstration)
    z = 2 * np.sqrt(2*D_e)/α * np.exp(-α*x)
    # Proper normalization omitted for brevity
    return z**(2*np.sqrt(2*D_e)/α - 1) * np.exp(-z)

# --- 3.2 Importance sampler: proposal q(x)=N(0,σ²) --- #
σ = 2.0
@njit(fastmath=True)
def q_pdf(x):
    return 1/(np.sqrt(2*np.pi)*σ) * np.exp(-x*x/(2*σ*σ))

# --- 3.3 Numba-accelerated MC estimator --- #
@njit(parallel=True, fastmath=True)
def mc_expectation(num_samples, seed, target_type=0):
    np.random.seed(seed)
    total = 0.0
    for i in prange(num_samples):
        x = np.random.normal(0.0, σ)
        if target_type == 0:
            p = psi0_sq_harmonic(x)
        else:
            p = psi0_sq_morse(x)
        w = p / q_pdf(x)
        total += x*x * w
    return total / num_samples

# --- 3.4 Benchmarking --- #
if __name__ == "__main__":
    N = 2_000_000
    # Warm up Numba
    mc_expectation(10_000, 0, 0)
    for target in (0, 1):
        t0 = time.time()
        result = mc_expectation(N, seed=12345, target_type=target)
        dt = time.time() - t0
        label = "Harmonic" if target==0 else "Morse"
        print(f"{label:8s} ⟨x²⟩ ≈ {result:.5f}  in {dt:.3f}s")
```
@njit(parallel=True) and prange distribute the N-sample loop over CPU threads.
fastmath=True vectorizes exponentials and arithmetic.
Seeding once at entry ensures reproducibility (with thread-dependent interleaving).
On a 6-core machine this typically yields a 20–50× speed-up over a pure-Python/NumPy loop.

High-performance computing is at the heart of modern computational physics. While Python offers a rich ecosystem for scientific programming, its interpreter overhead can become a bottleneck when tight loops or complex algorithms are involved. Numba bridges this gap by compiling selected Python functions to optimized machine code at runtime, often delivering C- or Fortran-like performance with minimal code changes. In this post, we’ll explore how Numba works under the hood, illustrate its key features with Python examples, and demonstrate its impact on two canonical physics problems.
To learn more about Numba click [here](https://numba.pydata.org/)

### 4.1 Monte Carlo Sampling of $\langle r^2 \rangle$ in 3D

Physically, many problems reduce to asking “what is the average squared displacement” under some distribution of steps—e.g. thermal kicks in a fluid, zero-point vibrational amplitudes in a crystal, or fluctuations in a nanomechanical resonator. In 3D,

$$
\langle r^2 \rangle = \int_{\mathbb{R}^3} (x^2 + y^2 + z^2) \, p(x, y, z) \, d^3r \approx \frac{1}{N} \sum_{i=1}^{N} (x_i^2 + y_i^2 + z_i^2),
$$

where $(x_i, y_i, z_i)$ are drawn from the target distribution—or via importance sampling.

#### 4.1.1 Nested Loops with `prange`
```python
import numpy as np
from numba import njit, prange
import time

σ = 1.0  # assume unit-variance isotropic Gaussian

@njit(parallel=True, fastmath=True)
def mc_r2_3d_nested(n_particles, n_samples_per, seed):
    """
    Monte Carlo: for each 'particle' we draw n_samples_per
    3D Gaussian displacements and accumulate r^2.
    """
    np.random.seed(seed)
    total = 0.0
    for i in prange(n_particles):
        for j in range(n_samples_per):
            x, y, z = np.random.normal(0.0, σ, size=3)
            total += x*x + y*y + z*z
    return total / (n_particles * n_samples_per)

# benchmark
if __name__ == "__main__":
    Np, Ns = 5000, 2000
    # warm-up compile
    mc_r2_3d_nested(10, 10, 0)
    t0 = time.time()
    val = mc_r2_3d_nested(Np, Ns, seed=42)
    print(f"Nested: ⟨r²⟩ ≈ {val:.5f} in {time.time()-t0:.3f}s")
```

Outer prange distributes particles across threads.
Inner loop stays serial, but overall work scales as O(Np×Ns)

### 4.2 Extracting MSD from Trajectory Data

In molecular dynamics or Brownian‐motion experiments, you often have a trajectory $\{r_i(t_k)\}$ for $i = 1 \dots N$ particles over times $t_k$. The mean-square displacement (MSD) at lag $\Delta t = t_{k+\ell} - t_k$ is

$$
\text{MSD}(\Delta t) = \frac{1}{NM} \sum_{i=1}^{N} \sum_{k=1}^{M} \|r_i(t_{k+\ell}) - r_i(t_k)\|^2.
$$

Below is a Numba-accelerated routine to compute MSD for a single lag $\ell$. We assume `positions` is a `float64` array of shape `(N_particles, N_frames, 3)`.

```python
import numpy as np
from numba import njit, prange
import time

@njit(parallel=True, fastmath=True)
def compute_msd(positions, lag):
    Np, Nf, _ = positions.shape
    M = Nf - lag
    total = 0.0

    # Loop over particles in parallel
    for i in prange(Np):
        for k in range(M):
            dx = positions[i, k+lag, 0] - positions[i, k, 0]
            dy = positions[i, k+lag, 1] - positions[i, k, 1]
            dz = positions[i, k+lag, 2] - positions[i, k, 2]
            total += dx*dx + dy*dy + dz*dz

    # Average over all (i, k)
    return total / (Np * M)

# Example usage
if __name__ == "__main__":
    # synthetic random-walk data
    Np, Nf = 1000, 500
    rng = np.random.default_rng(0)
    steps = rng.normal(scale=0.1, size=(Np, Nf, 3))
    positions = np.cumsum(steps, axis=1)

    # warm-up
    compute_msd(positions, lag=10)

    t0 = time.time()
    msd10 = compute_msd(positions, lag=10)
    print(f"MSD(Δt=10) = {msd10:.5f}  computed in {time.time()-t0:.3f}s")
```
## Why This Matters

The long-time slope of $\text{MSD}(\Delta t)$ vs.\ $\Delta t$ gives the diffusion coefficient:

$$
D = \frac{1}{6} \frac{d\langle r^2 \rangle}{dt}.
$$

- In biophysics, one tracks particle motions in cells to extract viscosity or active transport rates.
- In materials science, MSD of atoms over time reveals melting transitions or glassy arrest.

## 4.3 Performance Tips & Best Practices

- **Seed once at function entry** to maintain reproducibility—accept that interleaving across threads may differ if you change thread-count.
- **Flatten loops** when per-iteration work is uniform; nested `prange` can lead to imbalance if inner loop counts vary.
- **Contiguous arrays**: ensure `positions` and any large arrays are C-contiguous floats for best memory throughput.
- **`fastmath`** yields significant vectorization of exponentials and divisions—essential in thermal-motion or Coulomb‐force loops.
- **Cache warming**: the first call pays compilation overhead; subsequent calls run at full speed.

## Using `@njit(..., cache=True)` in Numba

When you decorate a Numba-compiled function with `@njit(..., cache=True)`, you’re telling Numba to persist the generated native machine code (and its LLVM IR) to disk—typically under a `__pycache__` directory next to your Python module—so that on later runs (or in subsequent processes) you skip the costly compile step. Here’s why that matters:

### Avoiding Recompilation Overhead

On first invocation, Numba must parse your Python function, perform type inference, lower to LLVM IR, run LLVM’s optimizations, and emit machine code. That can easily be 50–200 ms (or more for complex functions).

By caching, subsequent imports or calls of the same function signature simply load the precompiled object from disk, so your code “starts up” at C-like speed immediately.

### Faster Iteration in Development

When you’re experimenting interactively—tweaking parameters, rerunning scripts, or re-importing modules—repeated compilation can become a painful drag. Cache lets you modify unrelated functions or code paths without paying to rebuild your hot kernels each time.

### Consistent Performance in Production

On HPC clusters or cloud deployments where cold-start time matters, caching ensures that worker processes (e.g.\ in a Dask or MPI pool) won’t all recompile the same kernels in parallel. They’ll share the on-disk cache, leading to faster overall job startup and reduced I/O contention.

### Discrete Cache Keys by Signature

Numba keys the cache on the combination of function bytecode, argument types, and compiler flags (`parallel`, `fastmath`, etc.). If you change your function’s signature (e.g. start passing `float32` instead of `float64`) or toggle `fastmath`, Numba will recompile and cache the new variant.

### Trade-Offs

- **Disk Usage**: Each cached variant consumes a few hundred kilobytes to a few megabytes, depending on complexity.
- **Stale Caches**: If you edit the function’s code but keep the same signature, Numba will detect the change (via bytecode hash) and rebuild. You can also manually clear `__pycache__` if needed.

---

In practice, for any non-trivial function—Monte Carlo loops, N-body force kernels, FFT-based solvers—adding `cache=True` is a no-brainer: it turns that one-time compilation cost into something you pay just once per code change, rather than every time you launch your analysis or simulation script.

# 1. Newton’s Law of Universal Gravitation

For a system of $N$ particles with masses $m_i$ and position vectors $r_i$, the pairwise gravitational force on particle $i$ due to particle $j$ is

$$
F_{ij} = -G \, \frac{m_i m_j}{\|r_i - r_j\|^3} (r_i - r_j),
$$

where:

- $G$ is the gravitational constant (here set to 1 in arbitrary units),
- $\|r_i - r_j\|$ is the Euclidean distance between particles $i$ and $j$.

Summing over all $j \ne i$, the total force on particle $i$ is

$$
F_i = \sum_{\substack{j=1 \\ j \ne i}}^{N} F_{ij}.
$$

# 2. Equations of Motion and Time Integration

Newton’s second law gives:

$$
m_i \, \ddot{r}_i = F_i.
$$

We convert this second-order ODE into two first-order updates:

$$
v_i(t + \Delta t) = v_i(t) + \frac{\Delta t}{m_i} F_i(t),
$$

$$
r_i(t + \Delta t) = r_i(t) + \Delta t \, v_i(t + \Delta t).
$$

This explicit Euler scheme is simple but only first-order accurate in time. More sophisticated integrators (e.g., leapfrog, Runge–Kutta, or symplectic methods) improve stability and energy conservation.

```python
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Physical constants ───────────────────────────────────────────────
pc_to_m   = 3.0857e16             # 1 parsec in meters
ly_to_pc  = 0.306601              # 1 light‐year in parsecs
G_SI      = 6.67430e-11           # gravitational constant (m³ kg⁻¹ s⁻²)
M_sun     = 1.98847e30            # solar mass (kg)

# ── Simulation parameters ────────────────────────────────────────────
N_stars   = 1000                   # number of stars
R_ly      = 1000.0                # sphere radius in light‐years
R_pc      = R_ly * ly_to_pc       # convert to parsecs
dt        = 1e13                  # timestep in seconds (~0.3 Myr)
n_steps   = 50000                   # total steps
plot_every= 5                     # update plot every this many steps

# ── Initial conditions ──────────────────────────────────────────────
# Uniformly distribute N_stars within a sphere of radius R_pc (pc)
u    = np.random.rand(N_stars)
cost = 2*np.random.rand(N_stars) - 1
phi  = 2*np.pi*np.random.rand(N_stars)
r_pc = R_pc * np.cbrt(u)
sin_t = np.sqrt(1 - cost**2)

pos_pc = np.empty((N_stars,3))
pos_pc[:,0] = r_pc * sin_t * np.cos(phi)
pos_pc[:,1] = r_pc * sin_t * np.sin(phi)
pos_pc[:,2] = r_pc * cost

pos = pos_pc * pc_to_m            # positions in meters
vel = np.zeros((N_stars,3), dtype=np.float64)

# Sample masses from Salpeter IMF (0.1–50 M⊙)
α, mmin, mmax = 2.35, 0.1, 50.0
u2 = np.random.rand(N_stars)
mass_sun = (mmin**(1-α) + u2*(mmax**(1-α)-mmin**(1-α)))**(1/(1-α))
mass = mass_sun * M_sun

# ── Numba‐accelerated kernels ────────────────────────────────────────
@njit(fastmath=True)
def compute_forces(pos, mass):
    N = pos.shape[0]
    F  = np.zeros_like(pos)
    for i in range(N):
        xi, yi, zi = pos[i]
        mi         = mass[i]
        fxi = fyi = fzi = 0.0
        for j in range(N):
            if i == j: continue
            dx = pos[j,0] - xi
            dy = pos[j,1] - yi
            dz = pos[j,2] - zi
            r2 = dx*dx + dy*dy + dz*dz + 1e-12
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            f = G_SI * mi * mass[j] * inv_r3
            fxi += f * dx
            fyi += f * dy
            fzi += f * dz
        F[i,0], F[i,1], F[i,2] = fxi, fyi, fzi
    return F

@njit(parallel=True, fastmath=True)
def advance(pos, vel, mass, dt):
    F = compute_forces(pos, mass)
    for i in prange(pos.shape[0]):
        vel[i] += F[i] * dt / mass[i]
        pos[i] += vel[i] * dt
    return pos, vel

# ── Prepare animation ────────────────────────────────────────────────
# Warm‐up compile
advance(pos, vel, mass, dt)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-R_pc, R_pc)
ax.set_ylim(-R_pc, R_pc)
ax.set_xlabel('x (pc)')
ax.set_ylabel('y (pc)')
ax.set_title(f'{N_stars} Stars within {int(R_ly)} ly of the Sun')

# Scatter initial positions (convert back to pc)
scatter = ax.scatter(pos[:,0]/pc_to_m, pos[:,1]/pc_to_m, s=10)

def update(frame):
    global pos, vel
    # advance `plot_every` steps between frames
    for _ in range(plot_every):
        pos, vel = advance(pos, vel, mass, dt)
    scatter.set_offsets(pos[:,:2]/pc_to_m)
    return (scatter,)

ani = FuncAnimation(fig, update,
                    frames=n_steps//plot_every,
                    interval=50, blit=True)

# Display the animation in an interactive window
plt.show()


```

<img width="688" height="687" alt="image" src="https://github.com/user-attachments/assets/133c53d4-13be-40d1-921e-f3a78dd305a0" />

# 3. Computational Complexity

- **Force calculation**: Each of the $N$ particles interacts with $N - 1$ others ⇒ $\mathcal{O}(N^2)$ work per time step.
- **Time stepping**: Once forces are known, updating $\{v_i, r_i\}$ is $\mathcal{O}(N)$.

**Overall cost per frame**: $\mathcal{O}(N^2)$. For large $N$, one often resorts to:

- **Barnes–Hut** or **Fast Multipole Methods** to reduce to $\mathcal{O}(N \log N)$ or even $\mathcal{O}(N)$.
- **GPU acceleration** for massively parallel execution.
- **Symplectic integrators** to permit larger $\Delta t$.

# 4. Numba Acceleration

In our Python code:

- `@njit(fastmath=True)` on `compute_forces` compiles the double loop over $i, j$ into optimized machine code, inlining arithmetic and using vector instructions for multiplications and divisions.
- `@njit(parallel=True)` on `advance` lets Numba distribute the $N$-particle updates across CPU cores via `prange`, yielding near-linear scaling for the $\mathcal{O}(N)$ step.
- `cache=True` (if enabled) would store compiled kernels to disk, avoiding recompilation on subsequent runs.

These steps transform a naïve $\mathcal{O}(N^2)$ Python script into an efficient multi-core simulator capable of handling thousands of particles in real time.

## Beyond Numba: Strategies for Scaling Up Performance

After (or alongside) Numba, here are the **main avenues** people reach for when they still need more raw throughput or larger‐scale parallelism. Each tackles a different performance ceiling, and the smartest strategy often combines several:

| **Gap After Numba** | **What to Add** | **Why It Helps** | **Typical Speed-ups** |
|---------------------|------------------|------------------|------------------------|
| CPU is saturated, calculation is embarrassingly parallel | **Dask** (`dask.delayed`, `dask.distributed`) | Orchestrates batches of Numba-compiled tasks across all cores of a workstation or cluster with minimal code changes | Linear scaling up to dozens–hundreds of cores for independent trajectories ([dask.org](https://dask.org)) |
| Need to run on many nodes / HPC super-computer | **mpi4py** | Gives explicit MPI (MPICH/OpenMPI) in Python—Numba kernels run in each rank, while MPI handles domain decomposition or tree-exchange | $1000\times$ scale-outs in astrophysical N-body ([ADMIN Magazine](https://www.admin-magazine.com)) |
| Memory bandwidth or pairwise loops dominate | **Algorithmic change**: Barnes–Hut / Fast Multipole | Reduces force stage from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$ by grouping far-away particles into multipole expansions | Fewer interactions trump raw FLOPS ([Wikipedia](https://en.wikipedia.org/wiki/Barnes–Hut_simulation)) |
| GPU available, array math heavy | **CuPy** or **Numba-CUDA kernels** | $3\times$–$50\times$ on large (>10 kB) arrays via thousands of GPU cores. CuPy mirrors the NumPy API and plays well with Numba-CUDA  |
| Need custom kernels on GPU but want Python | **PyCUDA / PyOpenCL** | Write the kernel in CUDA/OpenCL C and launch from Python. Useful when Numba-CUDA can't express exotic memory patterns  |

---

## Putting It Together: A Concrete Recipe for an Astrophysical N-Body Simulation

### Tree Algorithm

Build a Barnes–Hut octree in Python, but put the heavy loops (`build_tree`, `compute_multipoles`, `traverse`) under `@njit(parallel=True)`:

- Drops complexity to $\mathcal{O}(N \log N)$
- Typically yields $10\times$–$100\times$ wall-clock speedup over naïve $\mathcal{O}(N^2)$

###  Two-Level Parallelism

**Inside each MPI rank**:

```python
# Numba-parallel Barnes–Hut force loop
forces = compute_force_tree_bh(pos, mass, tree)
```
Across ranks:

```python
comm.Allreduce(MPI.IN_PLACE, forces, op=MPI.SUM)
```

With a few dozen nodes you can simulate millions of stars in real time.

## Off-load big dense array math to GPU

```python
import cupy as cp
pos_gpu  = cp.asarray(pos)    # zero-copy on NVLink systems
vel_gpu += cp.asarray(forces) * dt / mass_gpu
```

Nuances: transfer once, keep arrays resident; use CuPy’s RawKernel if you need bespoke integrators.

## Task-orchestrate with Dask

```python
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=32, threads_per_worker=1)
client  = Client(cluster)

futures = client.map(run_single_realization, seeds)
results = client.gather(futures)
```

Each run_single_realization is itself a Numba-accelerated trajectory; Dask multiplexes thousands of them.

## Profile, then vectorize

Before adding yet another tool, spend an hour with

```python
python -m line_profiler script.py.lprof
```

and nvprof (GPU) to be sure the next bottleneck is real.

## Rule of thumb
First squeeze everything you can out of algorithmic scaling and thread-level speed (Numba).
Second decide whether your workload is data-parallel (Dask/MPI) or compute-dense (GPU).
Third mix both when the science demands and your hardware budget allows it.

Do that and you can drive modern N-body or molecular problems to billions of particles per step, while still writing mostly Python.




