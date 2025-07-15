# Zero‑Field Charge‑Carrier Mobility in Disordered Organic Semiconductors  
## Marcus‑Rate Graphs & Exact Kinetic Monte Carlo (n‑Fold Way)

A step‑by‑step guide to building a graph of hopping sites, generating Marcus rates, and extracting the intrinsic mobility $\mu_0$ with a fully rejection‑free kinetic Monte‑Carlo engine.

---

## 1 Why “zero‑field” mobility?

Organic semiconductors often display a pronounced field dependence, but every analytical model (Poole‑Frenkel, SCLC, ToF) reduces to the linear‑response mobility $\mu_0$ as $F \rightarrow 0$.  
$\mu_0$ provides the material parameter that ab‑initio and multiscale workflows must reproduce before tackling high‑field regimes.  
*AIP Publishing*

---

## 2 Assumptions & validity domain

| #  | Assumption                              | Consequence                                                                 |
|----|------------------------------------------|------------------------------------------------------------------------------|
| 1  | Non‑degenerate carrier density (Fermi level in DOS tail) | Classical Einstein relation holds to within ~3 % for $\sigma \lesssim 0.12$ eV at 300 K. |
| 2  | Static energetic & positional disorder   | All $E_i$, $J_{ij}$ frozen during the walk.                                 |
| 3  | Marcus non‑adiabatic hopping             | Slow nuclei, weak electronic coupling.                                      |
| 4  | Zero external field                      | Rates satisfy detailed balance; walk is unbiased.                           |


---

## 3 Graph construction

### 3.1 Nodes & edges

- Node i → localized molecule/segment  
  Store: position $\mathbf{r}_i$ and energy $E_i$

- Edge (i,j) → possible hop  
  Store: rate $k_{ij}$

---

### 3.2 Marcus rate

The hopping rate from site $i$ to site $j$ is given by:

$$
k_{ij} = \frac{2\pi}{\hbar} |J_{ij}|^2 \left(4\pi\lambda k_B T\right)^{-1/2} \exp\left[ -\frac{(\Delta G_{ij} + \lambda)^2}{4\lambda k_B T} \right]
$$

where:

$$
\Delta G_{ij} = E_j - E_i
$$

This satisfies detailed balance:

$$
k_{ji} = k_{ij} \, e^{-\beta \Delta G_{ij}}
$$


```python
for (i,j), J in couplings.items():
    dG = E[j]-E[i]
    kij = (2*np.pi/ħ)*J**2/np.sqrt(4*np.pi*λ*kB*T) * \
          np.exp(-(dG+λ)**2/(4*λ*kB*T))
    kji = kij*np.exp(-β*dG)
    G.add_edge(i,j, k_f=kij, k_b=kji)
```

## 4 Exact Kinetic Monte Carlo: The n‑Fold Way

The **n‑fold way** (Bortz–Kalos–Lebowitz) is a **rejection‑free sampler** for any continuous‑time Markov chain.  
[CMSR Rutgers](https://courses.physics.illinois.edu/phys466/fa2016/lnotes/KMC.pdf)  
[https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo]

---

### 4.1 Mathematics

At site $i$:

- **Total escape rate**
- 
$$
R_i = \sum_j k_{ij}
$$

- **Waiting time**  

$$
\Delta t \sim \text{Exp}(R_i)
$$

- **Hop selection probability**  

$$
\Pr(i \rightarrow j) = \frac{k_{ij}}{R_i}
$$

---

### 4.2 Algorithm

Draw $u_1, u_2 \sim \mathcal{U}(0, 1)$

- **Time step**

$$
\Delta t = \frac{-\ln u_1}{R_i}
$$

- **Hop selection**  
  Choose hop $j$ such that:

$$
\sum_{m \le j} k_{im} \ge u_2 R_i
$$

- **Update system state**
  
$$
t \leftarrow t + \Delta t, \quad i \leftarrow j
$$

Because every random number yields a real hop, the trajectory exactly reproduces the Master equation.

```python
t, pos = 0.0, start_node
while t < t_max:
    nbrs   = list(G.neighbors(pos))
    rates  = [G[pos][j]['k_f'] for j in nbrs]
    R      = sum(rates)
    u1,u2  = random.random(), random.random()
    dt     = -math.log(u1)/R
    t     += dt
    thresh = u2*R; c=0.0
    for j,kij in zip(nbrs,rates):
        c += kij
        if c>=thresh:
            pos=j; break
    record(t, pos)
```

## 5 Extracting $\mu_0$ from the Trajectory

For an unbiased walk, the mean‑square displacement is linear at long times:

$$
D = \lim_{t \to \infty} \frac{\langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle}{2 d t}
$$

Under assumptions 1–4, the classical Einstein relation is valid:

$$
\mu_0 = \frac{e D}{k_B T} \tag{1}
$$

Check linearity (log‑log slope $\rightarrow 1$) over at least one decade in $t$ to ensure diffusion has set in.

```python
msd  = np.mean((traj_pos - traj_pos[0])**2, axis=1)
D    = (msd[-1]-msd[-2]) / (2*dim*(t[-1]-t[-2]))
mu_0 = e*D/(kB*T)
```

