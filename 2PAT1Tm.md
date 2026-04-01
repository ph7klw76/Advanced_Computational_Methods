# Deriving the Cycle-Averaged Fraction of Triplet Population in $T_m$

When a molecule is repeatedly excited by a high-repetition-rate pulsed laser, the lowest triplet state $T_1$ can be re-excited into a higher triplet manifold, which I will call $T_m$ (with $m \ge 2$). In many practical cases, the quantity of interest is not the instantaneous population right after a pulse, but the **cycle-averaged fraction of the triplet population that resides in $T_m$**.

This post derives that result carefully, states the assumptions, and defines every symbol and unit.

---

## 1. What we want to calculate

We want the **cycle-averaged quasi-steady fraction of triplet population in $T_m$**:

$$
f_{T_m} = \frac{t_m}{t_1 + t_m}
$$

where:

- $t_1$ is the population fraction in $T_1$
- $t_m$ is the population fraction in the higher triplet manifold $T_m$

The companion quantity is:

$$
f_{T_1} = \frac{t_1}{t_1 + t_m} = 1 - f_{T_m}
$$

The final result will be:

$$
f_{T_m} = \frac{k_{1m}}{k_{1m} + k_m^{\mathrm{tot}}}
$$

where:

- $k_{1m}$ is the **cycle-averaged pumping rate** from $T_1$ to $T_m$
- $k_m^{\mathrm{tot}}$ is the **total depopulation rate** out of $T_m$

---

## 2. Physical picture

The minimal kinetic picture is:

$$
S_0 \xrightarrow{2\mathrm{PA}} S_1 \xrightarrow{\mathrm{ISC}} T_1 \xrightarrow{h\nu} T_m
$$

Then the higher triplet manifold $T_m$ can relax by several routes, for example:

- back to $T_1$
- to the ground state
- into a chemical loss channel such as radical generation

We do **not** need to know all microscopic details of the $T_m$ manifold to derive the cycle-averaged fraction. We only need its **total exit rate**.

---

## 3. Symbols and units

- $E_p$: pulse energy, in J
- $f_{\mathrm{rep}}$: laser repetition rate, in s$^{-1}$
- $\tau_p$: pulse width (FWHM for a Gaussian pulse), in s
- $\lambda$: laser wavelength, in m or cm
- $A_{\mathrm{eff}}$: effective illuminated area, in m$^2$ or cm$^2$
- $h$: Planck constant, in J$\cdot$s
- $c$: speed of light, in m/s
- $\nu$: optical frequency, in s$^{-1}$
- $\mathcal{F}_\gamma$: photon fluence per pulse, in photons/m$^2$ or photons/cm$^2$
- $\sigma_{1m}$: absorption cross section for $T_1 \to T_m$, in m$^2$ or cm$^2$
- $q_{1m}$: per-pulse probability of $T_1 \to T_m$, dimensionless
- $k_{1m}$: cycle-averaged rate of $T_1 \to T_m$ pumping, in s$^{-1}$
- $\tau_m$: effective residence time in $T_m$, in s
- $k_m^{\mathrm{tot}}$: total depopulation rate out of $T_m$, in s$^{-1}$
- $t_1$: population fraction in $T_1$, dimensionless
- $t_m$: population fraction in $T_m$, dimensionless
- $f_{T_1}$: triplet-manifold fraction in $T_1$, dimensionless
- $f_{T_m}$: triplet-manifold fraction in $T_m$, dimensionless

---

## 4. Assumptions of the model

This derivation uses a **cycle-averaged coarse-grained kinetic model**. The main assumptions are:

1. **The pulse train is replaced by an average rate.**  
   We do not resolve what happens during each individual pulse.

2. **$T_m$ is treated as one lumped manifold.**  
   All higher triplet states are grouped into a single effective state $T_m$.

3. **$T_1 \to T_m$ is modeled as a first-order pumping process.**  
   The effective rate is $k_{1m}$.

4. **The $T_m$ manifold has one effective total exit rate.**  
   That rate is $k_m^{\mathrm{tot}}$.

5. **Cross sections and rates are constant during the observation window.**

6. **We are calculating a quasi-steady triplet partition.**  
   If there is irreversible chemistry, then the full system does not have a nontrivial true steady state forever. The result here is the steady ratio **within the active triplet subspace**.

This last point matters. If molecules are permanently consumed, then ultimately all population can leak to products. But the **ratio** between $T_1$ and $T_m$ can still reach a well-defined quasi-steady value on a shorter timescale.

---

## 5. Step 1: photon fluence per pulse

The energy per photon is:

$$
h\nu = \frac{hc}{\lambda}
$$

So the number of photons in one pulse is:

$$
N_\gamma = \frac{E_p}{h\nu}
$$

If those photons are distributed over an effective area $A_{\mathrm{eff}}$, then the **photon fluence per pulse** is:

$$
\mathcal{F}_\gamma
= \frac{N_\gamma}{A_{\mathrm{eff}}}
= \frac{E_p}{h\nu \, A_{\mathrm{eff}}}
= \frac{E_p \lambda}{h c A_{\mathrm{eff}}}
$$

### Unit check

Using SI units:

- $E_p$ in J
- $\lambda$ in m
- $h$ in Js
- $c$ in m/s
- $A_{\mathrm{eff}}$ in m$^2$

Then:

$$
\frac{E_p \lambda}{h c A_{\mathrm{eff}}} \sim \frac{\mathrm{J} \cdot \mathrm{m}}{(\mathrm{J \cdot s})(\mathrm{m/s})(\mathrm{m}^2)} = \frac{1}{\mathrm{m}^2}
$$

which is photons per area, as required.

---

## 6. Step 2: per-pulse probability of $T_1 \to T_m$

Let $\sigma_{1m}$ be the absorption cross section for promotion from $T_1$ to $T_m$ at the laser wavelength.

The per-pulse probability that a molecule already in $T_1$ absorbs one photon and is promoted to $T_m$ is:

$$
q_{1m} = 1 - \exp\!\left(-\sigma_{1m}\mathcal{F}_\gamma\right)
$$

This is the standard absorption probability for a fluence $\mathcal{F}_\gamma$.

### Low-fluence limit

If $\sigma_{1m}\mathcal{F}_\gamma \ll 1$, then:

$$
q_{1m} \approx \sigma_{1m}\mathcal{F}_\gamma
$$

This approximation is often valid and is useful for intuition.

### Unit check

Since $\sigma_{1m}$ has units of area and $\mathcal{F}_\gamma$ has units of photons per area, the product $\sigma_{1m}\mathcal{F}_\gamma$ is dimensionless, so the exponential is valid.

---

## 7. Step 3: convert per-pulse probability into a cycle-averaged rate

The laser delivers pulses at repetition rate $f_{\mathrm{rep}}$, so the **cycle-averaged pumping rate** from $T_1$ to $T_m$ is:

$$
k_{1m} = f_{\mathrm{rep}} q_{1m}
$$

Substituting the exact per-pulse probability:

$$
k_{1m} = f_{\mathrm{rep}} \left[1 - \exp\!\left(-\sigma_{1m}\mathcal{F}_\gamma\right)\right]
$$

and substituting the photon fluence expression:

$$
k_{1m} = f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\frac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]
$$

### Low-fluence form

If $\sigma_{1m}\mathcal{F}_\gamma \ll 1$:

$$
k_{1m}\approx f_{\mathrm{rep}}\sigma_{1m}\mathcal{F}_\gamma=f_{\mathrm{rep}}\sigma_{1m}\frac{E_p \lambda}{h c A_{\mathrm{eff}}}
$$

### Unit check

- $f_{\mathrm{rep}}$ has units s$^{-1}$
- $q_{1m}$ is dimensionless

Therefore:

$$
k_{1m} \text{ has units s}^{-1}
$$

---

## 8. Step 4: define the total depopulation rate of $T_m$

The higher triplet manifold can decay through several channels. Write:

$$
k_m^{\mathrm{tot}} = k_{m1} + k_{m0} + k_{mR}
$$

where, for example:

- $k_{m1}$ = relaxation from $T_m$ back to $T_1$
- $k_{m0}$ = relaxation from $T_m$ to the ground state
- $k_{mR}$ = irreversible loss from $T_m$ into radical chemistry or another product channel

Sometimes it is convenient to define an **effective residence time** in $T_m$:

$$
\tau_m = \frac{1}{k_m^{\mathrm{tot}}}
$$

or equivalently:

$$
k_m^{\mathrm{tot}} = \frac{1}{\tau_m}
$$

---

## 9. Step 5: write the cycle-averaged kinetic equations

Now focus only on the triplet subspace. Let:

- $t_1(t)$ = fraction in $T_1$
- $t_m(t)$ = fraction in $T_m$

A minimal cycle-averaged kinetic model is:

$$
\frac{d t_1}{d t} =J-(k_{10} + k_{1m}) t_1+k_{m1} t_m
$$

$$
\frac{d t_m}{d t}=k_{1m} t_1-k_m^{\mathrm{tot}} t_m
$$

Here:

- $J$ is the source term feeding $T_1$ from the singlet channel
- $k_{10}$ is the effective loss rate out of $T_1$
- $k_{1m}$ pumps population from $T_1$ into $T_m$
- $k_m^{\mathrm{tot}}$ removes population from $T_m$

### Important point

To determine the **partition between $T_1$ and $T_m$**, we only need the second equation at quasi-steady state.

---

## 10. Step 6: derive the quasi-steady ratio $t_m/t_1$

Set the $T_m$ equation to quasi-steady state:

$$
\frac{d t_m}{d t} = 0
$$

Then:

$$
0 = k_{1m} t_1 - k_m^{\mathrm{tot}} t_m
$$

Rearranging:

$$
k_{1m} t_1 = k_m^{\mathrm{tot}} t_m
$$

so the ratio is:

$$
\frac{t_m}{t_1} = \frac{k_{1m}}{k_m^{\mathrm{tot}}}
$$

This is the key result.

If $k_m^{\mathrm{tot}} = 1/\tau_m$, then:

$$
\frac{t_m}{t_1} = k_{1m}\tau_m
$$

and substituting the cycle-averaged pumping rate gives:

$$
\frac{t_m}{t_1} = f_{\mathrm{rep}} q_{1m} \tau_m
$$

Using the exact $q_{1m}$:

$$
\frac{t_m}{t_1}=f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\frac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]\tau_m
$$

### Low-fluence form

If $\sigma_{1m}\mathcal{F}_\gamma \ll 1$:

$$
\frac{t_m}{t_1} \approx f_{\mathrm{rep}}\sigma_{1m}\mathcal{F}_\gamma\tau_m =f_{\mathrm{rep}}\sigma_{1m}\frac{E_p \lambda}{h c A_{\mathrm{eff}}}\tau_m
$$

---

## 11. Step 7: derive the cycle-averaged fraction in $T_m$

By definition:

$$
f_{T_m} = \frac{t_m}{t_1 + t_m}
$$

Use the ratio:

$$
\frac{t_m}{t_1} = \frac{k_{1m}}{k_m^{\mathrm{tot}}}
$$

Then:

$$
f_{T_m}= \frac{t_m/t_1}{1 + t_m/t_1}=\frac{k_{1m}/k_m^{\mathrm{tot}}}{1 + k_{1m}/k_m^{\mathrm{tot}}}
$$

which simplifies to:

$$
f_{T_m} = \frac{k_{1m}}{k_{1m} + k_m^{\mathrm{tot}}}
$$

Similarly:

$$
f_{T_1} = \frac{k_m^{\mathrm{tot}}}{k_{1m} + k_m^{\mathrm{tot}}} = 1 - f_{T_m}
$$

These are the desired cycle-averaged triplet-manifold fractions.

---

## 12. Final explicit formula in laser parameters

Substitute:

$$
k_{1m}=f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\frac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]
$$

and

$$
k_m^{\mathrm{tot}} = \frac{1}{\tau_m}
$$

Then:

$$
f_{T_m}=\frac{f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\dfrac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]}{f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\dfrac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]+ 1/\tau_m}
$$

and:

$$
f_{T_1} = 1 - f_{T_m}
$$

### Low-fluence approximation

If $\sigma_{1m} E_p \lambda / (h c A_{\mathrm{eff}}) \ll 1$, then:

$$
f_{T_m}\approx\frac{f_{\mathrm{rep}}\sigma_{1m}\dfrac{E_p \lambda}{h c A_{\mathrm{eff}}}}{f_{\mathrm{rep}}\sigma_{1m}\dfrac{E_p \lambda}{h c A_{\mathrm{eff}}}+ 1/\tau_m}
$$

---

## 13. A useful dimensionless control parameter

Define:

$$
\Lambda = \frac{k_{1m}}{k_m^{\mathrm{tot}}} = k_{1m}\tau_m
$$

Then:

$$
\frac{t_m}{t_1} = \Lambda
$$

and the fractions become:

$$
f_{T_m} = \frac{\Lambda}{1 + \Lambda}
$$

$$
f_{T_1} = \frac{1}{1 + \Lambda}
$$

This is often the cleanest way to interpret the physics.

### Interpretation

- If $\Lambda \ll 1$, then $T_m$ is only weakly populated and most triplet population remains in $T_1$.
- If $\Lambda \gg 1$, then $T_m$ dominates the triplet manifold.
- If $\Lambda \approx 1$, then $T_1$ and $T_m$ carry comparable fractions.

---

## 14. What this result means physically

The fraction in $T_m$ is controlled by a competition between:

1. **How fast the laser pumps $T_1 \to T_m$**
   - larger $f_{\mathrm{rep}}$
   - larger $\sigma_{1m}$
   - larger $E_p$
   - smaller $A_{\mathrm{eff}}$

2. **How fast $T_m$ empties**
   - shorter $\tau_m$ means faster depopulation and smaller $f_{T_m}$
   - longer $\tau_m$ means slower depopulation and larger $f_{T_m}$

So the cycle-averaged $T_m$ fraction is large when re-excitation is fast and $T_m$ relaxation is slow.

---

## 15. What this formula is *not*

This formula is **not** the same as:

- the instantaneous post-pulse fraction right after one pulse
- the exact time-resolved fraction inside each interpulse period
- the global steady state of the full chemical system if irreversible product formation continues forever

It is specifically the:

> **quasi-steady, cycle-averaged fraction of the triplet population residing in $T_m$ within the averaged-rate model**

That distinction is important.

---

## 16. Summary

Starting from the per-pulse excitation probability for $T_1 \to T_m$:

$$
q_{1m} = 1 - \exp\!\left(-\sigma_{1m}\mathcal{F}_\gamma\right)
$$

with:

$$
\mathcal{F}_\gamma = \frac{E_p \lambda}{h c A_{\mathrm{eff}}}
$$

the cycle-averaged pumping rate is:

$$
k_{1m} = f_{\mathrm{rep}} q_{1m}
$$

At quasi-steady state in the triplet manifold:

$$
\frac{t_m}{t_1} = \frac{k_{1m}}{k_m^{\mathrm{tot}}}
$$

Therefore the cycle-averaged fraction of triplet population in $T_m$ is:

$$
f_{T_m} = \frac{k_{1m}}{k_{1m} + k_m^{\mathrm{tot}}}
$$

or explicitly:

$$
f_{T_m}=\frac{f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\dfrac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]}{f_{\mathrm{rep}}\left[1 - \exp\!\left(-\sigma_{1m}\dfrac{E_p \lambda}{h c A_{\mathrm{eff}}}\right)\right]+ 1/\tau_m}
$$

with:

$$
f_{T_1} = 1 - f_{T_m}
$$

---

## 17. Optional notation block for code or simulation

For implementation, these are the minimum required inputs:

```text
Ep       = pulse energy [J]
frep     = repetition rate [s^-1]
lambda   = wavelength [m]
Aeff     = effective area [m^2]
sigma1m  = T1->Tm cross section [m^2]
taum     = effective Tm lifetime [s]
h        = 6.62607015e-34 [J s]
c        = 2.99792458e8 [m s^-1]
```

Then compute:

```text
Fgamma = Ep*lambda/(h*c*Aeff)
q1m    = 1 - exp(-sigma1m*Fgamma)
k1m    = frep*q1m
fTm    = k1m/(k1m + 1/taum)
fT1    = 1 - fTm
```

---

## 18. Closing remark

This derivation is compact, but the logic is simple:

**$T_m$ wins when optical pumping into $T_m$ is faster than relaxation out of $T_m$. But in this case $T_1$ can access $T_m$ for bond breakage**

The cycle-averaged fraction is just the normalized competition between those two rates.
