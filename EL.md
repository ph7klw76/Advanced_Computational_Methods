# Coupled Rate Equations for Charge and Exciton Dynamics in OLEDs

Below is a comprehensive set of generic coupled rate equations that describe charge and exciton dynamics in an organic light-emitting diode (OLED). This framework aims to capture:

- **Transport and recombination** of electrons ($n$) and holes ($p$) within the device.
- **Formation and decay** of singlet excitons ($S$) responsible for light emission.
- **Device-level effects** such as capacitance, electric field, and boundary conditions (including turning off the external injection).
- **Emission** (photons) from radiative decay of excitons.

These equations are suitable for transient modeling (in particular, the emission decay after charge injection is stopped) and can be extended or simplified depending on specific material systems and device structures. They form the basis of more specialized drift-diffusion-exciton (DDE) or semiconductor-device models adapted to OLEDs.

---

## 1. Governing Variables and Parameters

We consider a one-dimensional representation along the thickness ($x$) of the device. The device extends from $x=0$ (anode) to $x=d$ (cathode). The main variables are:

- $n(x, t)$: Electron density ($\text{m}^{-3}$),
- $p(x, t)$: Hole density ($\text{m}^{-3}$),
- $S(x, t)$: Singlet exciton density ($\text{m}^{-3}$),
- $\phi(x, t)$: Electrostatic potential (V),
- $E(x, t) = -\frac{d\phi}{dx}$: Electric field (V/m).

We also define:

- $\mu_n$, $\mu_p$: Electron and hole mobilities ($\text{m}^2/\text{V·s}$),
- $D_n$, $D_p$: Electron and hole diffusion coefficients ($\text{m}^2/\text{s}$),
- $\epsilon$: Dielectric permittivity of the organic layer (F/m),
- $q$: Elementary charge ($1.602 \times 10^{-19} \, \text{C}$),
- $R$: Recombination rate of electrons and holes into excitons ($\text{m}^{-3}/\text{s}$),
- $G$: Exciton generation rate (if relevant from optical excitation; in an OLED, we often have mostly electrically injected carriers forming excitons at recombination events),
- $\Gamma$: Net exciton quenching/decay (including radiative, non-radiative, and possibly intersystem crossing, TTA, etc.).

---

## 2. Poisson’s Equation (Electric Field / Potential)

The electric field inside the OLED layer is governed by Poisson’s equation:

$$
-\frac{d^2 \phi(x, t)}{dx^2} = \frac{q}{\epsilon} \, [p(x, t) - n(x, t) + N_\text{dop}(x)],
$$

where $N_\text{dop}(x)$ represents any (fixed) doping or trapped charges. Doping may be small or absent in many OLEDs, but it is included here for completeness. Numerically, one usually solves for $\phi(x, t)$ subject to electrode boundary conditions (i.e., specified voltages or zero current conditions).

---

## 3. Charge Transport (Continuity Equations)

### 3.1 Electron Continuity

$$
\frac{\partial n}{\partial t} = \frac{1}{q} \frac{\partial J_n}{\partial x} - R_\text{tot}(n, p) + \dots
$$

- $J_n$: Electron current density ($\text{A}/\text{m}^2$),
- $R_\text{tot}(n, p)$: Net recombination rate (electrons + holes $\rightarrow$ exciton or ground state).

For drift-diffusion models:

$$
J_n = q \, (n \mu_n E + D_n \frac{\partial n}{\partial x}),
$$

so:

$$
\frac{1}{q} \frac{\partial J_n}{\partial x} = \frac{\partial}{\partial x} \left( n \mu_n E + D_n \frac{\partial n}{\partial x} \right).
$$

---

### 3.2 Hole Continuity

$$
\frac{\partial p}{\partial t} = -\frac{1}{q} \frac{\partial J_p}{\partial x} - R_\text{tot}(n, p) + \dots
$$

- $J_p$: Hole current density ($\text{A}/\text{m}^2$).

For drift-diffusion models:

$$
J_p = q \, \left( -p \mu_p E - D_p \frac{\partial p}{\partial x} \right),
$$

so:

$$
-\frac{1}{q} \frac{\partial J_p}{\partial x} = -\frac{\partial}{\partial x} \left( p \mu_p E + D_p \frac{\partial p}{\partial x} \right).
$$

---

## 4. Exciton Formation and Dynamics

Singlet excitons ($S$) are primarily formed via bimolecular recombination of electrons and holes in an OLED. For electrically driven devices, the main source is $e$–$h$ recombination. The exciton dynamics can be written as:

$$
\frac{\partial S}{\partial t} = R_\text{gen}(n, p) - \Gamma_\text{loss}(S) + \dots
$$

---

### 4.1 Exciton Generation by Recombination

For bimolecular recombination:

$$
R_\text{gen}(n, p) = \gamma (np - n_i^2),
$$

- $\gamma$: Recombination coefficient ($\text{m}^3/\text{s}$),
- $n_i$: Intrinsic carrier concentration (often negligible in wide-gap organic semiconductors).

The fraction of recombinations forming excitons can be separated:

$$
R_\text{tot}(n, p) = \zeta \gamma np, \quad R_\text{gen}(n, p) = (1 - \zeta) \gamma np,
$$

- $\zeta$: Fraction of $e$–$h$ recombinations not forming excitons (loss channels).

Including spin statistics, the fraction of excitons forming singlets is:

$$
R_\text{gen}(S)(n, p) = \eta_\text{formation} \eta_\text{singlet} \gamma np,
$$

- $\eta_\text{singlet} \approx 0.25$ under simple spin-statistical assumptions.

---

### 4.2 Exciton Decay and Loss

The net singlet loss rate $\Gamma_\text{loss}(S)$ includes:

- Radiative decay ($k_r(S)$),
- Non-radiative decay ($k_{nr}(S)$),
- Quenching by polarons or annihilation processes.

For simplicity:

$$
\Gamma_\text{loss}(S) = (k_r(S) + k_{nr}(S)) S + k_q(n) n S + k_q(p) p S + k_{SS} S^2 + \dots
$$

Photon emission rate density:

$$
\Phi_\text{em}(x, t) = k_r(S) S(x, t).
$$

---

## 5. Complete Charge-Exciton-Recombination Model

### Summary of Equations:

1. **Poisson’s Equation**:
   
$$
-\frac{d^2 \phi}{dx^2} = \frac{q}{\epsilon} \, [p - n + N_\text{dop}(x)].
$$

3. **Electron Continuity**:
   
$$
\frac{\partial n}{\partial t} = \frac{\partial}{\partial x} \left( n \mu_n E + D_n \frac{\partial n}{\partial x} \right) - \zeta \gamma np + \dots
$$

5. **Hole Continuity**:

$$
\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x} \left( p \mu_p E + D_p \frac{\partial p}{\partial x} \right) - \zeta \gamma np + \dots
$$

7. **Exciton Dynamics**:

$$
\frac{\partial S}{\partial t} = (1 - \zeta) \eta_\text{singlet} \gamma np - (k_r(S) + k_{nr}(S)) S - k_q(n) n S - k_q(p) p S - \dots
$$

These equations provide the foundation for modeling charge transport and exciton dynamics in OLEDs.

# Governing Variables and Parameters

We consider a one-dimensional representation along the thickness ($x$) of the device. The device extends from $x=0$ (anode) to $x=d$ (cathode). The main variables are:

- $n(x,t)$: Electron density ($\text{m}^{-3}$),
- $p(x,t)$: Hole density ($\text{m}^{-3}$),
- $S(x,t)$: Singlet exciton density ($\text{m}^{-3}$),
- $\phi(x,t)$: Electrostatic potential (V),
- $E(x,t) = -\frac{d\phi}{dx}$: Electric field (V/m).

We also define:

- $\mu_n$, $\mu_p$: Electron and hole mobilities ($\text{m}^2/\text{V·s}$),
- $D_n$, $D_p$: Electron and hole diffusion coefficients ($\text{m}^2/\text{s}$),
- $\epsilon$: Dielectric permittivity of the organic layer (F/m),
- $q$: Elementary charge ($1.602 \times 10^{-19} \, \text{C}$),
- $R_\text{tot}$: Net rate of electron-hole recombination ($\text{m}^{-3}/\text{s}$),
- $R_\text{gen}(S)$: Rate of singlet exciton formation ($\text{m}^{-3}/\text{s}$),
- $\Gamma_\text{loss}$: Net exciton decay rate ($\text{m}^{-3}/\text{s}$).

Additionally, to incorporate device-level RC-effects and transient injection:

- $C_\text{dev}$: Geometric/device capacitance (F) or a distributed $\epsilon$-based treatment,
- $I(t)$ or $V_\text{ext}(t)$: External current/voltage supply as a function of time.

---

# Poisson’s Equation (Electric Field / Potential)

The electric field inside the organic layer is governed by Poisson’s equation:

$$
-\frac{d^2 \phi(x, t)}{dx^2} = \frac{q}{\epsilon} \, [p(x, t) - n(x, t) + N_\text{dop}(x)],
$$

where $N_\text{dop}(x)$ represents any (fixed) doping or trapped charges. In many OLEDs, doping may be small or absent, but we include it here for completeness. Numerically, one usually solves for $\phi(x, t)$ subject to electrode boundary conditions (i.e., specified voltages or zero-current conditions).

---

# Charge Transport (Continuity Equations)

## 3.1 Electron Continuity

$$
\frac{\partial n}{\partial t} = \frac{1}{q} \frac{\partial J_n}{\partial x} - R_\text{tot}(n, p) + \dots \quad (1)
$$

- $J_n$: Electron current density ($\text{A}/\text{m}^2$),
- $R_\text{tot}(n, p)$: Net recombination rate of electrons and holes.

In the drift-diffusion approximation:

$$
J_n = q (\mu_n n E + D_n \frac{\partial n}{\partial x}),
$$

so:

$$
\frac{1}{q} \frac{\partial J_n}{\partial x} = \frac{\partial}{\partial x} \left( \mu_n n E + D_n \frac{\partial n}{\partial x} \right).
$$

The term $\dots$ can include additional processes such as:

- Trapping/de-trapping (trap-limited transport),
- Polaron-exciton annihilation (if one chooses to write it in the electron continuity),
- Interfacial injection or extraction terms (if needed explicitly inside the bulk equations).

---

## 3.2 Hole Continuity

$$
\frac{\partial p}{\partial t} = -\frac{1}{q} \frac{\partial J_p}{\partial x} - R_\text{tot}(n, p) + \dots \quad (2)
$$

- $J_p$: Hole current density ($\text{A}/\text{m}^2$).

Similarly:

$$
J_p = q \left( -\mu_p p E - D_p \frac{\partial p}{\partial x} \right),
$$

so:

$$
-\frac{1}{q} \frac{\partial J_p}{\partial x} = -\frac{\partial}{\partial x} \left( \mu_p p E + D_p \frac{\partial p}{\partial x} \right).
$$

The negative sign arises from the standard convention in semiconductor device equations.

---

# Exciton Formation and Dynamics

Singlet excitons ($S$) are formed primarily via electron-hole recombination in an OLED. They can also be formed by optical absorption, but for electrically driven devices, the main source is $n + p$ recombination in the emissive layer.

$$
\frac{\partial S}{\partial t} = R_\text{gen}(S)(n, p) - \Gamma_\text{loss}(S, n, p) + \dots \quad (3)
$$

---

## 4.1 Exciton Generation

A common approach is to write the total $e$–$h$ recombination rate as:

$$
R_\text{tot}(n, p) = \gamma n p \quad (\text{basic Langevin/Bimolecular form}).
$$

Then, a fraction $\eta_\text{exc}$ of these recombinations leads to excitons:

$$
R_\text{gen}(S)(n, p) = \eta_\text{exc} \gamma n p \quad \Rightarrow \quad (\text{rate of new singlets}).
$$

Further, one can incorporate spin statistics: if only 25% of excitons are singlets in a purely fluorescent material, then:

$$
R_\text{gen}(S)(n, p) = \eta_\text{exc} \eta_\text{singlet} \gamma n p \quad \text{with} \quad \eta_\text{singlet} \approx 0.25.
$$

---

## 4.2 Exciton Decay and Loss

Once formed, excitons can:

- **Radiatively decay**: $k_r(S) S$,
- **Non-radiatively decay**: $k_{nr}(S) S$,
- **Undergo quenching** by polarons or impurities:
  - $k_q(n) n S$ (electron–exciton quenching),
  - $k_q(p) p S$ (hole–exciton quenching),
- **Exciton-Exciton Annihilation** (e.g., $k_{SS} S^2$ at high excitation densities),
- **Intersystem Crossing** to triplets (not shown explicitly here but can be added).

Hence, we write a generic exciton loss term:

$$
\Gamma_\text{loss}(S, n, p) = (k_r(S) + k_{nr}(S)) S + k_q(n) n S + k_q(p) p S + k_{SS} S^2 + \dots
$$

So, the exciton continuity (3) becomes:

$$
\frac{\partial S}{\partial t} = \eta_\text{exc} \gamma n p - (k_r(S) + k_{nr}(S)) S - k_q(n) n S - k_q(p) p S - k_{SS} S^2 - \dots
$$

---

# 5. Incorporating Device Capacitance and Transient Injection

When external injection (current or voltage) is suddenly stopped, the device does not instantaneously lose all charge because:

- There is a capacitance associated with the organic stack and electrodes.
- The carriers that accumulated in the device can continue to recombine, forming excitons and emitting light even after external bias is removed.

## 5.1 Effective Circuit Model

A simplified approach is to treat the device as a parallel plate capacitor of capacitance:

$$
C_\text{dev} \approx \frac{\epsilon A}{d},
$$

where:
- $A$ is the device area, and
- $d$ is the thickness.

The current through the external circuit is:

$$
I_\text{ext}(t) = C_\text{dev} \frac{dV_\text{dev}}{dt} + I_\text{rec}(t) + I_\text{transport}(t),
$$

where:
- $V_\text{dev}$ is the voltage across the device,
- $I_\text{rec}$ is the net recombination current (due to electron-hole recombination inside the device),
- $I_\text{transport}$ is the drift-diffusion current that flows to or from the contacts.

When $I_\text{ext}(t) \to 0$ (i.e., the injection is turned off or the device is switched to open-circuit), the device can still have residual charge that discharges or recombines over time. These circuit-level equations link to the continuity and Poisson equations in the bulk, with boundary conditions reflecting contact behavior (e.g., injection-limited, ohmic, or blocking).

---

# 6. Full Set of Coupled PDEs (Generic Form)

Combining everything, we arrive at a self-consistent system of PDEs:

### (A) Poisson's Equation:

$$
-\frac{d^2 \phi}{dx^2} = \frac{q}{\epsilon} \, [p - n + N_\text{dop}(x)],
$$

### (B) Electron Continuity:

$$
\frac{\partial n}{\partial t} = \frac{1}{q} \frac{\partial}{\partial x} \left( q \mu_n n E + q D_n \frac{\partial n}{\partial x} \right) - R_\text{tot}(n, p),
$$

### (C) Hole Continuity:

$$
\frac{\partial p}{\partial t} = -\frac{1}{q} \frac{\partial}{\partial x} \left( q \mu_p p E + q D_p \frac{\partial p}{\partial x} \right) - R_\text{tot}(n, p),
$$

### (D) Exciton Continuity:

$$
\frac{\partial S}{\partial t} = \eta_\text{exc} \gamma n p - \Gamma_\text{loss}(S, n, p).
$$

### With:
- $E = -\frac{d\phi}{dx}$,
- $R_\text{tot}(n, p) = \gamma n p$ (or a more complex formula),
- $\Gamma_\text{loss}(S, n, p) = (k_r(S) + k_{nr}(S)) S + k_q(n) n S + k_q(p) p S + k_{SS} S^2 + \dots$.

---

## 6.1 Boundary Conditions

Boundary conditions must be specified at $x=0$ (anode) and $x=d$ (cathode):

1. **Electrode Injection:**
   - If the device is biased, $\phi(0, t) = \phi_\text{anode}(t)$ and $\phi(d, t) = \phi_\text{cathode}(t)$.
   - If the device is disconnected, the net current at each contact is zero, implying Neumann boundary conditions on $n$ and $p$.

2. **Charge Densities at Contacts:**
   - May be set by thermionic emission or ohmic injection models,
   - Or by matching the quasi-Fermi levels to the metal work function.

When the external injection is turned off:

$$
I_\text{ext}(t > t_0) = 0,
$$

which enforces that no net current flows from the electrodes. The device potential evolves self-consistently based on the internal charge distribution and capacitance.

---

# 7. Modeling the Emission Decay After Injection Stops

### Turn-Off Condition:
- At $t = t_0$, set the external current (or voltage) to zero (open-circuit or short-circuit, depending on the experimental setup).

### Transient Discharge:
1. **Residual Charge:** Charges stored in the bulk and at interfaces start to recombine.
2. **Relaxation:** The electric field $\phi(x, t)$ and carrier densities $n(x, t)$, $p(x, t)$ will relax.
3. **Residual Exciton Formation:** While electrons and holes exist, recombination continues to form excitons, albeit at a diminishing rate.
4. **Capacitive Effect:** The potential might not drop immediately to $0$; it discharges over an RC timescale.
5. **Photon Emission:** The instantaneous light output at time $t$ can be estimated as:
   $$
   L(t) = \int_0^d k_r(S) S(x, t) F_\text{out}(x) \, dx,
   $$
   where $F_\text{out}(x)$ is an outcoupling factor or the fraction of photons escaping.

This emission decay is measured as transient electroluminescence.

---

# 8. Practical Considerations and Extensions

1. **Triplet States:** A complete OLED model might include triplet excitons ($T$) with rates for intersystem crossing, phosphorescence, TTA, etc.
2. **Trap States:** Trap distributions can affect carrier transport and recombination, necessitating trap continuity equations or effective rates.
3. **Inhomogeneous Layers:** Multilayer stacks (e.g., hole-injection, emission, electron-transport layers) require boundary matching conditions.
4. **Numerical Methods:** Solving these coupled PDEs typically requires finite-element or finite-difference methods with iterative solvers (e.g., Gummel, Newton-Raphson).

---

# Final Summary

This generic model captures the essential physics of:

- **Carrier injection and transport (electrons and holes),**
- **Exciton formation (from recombination) and subsequent emission,**
- **Device-level capacitive effects during transient discharge.**

By solving Poisson’s equation (A), continuity equations for electrons and holes (B, C), and the exciton rate equation (D) under time-dependent boundary conditions, one can:

1. Predict emission decay curves (electroluminescence vs. time),
2. Diagnose residual carrier and exciton behavior,
3. Optimize device architectures (layer thickness, mobility, doping, etc.) for performance.

This drift-diffusion-exciton model is foundational for OLED device simulation, providing insights into both transient and steady-state operations.


# 1. Governing Variables and Parameters

We consider a one-dimensional representation along the thickness ($x$) of the device. The device extends from $x=0$ (anode) to $x=d$ (cathode). The main variables are:

- $n(x,t)$: Electron density ($\text{m}^{-3}$)
- $p(x,t)$: Hole density ($\text{m}^{-3}$)
- $S(x,t)$: Singlet exciton density ($\text{m}^{-3}$)
- $\phi(x,t)$: Electrostatic potential (V)
- $E(x,t) = -\frac{d\phi}{dx}$: Electric field (V/m)

We also define:

- $\mu_n$, $\mu_p$: Electron and hole mobilities ($\text{m}^2/(\text{V·s})$)
- $D_n$, $D_p$: Electron and hole diffusion coefficients ($\text{m}^2/\text{s}$)
- $\epsilon$: Dielectric permittivity of the organic layer (F/m)
- $q$: Elementary charge ($1.602 \times 10^{-19} \, \text{C}$)
- $R_\text{tot}$: Net rate of electron-hole recombination ($\text{m}^{-3}/\text{s}$)
- $R_\text{gen}(S)$: Rate of singlet exciton formation ($\text{m}^{-3}/\text{s}$)
- $\Gamma_\text{loss}$: Net exciton decay rate ($\text{m}^{-3}/\text{s}$)

Additionally, to incorporate device-level RC-effects and transient injection:

- $C_\text{dev}$: Geometric/device capacitance (F) or a distributed $\epsilon$-based treatment.
- $I(t)$ or $V_\text{ext}(t)$: External current/voltage supply as a function of time.

---

# 2. Poisson’s Equation (Electric Field / Potential)

The electric field inside the organic layer is governed by Poisson’s equation:

$$
-\frac{d^2 \phi(x,t)}{dx^2} = \frac{q}{\epsilon} \, [p(x,t) - n(x,t) + N_\text{dop}(x)],
$$

where $N_\text{dop}(x)$ represents any (fixed) doping or trapped charges. In many OLEDs, doping may be small or absent, but we include it here for completeness. Numerically, one usually solves for $\phi(x,t)$ subject to electrode boundary conditions (i.e., specified voltages or zero-current conditions).

---

# 3. Charge Transport (Continuity Equations)

## 3.1 Electron Continuity


![image](https://github.com/user-attachments/assets/14c9abe1-81a7-4cdf-8fc1-a9969954124f)


- $J_n$: Electron current density ($\text{A}/\text{m}^2$)
- $R_\text{tot}(n,p)$: Net recombination rate of electrons and holes.

In the drift-diffusion approximation:

$$
J_n = q \, (\mu_n \, n \, E + D_n \frac{\partial n}{\partial x}),
$$

hence:

$$
\frac{1}{q} \frac{\partial J_n}{\partial x} = \frac{\partial}{\partial x} \left( \mu_n \, n \, E + D_n \frac{\partial n}{\partial x} \right).
$$

The term $\dots$ can include additional processes such as:
- Trapping/de-trapping (trap-limited transport),
- Polaron-exciton annihilation,
- Interfacial injection or extraction terms.

---

## 3.2 Hole Continuity


![image](https://github.com/user-attachments/assets/397379d9-29ad-4735-9c87-0ff4f9da8828)


- $J_p$: Hole current density ($\text{A}/\text{m}^2$).

Similarly:

$$
J_p = q \, (\mu_p \, p \, (-E) - D_p \frac{\partial p}{\partial x}) = q \, (-\mu_p \, p \, E - D_p \frac{\partial p}{\partial x}),
$$

hence:

$$
-\frac{1}{q} \frac{\partial J_p}{\partial x} = -\frac{\partial}{\partial x} \left( \mu_p \, p \, E + D_p \frac{\partial p}{\partial x} \right).
$$

---

# 4. Exciton Formation and Dynamics

Singlet excitons ($S$) are formed primarily via electron-hole recombination in an OLED. They can also be formed by optical absorption, but for electrically driven devices, the main source is $n+p$ recombination in the emissive layer.


![image](https://github.com/user-attachments/assets/6e19badc-cf65-41cf-ba18-a437aaa32377)

## 4.1 Exciton Generation

A common approach is to write the total $e$–$h$ recombination rate as:

$$
R_\text{tot}(n,p) = \gamma n p \quad (\text{basic Langevin/Bimolecular form}),
$$

where a fraction $\eta_\text{exc}$ of these recombinations leads to excitons:

$$
R_\text{gen}(S)(n,p) = \eta_\text{exc} \, \gamma n p \implies (\text{rate of new singlets}).
$$

Further, incorporating spin statistics, if only 25% of excitons are singlets in a purely fluorescent material:

$$
R_\text{gen}(S)(n,p) = \eta_\text{exc} \, \eta_\text{singlet} \, \gamma n p \quad \text{with} \quad \eta_\text{singlet} \approx 0.25.
$$

---

## 4.2 Exciton Decay and Loss

Once formed, excitons can:
1. Radiatively decay: $k_r(S) \, S$,
2. Non-radiatively decay: $k_{nr}(S) \, S$,
3. Undergo quenching by polarons or impurities:
   - $k_q(n) \, n \, S$ (electron–exciton quenching),
   - $k_q(p) \, p \, S$ (hole–exciton quenching),
4. Exciton-Exciton Annihilation (e.g., $k_{SS} \, S^2$),
5. Intersystem Crossing to triplets.

Thus, a generic exciton loss term can be written:

$$
\Gamma_\text{loss}(S,n,p) = (k_r(S) + k_{nr}(S)) \, S + k_q(n) \, n \, S + k_q(p) \, p \, S + k_{SS} \, S^2 + \dots
$$

The exciton continuity (3) becomes:

$$
\frac{\partial S}{\partial t} = \eta_\text{exc} \, \gamma n p - (k_r(S) + k_{nr}(S)) \, S - k_q(n) \, n \, S - k_q(p) \, p \, S - k_{SS} \, S^2 - \dots
$$


Requires validation

```python
# -*- coding: utf-8 -*-
"""
Simulation script for a 1D drift-diffusion model of an Organic Light-Emitting Diode (OLED),
including advanced features like Gaussian Disorder Mobility (GDM), Auger recombination,
exciton dynamics (singlet and triplet populations), diffusion, and various quenching mechanisms.
The simulation performs voltage continuation steps to reach a target steady-state voltage
and then simulates the transient decay after the voltage is turned off.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp # ODE solver
from scipy.special import expit, erfc # Special math functions (not used in final version but potentially useful)
import warnings # To display warnings
from tqdm.auto import tqdm # Progress bars
import multiprocessing # Potentially for future parallelization (not used currently)
import time # For timing (not used actively)
import os # For interacting with the operating system (creating directories, checking files)

# --- Physical Constants ---
Q = 1.602e-19       # Elementary charge (Coulombs)
KB = 1.38e-23       # Boltzmann constant (J/K)
EPS0 = 8.854e-12    # Permittivity of free space (F/m)
HBAR = 1.054571817e-34 # Reduced Planck constant (J*s) (Not actively used in this drift-diffusion part)

# --- Helper Functions ---

def bernoulli(x):
    """
    Computes the Bernoulli function B(x) = x / (exp(x) - 1).

    Physics/Numerical Relevance:
    This function is the cornerstone of the Scharfetter-Gummel discretization scheme used for the
    drift-diffusion current equations (Jn, Jp). Integrating the current equation between two
    grid points (assuming a constant electric field or linearly varying potential between them)
    leads to this function. It correctly weights the contributions of carrier densities at
    adjacent nodes to the current flowing between them, ensuring numerical stability and accuracy
    regardless of whether drift or diffusion dominates, and properly handling the exponential
    relationship between carrier density and potential differences (dV/Vth).

    Numerical Stability Handling:
    - For |x| << 1 (small potential difference relative to thermal voltage): Uses a Taylor expansion
      B(x) ≈ 1 - x/2 + x^2/12 to avoid 0/0 indeterminate form and maintain accuracy.
    - For x >> 1 (large positive potential difference): Uses B(x) ≈ x * exp(-x) to avoid overflow in exp(x).
    - For x << -1 (large negative potential difference): Uses B(x) ≈ -x as exp(x) -> 0.
    - For intermediate x: Uses x / expm1(x), where expm1(x) = exp(x) - 1 computes the denominator
      accurately even when exp(x) is close to 1.

    Args:
        x (np.ndarray or float): Input array or scalar, typically representing (phi_i - phi_{i+1}) / Vth.

    Returns:
        np.ndarray or float: The computed Bernoulli function values.
    """
    x = np.asanyarray(x)  # Ensure input is a numpy array
    b = np.zeros_like(x)  # Initialize output array

    # --- Handle different regimes for numerical stability ---
    # For |x| very small, use Taylor expansion: B(x) ≈ 1 - x/2 + x^2/12
    small_mask = np.abs(x) < 1e-10
    b[small_mask] = 1.0 - 0.5 * x[small_mask] + x[small_mask]**2 / 12.0

    # For large positive x, exp(x) dominates: B(x) ≈ x * exp(-x)
    large_pos_mask = x > 700  # Avoid overflow in exp(x)
    b[large_pos_mask] = x[large_pos_mask] * np.exp(-x[large_pos_mask])

    # For large negative x (x -> -inf), exp(x) -> 0: B(x) ≈ -x
    large_neg_mask = x < -700 # Avoid overflow in exp(x) in denominator if using expm1
    b[large_neg_mask] = -x[large_neg_mask]

    # For intermediate values, compute directly using expm1(x) = exp(x) - 1 for better precision
    valid_mask = ~(small_mask | large_pos_mask | large_neg_mask)
    b[valid_mask] = x[valid_mask] / np.expm1(x[valid_mask])

    # Handle potential NaN inputs
    b[np.isnan(x)] = np.nan

    return b

def gaussian_disorder_mobility(E_abs, T, mu0, sigma_hop, C_gdm):
    """
    Calculates the charge carrier mobility based on the Gaussian Disorder Model (GDM).

    Physics Background:
    In disordered organic materials, charge transport occurs via hopping between localized
    states (e.g., individual molecules or segments of polymer chains). The GDM assumes these
    states have energies distributed according to a Gaussian function with width `sigma_hop`.
    Hopping is thermally activated and influenced by the electric field.

    The GDM formula used here captures two main effects:
    1. Temperature Dependence: `exp[-(2σ/3kT)²]` term. Higher temperature (T) provides more thermal
       energy (kT) for carriers to overcome energy barriers between sites. Higher disorder (σ)
       increases the average barrier height, reducing mobility. The (2/3)² factor arises from
       percolation theory arguments in this model.
    2. Field Dependence: `exp[C * sqrt(E)]` term (often linked to Poole-Frenkel effect or field-induced
       barrier lowering). Higher electric fields (E) effectively lower the energy barriers for
       hopping in the field direction, increasing mobility. `C_gdm` is an empirical factor
       characterizing the strength of this field activation.
    `mu0` represents the mobility prefactor, conceptually the mobility in the limit of
    zero field and very high temperature (or low disorder).

    Args:
        E_abs (np.ndarray): Absolute value of the electric field (V/m).
        T (float): Device temperature (K).
        mu0 (float): Zero-field, high-temperature mobility prefactor (m²/Vs).
        sigma_hop (float): Width (standard deviation) of the Gaussian density of states (DOS) (eV).
                           Represents the energetic disorder.
        C_gdm (float): GDM field activation factor (units sqrt(m/V)). Often related to lattice constant
                       and attempts frequency in more detailed models.

    Returns:
        np.ndarray: Calculated mobility values (m²/Vs), clipped to a minimum value for numerical stability
                    and physical realism (mobility doesn't become truly zero).
    """
    kBT_eV = KB * T / Q # Thermal energy in eV
    if kBT_eV < 1e-6:  # Avoid division by zero at T=0K
        return np.zeros_like(E_abs) + 1e-18 # Return a very small mobility

    # GDM temperature-dependent factor
    factor_T = (2.0 * sigma_hop / (3.0 * kBT_eV))**2

    # Ensure field is slightly positive to avoid sqrt(0) issues or warnings, though sqrt(0) is valid
    E_safe = np.maximum(E_abs, 1e-10)

    # GDM formula
    mobility = mu0 * np.exp(-factor_T) * np.exp(C_gdm * np.sqrt(E_safe))

    # Clip mobility to avoid unrealistically small or large values
    return np.clip(mobility, 1e-18, 1e-1) # Min 1e-18, Max 0.1 m^2/Vs

def calculate_langevin_gamma(mu_n_local, mu_p_local, eps):
    """
    Calculates the Langevin bimolecular recombination rate coefficient (gamma_L).

    Physics Background:
    The Langevin model provides a classical estimate for the rate at which free electrons
    and holes encounter each other due to their mutual Coulomb attraction and recombine.
    It assumes that recombination happens inevitably once an electron and hole drift
    within a certain capture radius of each other. The rate at which they approach
    is proportional to their relative mobility (μ_n + μ_p). The Coulomb interaction
    strength is modulated by the material's permittivity (ε).

    The formula is: gamma_L = q * (μ_n + μ_p) / ε

    This coefficient represents the volume per unit time swept out by the relative motion
    of electron-hole pairs leading to recombination. The bimolecular recombination rate
    is then R_Langevin = gamma_L * n * p.
    In practice, this classical rate often overestimates the actual recombination rate in
    organic materials, hence a `gamma_reduction` factor is typically applied in the main code.

    Args:
        mu_n_local (np.ndarray): Local electron mobility (m²/Vs) at each grid point.
        mu_p_local (np.ndarray): Local hole mobility (m²/Vs) at each grid point.
        eps (float): Material permittivity (ε = ε_r * ε_0) (F/m).

    Returns:
        np.ndarray: Langevin recombination coefficient (m³/s) at each grid point.
    """
    # Ensure mobilities are non-negative
    mu_n_safe = np.maximum(mu_n_local, 0.0)
    mu_p_safe = np.maximum(mu_p_local, 0.0)

    # Calculate Langevin coefficient
    gamma_L = Q * (mu_n_safe + mu_p_safe) / eps

    # Ensure the rate coefficient is non-negative
    return np.maximum(gamma_L, 0.0)

# --- Ramped Voltage (For Continuation Steps) ---
def ramped_drive_voltage_continuation(t, params, V_start, V_end):
    """
    Calculates the applied voltage during a voltage continuation step.

    Simulation Strategy:
    To avoid numerical instabilities associated with applying a large voltage instantly,
    the simulation uses voltage continuation. The total voltage change is broken down
    into smaller steps. Within each step, the voltage is ramped linearly from the
    previous step's end voltage (`V_start`) to the target voltage for this step (`V_end`)
    over a finite time `t_ramp_step`. This allows the system (charge distributions,
    potential profile) to adjust more gradually. After the ramp, the voltage is held
    constant for a stabilization period (handled by the `t_span` in the solver call).

    Args:
        t (float): Current time within the step's simulation time span (s).
        params (dict): Dictionary of simulation parameters, must contain 't_ramp_step'.
        V_start (float): Starting voltage for this ramp step (V). Usually the V_end of the previous step.
        V_end (float): Target voltage for this ramp step (V).

    Returns:
        float: Applied voltage at the anode (x=0) boundary at time t.
    """
    t_ramp = params['t_ramp_step'] # Duration of the voltage ramp for this step
    if t < 0:
        return V_start # Before the ramp starts
    elif t < t_ramp:
        # Linear ramp from V_start to V_end
        return V_start + (V_end - V_start) * (t / t_ramp)
    else:
        # Voltage holds at V_end after the ramp
        return V_end

# --- Ramped Voltage (For Decay Phase) ---
def ramped_drive_voltage_decay(t, params, V_steady):
    """
    Calculates the applied voltage during the transient decay phase.

    Simulation Strategy:
    This function defines the voltage profile for simulating the device behavior *after*
    it has reached a steady state at `V_steady` and the external voltage is turned off.
    It assumes the voltage is held at `V_steady` until a specific turn-off time (`t_off_decay`,
    which is relative to the start of the decay simulation time `t=0`) and then instantaneously
    drops to 0V (or potentially another value, but typically 0V for turn-off).

    Args:
        t (float): Current time within the decay simulation's time span (s).
        params (dict): Dictionary of simulation parameters, expects 't_off_decay', the time
                       relative to the start of the decay simulation when the voltage turns off.
        V_steady (float): The steady-state voltage value just before turn-off (V).

    Returns:
        float: Applied voltage at the anode (x=0) boundary at time t during the decay phase.
    """
    t_off = params.get('t_off_decay', 0.0) # Time at which voltage turns off
                                           # Default to 0 if not specified
    if t < t_off:
        return V_steady # Voltage is held before turn-off time
    else:
        return 0.0      # Voltage is zero after turn-off

# --- Initial Condition (Smoothed) ---
def build_initial_condition_smoothed(N, params):
    """
    Builds the initial state vector y0 = [phi, n, p, S, T] for the *first* simulation step (usually V=0).

    Physics & Numerical Strategy:
    - Initial Potential (`phi0`): Typically starts flat at 0V across the device before any voltage is applied.
      The boundary conditions (anode voltage, cathode grounded) will be enforced later by the solver/ODE function.
    - Initial Carrier Densities (`n0`, `p0`): Set to low background levels (`n0_background`, `p0_background`)
      representing intrinsic carriers or residual doping in the "off" state.
    - Smoothing near Contacts: A sharp discontinuity between the low background density in the bulk and the
      (typically) high fixed boundary densities required for injection (`n_cathode_bc`, `p_anode_bc`) can
      cause large initial gradients and numerical instability. This function applies a smooth transition
      (often linear in log-space for densities spanning orders of magnitude) over a few grid points near
      the contacts. This provides a more numerically stable and somewhat more physically plausible
      starting point than an abrupt step.
    - Initial Exciton Densities (`S0`, `T0`): Assumed to be zero initially, as no injection or recombination
      has occurred yet.

    Args:
        N (int): Number of spatial grid points.
        params (dict): Dictionary of simulation parameters, including background densities
                       ('n0_background', 'p0_background') and boundary condition target densities
                       ('n_cathode_bc', 'p_anode_bc').

    Returns:
        np.ndarray: The initial state vector `y0` of size 5*N, flattened as [phi_0..N-1, n_0..N-1, ...].
    """
    # Initial potential profile (zero everywhere initially, BCs applied later)
    phi0 = np.zeros(N)

    # Background carrier densities
    n0_bg = params.get('n0_background', 1e6) # Default if not specified
    p0_bg = params.get('p0_background', 1e6) # Default if not specified

    # Target boundary densities
    n_cathode_target = params['n_cathode_bc'] # Electron density at cathode (x=L)
    p_anode_target = params['p_anode_bc']   # Hole density at anode (x=0)

    # Initialize carrier densities to background levels
    n0 = np.full(N, n0_bg)
    p0 = np.full(N, p0_bg)

    # Smooth the transition from boundary condition to background near contacts
    # Use a few grid points for smoothing to avoid sharp initial gradients
    num_smooth_points = min(5, N // 4) # Use up to 5 points or 1/4 of grid, whichever is smaller

    if num_smooth_points > 1:
        # Smooth holes near anode (x=0)
        p_smooth_indices = np.arange(num_smooth_points)
        # Use log scale for smoother transition over orders of magnitude
        log_p_target = np.log10(max(p_anode_target, 1e-10)) # Avoid log10(0)
        log_p_bg = np.log10(max(p0_bg, 1e-10))
        # Linspace in log space, then convert back, reverse to go from anode BC to bulk
        p0[p_smooth_indices] = 10**(np.linspace(log_p_target, log_p_bg, num_smooth_points)[::-1])
        p0[0] = p_anode_target # Ensure exact BC at the boundary node

        # Smooth electrons near cathode (x=L)
        n_smooth_indices = np.arange(N - num_smooth_points, N)
        # Use log scale
        log_n_bg = np.log10(max(n0_bg, 1e-10))
        log_n_target = np.log10(max(n_cathode_target, 1e-10))
        # Linspace in log space, then convert back
        n0[n_smooth_indices] = 10**(np.linspace(log_n_bg, log_n_target, num_smooth_points))
        n0[-1] = n_cathode_target # Ensure exact BC at the boundary node
    else:
        # If grid is too small for smoothing, just set boundary values directly
        n0[-1] = n_cathode_target
        p0[0] = p_anode_target

    # Initial exciton densities (Singlet S, Triplet T) are zero
    S0 = np.zeros(N)
    T0 = np.zeros(N)

    # Concatenate all variables into a single state vector y = [phi, n, p, S, T]
    return np.concatenate([phi0, n0, p0, S0, T0])

# --- Main PDE System Definition (FOR CONTINUATION STEPS) ---
def oled_1d_dde_equations_advanced_cont(t, y, params, V_start, V_end):
    """
    Defines the system of coupled partial differential equations (PDEs) for the OLED model,
    discretized in space using the Method of Lines, resulting in a system of ODEs dy/dt = f(t, y).
    This version is specifically used during the voltage continuation steps where the voltage is ramping up.

    Equations Solved (Spatially Discretized):
    1. Poisson's Eq (Relaxed Form): `eps_phi * d(phi)/dt = Div(eps * Grad(phi)) + q*(n - p)`
       - Solves for potential `phi` consistent with charge densities `n`, `p`.
       - `eps_phi` is an artificial capacitance for numerical stability, not physical time evolution of phi.
       - Discretized using central differences for the second derivative (Grad(phi) -> E, Div(E) -> d2phi/dx2).
    2. Electron Continuity Eq: `d(n)/dt = (1/q) * Div(Jn) - R_total_carrier_loss`
       - Tracks electron density `n`.
       - `Div(Jn)` is the divergence of electron current density `Jn`.
       - `R_total_carrier_loss` includes bimolecular and Auger recombination.
       - Discretized using Scharfetter-Gummel for `Jn` between nodes, then finite difference for divergence.
    3. Hole Continuity Eq: `d(p)/dt = -(1/q) * Div(Jp) - R_total_carrier_loss`
       - Tracks hole density `p`. Note the minus sign for divergence term relative to electrons.
       - `Div(Jp)` is the divergence of hole current density `Jp`.
       - Discretized similarly to electrons.
    4. Singlet Exciton Continuity Eq: `d(S)/dt = G_S - Gamma_loss_S_net + Div(Ds * Grad(S))`
       - Tracks singlet density `S`.
       - `G_S` = Generation rate (from bimolecular recombination).
       - `Gamma_loss_S_net` = Net loss rate (decay, ISC, quenching, annihilation, gain from RISC).
       - `Div(Ds * Grad(S))` = Diffusion term (approximated by Ds * d2S/dx2).
    5. Triplet Exciton Continuity Eq: `d(T)/dt = G_T - Gamma_loss_T_net + Div(Dt * Grad(T))`
       - Tracks triplet density `T`.
       - `G_T` = Generation rate (from bimolecular recombination).
       - `Gamma_loss_T_net` = Net loss rate (decay, RISC loss, quenching, annihilation, gain from ISC).
       - `Div(Dt * Grad(T))` = Diffusion term (approximated by Dt * d2T/dx2).

    Args:
        t (float): Current simulation time (s). Passed by the ODE solver.
        y (np.ndarray): Current state vector [phi, n, p, S, T] flattened (size 5*N).
        params (dict): Dictionary containing all physical and numerical parameters.
        V_start (float): Starting voltage for the current ramp step (V). Used by voltage ramp function.
        V_end (float): Ending voltage for the current ramp step (V). Used by voltage ramp function.

    Returns:
        np.ndarray: The time derivatives dy/dt (flattened, size 5*N) for the ODE solver.
    """
    # --- Unpack Parameters ---
    N = params.get('N')                 # Number of grid points
    dL = params.get('dL')               # Grid spacing (m)
    eps = params.get('eps')             # Material permittivity (F/m)
    q = params.get('q')                 # Elementary charge (C) - use uppercase Q consistently? No, local var q fine.
    T_dev = params.get('T_dev')         # Device temperature (K)
    Vth = params.get('Vth')             # Thermal voltage kT/q (V)
    # Check essential parameters are present
    if any(v is None for v in [N, dL, eps, q, T_dev, Vth]):
        raise ValueError("Essential parameters (N, dL, eps, q, T_dev, Vth) missing in params dict")

    # Mobility parameters (GDM)
    mu0_n, sigma_hop_n, C_gdm_n = params['gdm_n'] # Electron GDM params
    mu0_p, sigma_hop_p, C_gdm_p = params['gdm_p'] # Hole GDM params
    use_gdm = params.get('use_gdm', True) # Flag to enable/disable GDM

    # Recombination parameters
    gamma_reduction = params['gamma_reduction']   # Reduction factor for Langevin recombination
    C_n_aug, C_p_aug = params.get('auger_coeffs', (0.0, 0.0)) # Auger coefficients (nnp, ppn) (m^6/s)
    spin_fraction_singlet = params['spin_fraction_singlet'] # Fraction of recombinations forming singlets

    # Exciton dynamics parameters
    kr = params['kr']                   # Singlet radiative decay rate (1/s)
    knr = params['knr']                 # Singlet non-radiative decay rate (1/s)
    k_isc = params['k_isc']             # Intersystem crossing rate (S -> T) (1/s)
    k_risc = params.get('k_risc', 0.0)  # Reverse intersystem crossing rate (T -> S) (1/s)
    k_tph = params.get('k_tph', 0.0)    # Triplet phosphorescent decay rate (1/s)
    k_tnr = params['k_tnr']             # Triplet non-radiative decay rate (1/s)
    Ds = params['Ds']                   # Singlet diffusion coefficient (m^2/s)
    Dt = params['Dt']                   # Triplet diffusion coefficient (m^2/s)

    # Quenching parameters
    kq_sn = params['kq_sn']             # Singlet quenching by electrons (m³/s)
    kq_sp = params['kq_sp']             # Singlet quenching by holes (m³/s)
    kq_tn = params['kq_tn']             # Triplet quenching by electrons (m³/s)
    kq_tp = params['kq_tp']             # Triplet quenching by holes (m³/s)
    k_ss = params['k_ss']               # Singlet-singlet annihilation rate (m³/s)
    k_tta = params['k_tta']             # Triplet-triplet annihilation rate (m³/s)

    # Boundary conditions values
    n_cathode_bc_val = params['n_cathode_bc'] # Electron density at cathode (m^-3)
    p_anode_bc_val = params['p_anode_bc']   # Hole density at anode (m^-3)

    # Numerical parameters
    eps_phi = params['eps_phi']         # Artificial permittivity for Poisson eq. stabilization (F/m)

    # --- Reshape State Vector ---
    # Extract individual variables from the flattened state vector y
    phi = y[0*N : 1*N]  # Potential (V)
    n = y[1*N : 2*N]  # Electron density (m^-3)
    p = y[2*N : 3*N]  # Hole density (m^-3)
    S = y[3*N : 4*N]  # Singlet exciton density (m^-3)
    T = y[4*N : 5*N]  # Triplet exciton density (m^-3)

    # --- Ensure Physical Bounds ---
    # Apply minimum carrier densities and non-negative exciton densities
    # Avoids issues with log scales or division by zero in calculations
    n = np.maximum(n, 1e4) # Minimum electron density (e.g., 1 m^-3 or slightly higher)
    p = np.maximum(p, 1e4) # Minimum hole density
    S = np.maximum(S, 0.0) # Singlets cannot be negative
    T = np.maximum(T, 0.0) # Triplets cannot be negative

    # --- Initialize Derivative Arrays ---
    dphi_dt=np.zeros(N)
    dn_dt=np.zeros(N)
    dp_dt=np.zeros(N)
    dS_dt=np.zeros(N)
    dT_dt=np.zeros(N)

    # --- Apply Boundary Conditions to State Variables ---
    # Dirichlet boundary conditions are applied directly to the state variables
    # Potential: Time-dependent voltage at anode (x=0), grounded cathode (x=L)
    phi[0] = ramped_drive_voltage_continuation(t, params, V_start, V_end)
    phi[-1] = 0.0
    # Carrier densities: Fixed at contacts
    n[-1] = n_cathode_bc_val # Electron density at cathode (N-1 index)
    p[0] = p_anode_bc_val    # Hole density at anode (0 index)
    # Excitons: Assumed to be zero at contacts (e.g., quenched or non-existent)
    S[0] = S[-1] = 0.0
    T[0] = T[-1] = 0.0

    # --- Calculate Spatially Varying Parameters ---
    # Electric Field (E = -dV/dx), using central difference for interior points
    E = np.zeros(N)
    E[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dL) # Central difference
    E[0]   = -(phi[1] - phi[0]) / dL        # Forward difference at anode
    E[-1]  = -(phi[-1] - phi[-2]) / dL       # Backward difference at cathode
    E_abs = np.abs(E)

    # Carrier Mobilities (using GDM or constant values)
    if use_gdm:
        mu_n_local = gaussian_disorder_mobility(E_abs, T_dev, mu0_n, sigma_hop_n, C_gdm_n)
        mu_p_local = gaussian_disorder_mobility(E_abs, T_dev, mu0_p, sigma_hop_p, C_gdm_p)
    else:
        # Use constant mobilities if GDM is disabled
        mu_n_local = np.maximum(np.full(N, mu0_n), 1e-18) # Ensure positive
        mu_p_local = np.maximum(np.full(N, mu0_p), 1e-18) # Ensure positive

    # Carrier Diffusion Coefficients (Einstein relation: D = μ * Vth)
    Dn_local = mu_n_local * Vth
    Dp_local = mu_p_local * Vth

    # --- Finite Difference Calculations (Scharfetter-Gummel Scheme) ---
    # Calculate currents Jn and Jp at the interfaces between grid points (N-1 interfaces)
    # Potential difference between adjacent points
    dV_int = phi[:-1] - phi[1:]
    # Argument for Bernoulli functions (clipped for stability)
    x_int = np.clip(dV_int / Vth, -500, 500)
    # Bernoulli functions evaluated at interfaces
    B_plus = bernoulli(x_int)
    B_minus = bernoulli(-x_int)

    # Average diffusion coefficients at interfaces
    Dn_int = 0.5 * (Dn_local[:-1] + Dn_local[1:])
    Dp_int = 0.5 * (Dp_local[:-1] + Dp_local[1:])

    # Electron current density (Jn) at interfaces
    Jn_int = (q * Dn_int / dL) * ( B_minus * n[1:] - B_plus * n[:-1] )
    # Hole current density (Jp) at interfaces
    Jp_int = (q * Dp_int / dL) * ( B_plus * p[:-1] - B_minus * p[1:] ) # Note sign convention difference

    # Divergence of currents (dJ/dx) at interior grid points (N-2 points)
    # Calculated using central difference of interface currents
    div_Jn = (Jn_int[1:] - Jn_int[:-1]) / dL # Jn[i+1/2] - Jn[i-1/2]
    div_Jp = (Jp_int[1:] - Jp_int[:-1]) / dL # Jp[i+1/2] - Jp[i-1/2]

    # --- Calculate Rates (Recombination, Generation, Diffusion, Quenching) ---
    # Select interior points for rate calculations (excluding boundaries)
    n_int = n[1:-1]; p_int = p[1:-1]; S_int = S[1:-1]; T_int = T[1:-1]

    # Bimolecular recombination (Langevin * reduction factor)
    gamma_L_local = calculate_langevin_gamma(mu_n_local[1:-1], mu_p_local[1:-1], eps)
    R_bimol = gamma_reduction * gamma_L_local * (n_int * p_int)
    R_bimol = np.maximum(R_bimol, 0.0) # Ensure non-negative rate

    # Auger recombination
    R_auger = C_n_aug * n_int**2 * p_int + C_p_aug * n_int * p_int**2
    R_auger = np.maximum(R_auger, 0.0) # Ensure non-negative rate

    # Total carrier loss rate due to recombination
    R_total_carrier_loss = R_bimol + R_auger

    # Exciton generation rates (from bimolecular recombination)
    G_S = spin_fraction_singlet * R_bimol       # Singlet generation
    G_T = (1.0 - spin_fraction_singlet) * R_bimol # Triplet generation

    # Exciton diffusion terms (d2/dx2 using central difference)
    # d2S/dx2 = (S[i+1] - 2*S[i] + S[i-1]) / dL^2
    d2S_dx2 = (S[2:] - 2*S_int + S[:-2]) / (dL**2)
    Diffusion_S = Ds * d2S_dx2
    # d2T/dx2 = (T[i+1] - 2*T[i] + T[i-1]) / dL^2
    d2T_dx2 = (T[2:] - 2*T_int + T[:-2]) / (dL**2)
    Diffusion_T = Dt * d2T_dx2

    # Singlet exciton loss/gain terms
    Loss_S_decay = (kr + knr) * S_int        # Radiative and non-radiative decay
    Loss_S_isc = k_isc * S_int                # Intersystem crossing (S -> T)
    Loss_S_quench_n = kq_sn * n_int * S_int     # Quenching by electrons
    Loss_S_quench_p = kq_sp * p_int * S_int     # Quenching by holes
    Loss_S_ssa = k_ss * S_int**2              # Singlet-singlet annihilation
    Gain_S_risc = k_risc * T_int              # Gain from RISC (T -> S)
    # Net loss rate for singlets
    Gamma_loss_S_net = Loss_S_decay + Loss_S_isc + Loss_S_quench_n + Loss_S_quench_p + Loss_S_ssa - Gain_S_risc

    # Triplet exciton loss/gain terms
    Loss_T_decay = (k_tph + k_tnr) * T_int   # Phosphorescent and non-radiative decay
    Loss_T_risc = k_risc * T_int             # Loss to RISC (T -> S)
    Loss_T_quench_n = kq_tn * n_int * T_int    # Quenching by electrons
    Loss_T_quench_p = kq_tp * p_int * T_int    # Quenching by holes
    Loss_T_tta = k_tta * T_int**2             # Triplet-triplet annihilation
    Gain_T_isc = Loss_S_isc                 # Gain from ISC (S -> T)
    # Net loss rate for triplets
    Gamma_loss_T_net = Loss_T_decay + Loss_T_risc + Loss_T_quench_n + Loss_T_quench_p + Loss_T_tta - Gain_T_isc

    # --- Assemble Time Derivatives ---
    # Continuity equations for interior points
    # dn/dt = (1/q) * dJn/dx - R_total
    dn_dt[1:-1] = (1.0/q) * div_Jn - R_total_carrier_loss
    # dp/dt = -(1/q) * dJp/dx - R_total
    dp_dt[1:-1] = -(1.0/q) * div_Jp - R_total_carrier_loss
    # dS/dt = G_S - Gamma_loss_S_net + Diffusion_S
    dS_dt[1:-1] = G_S - Gamma_loss_S_net + Diffusion_S
    # dT/dt = G_T - Gamma_loss_T_net + Diffusion_T
    dT_dt[1:-1] = G_T - Gamma_loss_T_net + Diffusion_T

    # Poisson equation (relaxed form: eps_phi * dphi/dt = eps * d2phi/dx2 - q*(p-n))
    # d2phi/dx2 using central difference
    d2phi_dx2 = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / (dL**2)
    charge_term = q * (p[1:-1] - n[1:-1]) # Net charge density
    dphi_dt[1:-1] = (eps * d2phi_dx2 - charge_term) / eps_phi

    # --- Fix Derivatives at Boundaries ---
    # Set time derivatives to zero at the boundaries (Dirichlet conditions handled above)
    # This ensures the boundary values themselves don't change due to internal dynamics
    # (their values are fixed or externally controlled).
    dphi_dt[0] = dphi_dt[-1] = 0.0
    dn_dt[0] = dn_dt[-1] = 0.0 # Electron density derivative fixed at anode and cathode
    dp_dt[0] = dp_dt[-1] = 0.0 # Hole density derivative fixed at anode and cathode
    dS_dt[0] = dS_dt[-1] = 0.0
    dT_dt[0] = dT_dt[-1] = 0.0

    # Concatenate derivatives into a single vector dy/dt for the ODE solver
    dydt = np.concatenate([dphi_dt, dn_dt, dp_dt, dS_dt, dT_dt])

    return dydt


# --- Main PDE System Definition (FOR DECAY PHASE - Modified BCs) ---
def oled_1d_dde_equations_advanced_decay(t, y, params, V_steady):
    """
    Defines the system of coupled PDEs for the OLED model, specifically tailored for the
    transient decay phase simulation *after* the external voltage has been turned off.

    Key Difference from `oled_1d_dde_equations_advanced_cont`:
    - **Voltage Boundary Condition:** Uses `ramped_drive_voltage_decay(t, params, V_steady)`
      to set `phi[0]`, which typically holds `V_steady` briefly then drops to 0V at `t_off_decay`.
    - **Carrier Boundary Conditions (CRUCIAL):** Instead of fixing `n[-1]` and `p[0]` to high
      injection levels (`n_cathode_bc`, `p_anode_bc`), they are set to the low background
      density levels (`n0_background`, `p0_background`).
      **Physical Justification:** When the external drive voltage is removed, the contacts
      usually cease to inject carriers efficiently. Modeling them as reverting to low,
      near-equilibrium densities (or potentially becoming blocking/extracting contacts)
      is more realistic for simulating the decay of carriers already present *inside* the device.
      This change significantly impacts how quickly carriers are swept out or recombine.
    - **All other physics equations** (Poisson, Continuity, GDM, Recombination, Exciton dynamics)
      and their numerical implementation remain the same as in the continuation function.

    Args:
        t (float): Current simulation time (s) within the decay phase.
        y (np.ndarray): Current state vector [phi, n, p, S, T] flattened (size 5*N).
        params (dict): Dictionary containing all physical and numerical parameters.
        V_steady (float): The steady-state voltage from which the decay started (V). Used by the ramp function.

    Returns:
        np.ndarray: The time derivatives dy/dt (flattened, size 5*N) for the ODE solver during decay.
    """
    # --- Unpack Parameters ---
    # (Identical unpacking as in the _cont function)
    N = params.get('N')
    dL = params.get('dL')
    eps = params.get('eps')
    q = params.get('q')
    T_dev = params.get('T_dev')
    Vth = params.get('Vth')
    if any(v is None for v in [N, dL, eps, q, T_dev, Vth]): raise ValueError("Essential params missing")
    mu0_n, sigma_hop_n, C_gdm_n = params['gdm_n']
    mu0_p, sigma_hop_p, C_gdm_p = params['gdm_p']
    use_gdm = params.get('use_gdm', True)
    gamma_reduction = params['gamma_reduction']
    C_n_aug, C_p_aug = params.get('auger_coeffs', (0.0, 0.0))
    spin_fraction_singlet = params['spin_fraction_singlet']
    kr = params['kr']
    knr = params['knr']
    k_isc = params['k_isc']
    k_risc = params.get('k_risc', 0.0)
    k_tph = params.get('k_tph', 0.0)
    k_tnr = params['k_tnr']
    Ds = params['Ds']
    Dt = params['Dt']
    kq_sn = params['kq_sn']
    kq_sp = params['kq_sp']
    kq_tn = params['kq_tn']
    kq_tp = params['kq_tp']
    k_ss = params['k_ss']
    k_tta = params['k_tta']
    # Boundary conditions (used differently here)
    # n_cathode_bc_val = params['n_cathode_bc'] # Not used directly for n[-1] in decay
    # p_anode_bc_val = params['p_anode_bc']     # Not used directly for p[0] in decay
    eps_phi = params['eps_phi']
    # Get background levels for decay boundary conditions
    n0_background = params.get('n0_background', 1e6)
    p0_background = params.get('p0_background', 1e6)

    # --- Reshape State Vector ---
    phi = y[0*N : 1*N]
    n = y[1*N : 2*N]
    p = y[2*N : 3*N]
    S = y[3*N : 4*N]
    T = y[4*N : 5*N]
    # --- Ensure Physical Bounds ---
    n = np.maximum(n, 1e4)
    p = np.maximum(p, 1e4)
    S = np.maximum(S, 0.0)
    T = np.maximum(T, 0.0)

    # --- Initialize Derivative Arrays ---
    dphi_dt=np.zeros(N)
    dn_dt=np.zeros(N)
    dp_dt=np.zeros(N)
    dS_dt=np.zeros(N)
    dT_dt=np.zeros(N)

    # --- Apply Boundary Conditions to State Variables ---
    # Potential: Uses the decay voltage ramp function
    phi[0] = ramped_drive_voltage_decay(t, params, V_steady)
    phi[-1] = 0.0

    # !! **** MODIFIED BOUNDARY CONDITIONS FOR DECAY PHASE **** !!
    # Carrier densities at contacts are set to low background levels,
    # simulating poor injection or blocking contacts after turn-off.
    n[-1] = n0_background  # Electron density at cathode drops to background
    p[0] = p0_background   # Hole density at anode drops to background
    # !! ***************************************************** !!

    # Excitons: Still assumed zero at contacts
    S[0] = S[-1] = 0.0
    T[0] = T[-1] = 0.0

    # --- Calculate Spatially Varying Parameters ---
    # (Calculation of E, mu, D remains the same as in _cont function)
    E = np.zeros(N); E[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dL)
    E[0] = -(phi[1] - phi[0]) / dL
    E[-1] = -(phi[-1] - phi[-2]) / dL
    E_abs = np.abs(E)
    if use_gdm:
        mu_n_local = gaussian_disorder_mobility(E_abs, T_dev, mu0_n, sigma_hop_n, C_gdm_n)
        mu_p_local = gaussian_disorder_mobility(E_abs, T_dev, mu0_p, sigma_hop_p, C_gdm_p)
    else:
        mu_n_local = np.maximum(np.full(N, mu0_n), 1e-18); mu_p_local = np.maximum(np.full(N, mu0_p), 1e-18)
    Dn_local = mu_n_local * Vth; Dp_local = mu_p_local * Vth

    # --- Finite Difference Calculations ---
    # (Calculation of Jn, Jp, divergences remains the same)
    dV_int = phi[:-1] - phi[1:]
    x_int = np.clip(dV_int / Vth, -500, 500)
    B_plus = bernoulli(x_int); B_minus = bernoulli(-x_int)
    Dn_int = 0.5 * (Dn_local[:-1] + Dn_local[1:])
    Dp_int = 0.5 * (Dp_local[:-1] + Dp_local[1:])
    Jn_int = (q * Dn_int / dL) * ( B_minus * n[1:] - B_plus * n[:-1] )
    Jp_int = (q * Dp_int / dL) * ( B_plus * p[:-1] - B_minus * p[1:] )
    div_Jn = (Jn_int[1:] - Jn_int[:-1]) / dL; div_Jp = (Jp_int[1:] - Jp_int[:-1]) / dL

    # --- Calculate Rates ---
    # (Calculation of R, G, Gamma, Diffusion remains the same)
    n_int = n[1:-1]
    p_int = p[1:-1]
    S_int = S[1:-1]
    T_int = T[1:-1]
    gamma_L_local = calculate_langevin_gamma(mu_n_local[1:-1], mu_p_local[1:-1], eps)
    R_bimol = gamma_reduction * gamma_L_local * (n_int * p_int)
    R_bimol = np.maximum(R_bimol, 0.0)
    R_auger = C_n_aug * n_int**2 * p_int + C_p_aug * n_int * p_int**2
    R_auger = np.maximum(R_auger, 0.0)
    R_total_carrier_loss = R_bimol + R_auger
    G_S = spin_fraction_singlet * R_bimol
    G_T = (1.0 - spin_fraction_singlet) * R_bimol
    d2S_dx2 = (S[2:] - 2*S_int + S[:-2]) / (dL**2)
    Diffusion_S = Ds * d2S_dx2
    d2T_dx2 = (T[2:] - 2*T_int + T[:-2]) / (dL**2)
    Diffusion_T = Dt * d2T_dx2
    Loss_S_decay = (kr + knr) * S_int
    Loss_S_isc = k_isc * S_int
    Loss_S_quench_n = kq_sn * n_int * S_int
    Loss_S_quench_p = kq_sp * p_int * S_int
    Loss_S_ssa = k_ss * S_int**2
    Gain_S_risc = k_risc * T_int
    Gamma_loss_S_net = Loss_S_decay + Loss_S_isc + Loss_S_quench_n + Loss_S_quench_p + Loss_S_ssa - Gain_S_risc
    Loss_T_decay = (k_tph + k_tnr) * T_int; Loss_T_risc = k_risc * T_int; Loss_T_quench_n = kq_tn * n_int * T_int
    Loss_T_quench_p = kq_tp * p_int * T_int; Loss_T_tta = k_tta * T_int**2; Gain_T_isc = Loss_S_isc
    Gamma_loss_T_net = Loss_T_decay + Loss_T_risc + Loss_T_quench_n + Loss_T_quench_p + Loss_T_tta - Gain_T_isc

    # --- Assemble Time Derivatives ---
    # (Identical assembly as in _cont function)
    dn_dt[1:-1] = (1.0/q) * div_Jn - R_total_carrier_loss
    dp_dt[1:-1] = -(1.0/q) * div_Jp - R_total_carrier_loss
    dS_dt[1:-1] = G_S - Gamma_loss_S_net + Diffusion_S
    dT_dt[1:-1] = G_T - Gamma_loss_T_net + Diffusion_T
    d2phi_dx2 = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / (dL**2); charge_term = q * (p[1:-1] - n[1:-1])
    dphi_dt[1:-1] = (eps * d2phi_dx2 - charge_term) / eps_phi

    # --- Fix Derivatives at Boundaries ---
    # (Derivatives still fixed to zero, the boundary *values* were set above)
    dphi_dt[0] = dphi_dt[-1] = 0.0; dn_dt[0] = dn_dt[-1] = 0.0
    dp_dt[0] = dp_dt[-1] = 0.0; dS_dt[0] = dS_dt[-1] = 0.0
    dT_dt[0] = dT_dt[-1] = 0.0

    # Concatenate derivatives
    dydt = np.concatenate([dphi_dt, dn_dt, dp_dt, dS_dt, dT_dt])
    return dydt

# --- Simulation Runner for Voltage Step ---
def run_1d_oled_sim_step(params, V_start, V_end, y0_input=None, solver_method='BDF', save_filename=None):
    """
    Runs a single step of the voltage continuation simulation.

    Workflow:
    1. Initializes the state vector `y0`:
       - If `y0_input` is None (first step), calls `build_initial_condition_smoothed`.
       - Otherwise, uses the final state `y0_input` from the previous voltage step.
    2. Sets the potential boundary conditions in `y0` to match the `V_start` of the current step.
    3. Defines the time span for this step: `t_span_step = (0, t_ramp_step + t_stabilize_step)`.
       The simulation runs for the duration of the voltage ramp plus an additional stabilization period
       at the target voltage `V_end`.
    4. Calls the `scipy.integrate.solve_ivp` ODE solver:
       - Uses the `oled_1d_dde_equations_advanced_cont` function to calculate `dy/dt`.
       - Passes `params`, `V_start`, and `V_end` to the ODE function.
       - Uses specified tolerances (`rtol`, `atol`) and solver method (`BDF` recommended for stiff problems).
    5. Checks the solver success status and handles potential errors.
    6. Optionally saves the final state vector `y` at the end of the step to a file (`save_filename`).
    7. Returns the final state vector, which serves as `y0_input` for the next step.

    Args:
        params (dict): Simulation parameters dictionary.
        V_start (float): Starting voltage for this step (V).
        V_end (float): Target voltage for this step (V).
        y0_input (np.ndarray, optional): Initial state vector [phi, n, p, S, T] from the previous step.
                                        If None, it's the first step, and initial conditions are built.
        solver_method (str, optional): ODE solver method passed to `solve_ivp` (e.g., 'BDF', 'RK45', 'Radau').
                                       Defaults to 'BDF', which is generally suitable for stiff drift-diffusion problems.
        save_filename (str, optional): Full path to save the final state vector `y` of this step using `np.savez_compressed`.
                                       If None, the state is not saved.

    Returns:
        np.ndarray: The final state vector `y` (size 5*N) at the end of this simulation step (at t = t_ramp + t_stabilize).

    Raises:
        RuntimeError: If the ODE solver `solve_ivp` fails to converge or encounters a critical error.
        ValueError: If essential parameters are missing (though checks are basic).
    """
    params = params.copy() # Work on a copy to avoid modifying the original dict
    # Calculate derived parameters if not already present (or update them)
    params['dL'] = params['d'] / (params['N'] - 1) # Grid spacing
    params['eps'] = params['eps_r'] * EPS0        # Permittivity
    params['q'] = Q                               # Elementary charge
    params['Vth'] = KB * params['T_dev'] / Q      # Thermal voltage

    N = params['N'] # Number of grid points

    # Set initial condition
    if y0_input is None:
        # Build initial state from scratch if no previous state is provided
        y0 = build_initial_condition_smoothed(N, params)
        # Ensure initial potential boundary condition matches V_start (anode)
        y0[0] = V_start
        y0[N-1] = 0.0 # Cathode grounded
    else:
        # Use the final state from the previous step as the initial condition
        y0 = y0_input.copy()
        # Update potential boundary conditions to match V_start for the new ramp
        y0[0] = V_start
        y0[N-1] = 0.0 # Cathode grounded

    # Define time span for this step
    t_ramp = params['t_ramp_step']
    # Add stabilization time after the ramp completes
    stabilization_time = params.get('t_stabilize_step', 5 * t_ramp) # Default 5x ramp time
    t_span_step = (0, t_ramp + stabilization_time) # Simulate ramp + stabilization
    # Define time points for output evaluation
    num_t_points_step = params.get('num_t_points_step', 101) # Number of output points
    t_eval_step = np.linspace(t_span_step[0], t_span_step[1], num_t_points_step)

    # Set solver options
    max_step_val = params.get('max_step_init', np.inf) # Max internal step size for solver

    sol = None # Initialize solution object
    # Use tqdm for progress bar
    with tqdm(total=1, desc=f"Step {V_start:.2f}V->{V_end:.2f}V") as pbar:
        try:
            # Call the ODE solver (solve_ivp)
            sol = solve_ivp(
                fun=oled_1d_dde_equations_advanced_cont, # The ODE system function for continuation
                t_span=t_span_step,                     # Time interval
                y0=y0,                                  # Initial state
                method=solver_method,                   # Solver algorithm
                t_eval=t_eval_step,                     # Output times
                args=(params, V_start, V_end),          # Additional args for the ODE function
                rtol=params['rtol'],                    # Relative tolerance
                atol=params['atol'],                    # Absolute tolerance
                max_step=max_step_val                   # Max internal step size
            )
            pbar.update(1) # Update progress bar on completion
        except Exception as e:
            pbar.update(1) # Ensure progress bar updates even on error
            print(f"\nError during solve_ivp for step {V_start:.2f}V->{V_end:.2f}V: {e}")
            raise # Re-raise the exception

    # Check solver status
    if sol and not sol.success:
        warnings.warn(f"Solver Warning (Step {V_start:.2f}V->{V_end:.2f}V): {sol.message}")
    if not sol or sol.t.size < 2 :
        raise RuntimeError(f"ODE solver failed for step {V_start:.2f}V->{V_end:.2f}V.")

    # Extract the final state vector at the end of the simulation time
    final_state = sol.y[:, -1]

    # Save the final state if requested
    if save_filename:
        try:
            # Use numpy's compressed saving for efficiency
            np.savez_compressed(save_filename, y_final=final_state, V_end=V_end, params=params)
        except Exception as e:
            warnings.warn(f"Failed to save final state to {save_filename}: {e}")

    return final_state

# --- Simulation Runner for Decay Phase ---
def run_1d_oled_decay(params, V_steady, y0_decay, solver_method='BDF'):
    """
    Runs the transient decay simulation after the device has reached steady state.

    Workflow:
    1. Takes the final steady-state vector `y0_decay` (obtained at `V_steady` from the continuation)
       as the initial condition.
    2. Defines the time span for the decay simulation: `t_span_decay = (0, t_end_decay)`.
       The voltage turn-off event happens *within* this span at `t = t_off_decay`.
    3. Defines the time points (`t_eval_decay`) at which to store the solution during decay.
    4. Calls the `scipy.integrate.solve_ivp` ODE solver:
       - Uses the `oled_1d_dde_equations_advanced_decay` function, which incorporates the
         modified boundary conditions (voltage drop, low carrier densities at contacts).
       - Passes `params` and the initial steady-state voltage `V_steady` to the ODE function.
    5. Checks solver status and handles errors.
    6. Returns the full `OdeResult` object (`sol`) containing the solution (states `y`) at all
       evaluated time points (`t`) during the decay.

    Args:
        params (dict): Simulation parameters dictionary. Must include decay-specific controls
                       like `t_off_decay`, `t_end_decay`, `num_t_points_decay`.
        V_steady (float): The steady-state voltage (V) at which the device was operating *before*
                          the decay simulation starts. This value is needed by the decay ODE function's
                          voltage boundary condition logic.
        y0_decay (np.ndarray): The initial state vector [phi, n, p, S, T] (size 5*N) representing the
                               steady state achieved at `V_steady`.
        solver_method (str, optional): ODE solver method ('BDF', 'RK45', etc.). Defaults to 'BDF'.

    Returns:
        scipy.integrate.OdeResult: The solution object returned by `solve_ivp`. Key attributes are:
                                   - `sol.t`: Array of time points (s).
                                   - `sol.y`: Array of state vectors at each time point (shape [5*N, num_t_points_decay]).
                                   - `sol.success`: Boolean indicating success.
                                   - `sol.message`: Solver status message.

    Raises:
        RuntimeError: If the ODE solver fails critically.
        ValueError: If essential decay parameters are missing in `params`.
    """
    params = params.copy() # Work on a copy
    # Calculate/update derived parameters
    params['dL'] = params['d'] / (params['N'] - 1)
    params['eps'] = params['eps_r'] * EPS0
    params['q'] = Q
    params['Vth'] = KB * params['T_dev'] / Q

    N = params['N']
    y0 = y0_decay.copy() # Use the provided steady state as initial condition

    # Define time span for the decay simulation
    t_off_decay = params.get('t_off_decay', 1e-12) # Time offset when V drops (can be near 0)
    t_end_decay = params.get('t_end_decay', 5e-6)  # Total simulation time for decay
    t_span_decay = (0, t_end_decay) # Simulate from t=0 up to t_end_decay
                                     # Voltage turn-off happens *during* this span based on t_off_decay

    # Define time points for output evaluation during decay
    num_t_points_decay = params.get('num_t_points_decay', 501)
    # Often useful to have log-spaced points for decay, but linear is simpler here
    t_eval_decay = np.linspace(t_span_decay[0], t_span_decay[1], num_t_points_decay)

    # Set solver options for decay phase
    max_step_val = params.get('max_step_decay', np.inf) # Max internal step size for decay

    print(f"Starting Decay Simulation (V_steady={V_steady:.2f}V, t_end={t_end_decay:.1e}s)...")
    sol = None # Initialize solution object
    # Use tqdm for progress bar
    with tqdm(total=1, desc=f"Decay from {V_steady:.2f}V") as pbar:
        try:
            # Call the ODE solver with the decay-specific ODE function
            sol = solve_ivp(
                fun=oled_1d_dde_equations_advanced_decay, # The ODE system function for decay
                t_span=t_span_decay,                     # Time interval
                y0=y0,                                  # Initial state (steady state)
                method=solver_method,                   # Solver algorithm
                t_eval=t_eval_decay,                     # Output times
                args=(params, V_steady),                # Pass V_steady to decay ODE function
                rtol=params['rtol'],                    # Relative tolerance
                atol=params['atol'],                    # Absolute tolerance
                max_step=max_step_val                   # Max internal step size
            )
            pbar.update(1) # Update progress bar
        except Exception as e:
            pbar.update(1) # Update progress bar on error
            print(f"\nError during solve_ivp for decay phase: {e}")
            raise # Re-raise exception

    print(f"Decay ODE solver finished: {sol.message if sol else 'Error occurred'}")
    # Check solver status
    if sol and not sol.success:
        warnings.warn(f"Decay Solver Warning: {sol.message}")
    if not sol or sol.t.size < 2:
        raise RuntimeError("Decay ODE solver failed.")

    print("Decay simulation run completed.")
    # Return the full solution object (contains time steps and states)
    return sol


# --- Main Execution (Voltage Continuation + Decay) ---
def main_voltage_continuation_and_decay():
    """
    Main function to orchestrate the full OLED simulation workflow.

    Workflow:
    1. Defines the base dictionary `base_params` containing all physical, numerical,
       and simulation control parameters for the OLED device and simulation.
    2. Calculates derived parameters (dL, eps, Vth) and adds them to the `params` dict.
    3. Sets up the voltage steps (`voltage_steps`) for the continuation procedure (e.g., 0V to 5V).
    4. Creates a directory (`save_dir`) to store intermediate state files.
    5. Iterates through the voltage steps:
       - For each step (V_start -> V_end):
         - Defines a filename (`save_fname`) for the state at V_end.
         - Checks if a valid saved state file already exists for V_end.
         - If yes, loads the state (`current_y`) from the file.
         - If no (or load fails), calls `run_1d_oled_sim_step` to simulate the step,
           passing the `current_y` from the previous step (or None for the first step).
           Saves the result if `save_intermediate` is True.
         - Stores the resulting `current_y` (the state at V_end) in the `final_states` dictionary.
    6. After the loop, if continuation was successful:
       - Retrieves the final steady state (`y0_decay`) at the target voltage (`target_voltage`).
       - Calls `run_1d_oled_decay` to simulate the transient decay starting from `y0_decay`.
       - Calls `plot_final_state_and_decay` to visualize both the steady-state profiles
         at the target voltage and the results of the transient decay simulation.
    7. Includes error handling to catch issues during continuation or decay/plotting phases.
    """
    print("--- Starting OLED Simulation with Voltage Continuation & Decay ---")

    # --- Base Simulation Parameters ---
    # Define all physical, numerical, and simulation control parameters in a dictionary
    base_params = {
        # Device structure
        'N': 101,              # Number of spatial grid points (-)
        'd': 100e-9,           # Device thickness (m)
        'eps_r': 3.0,          # Relative permittivity (-)
        'T_dev': 300.0,        # Device temperature (K)

        # Mobility (Gaussian Disorder Model)
        # Tuple format: (mu0 [m^2/Vs], sigma_hop [eV], C_gdm [sqrt(m/V)])
        'gdm_n': (1e-7, 0.08, 3e-4), # Electron mobility parameters
        'gdm_p': (1e-7, 0.08, 3e-4), # Hole mobility parameters
        'use_gdm': True,       # Whether to use GDM or constant mobility (mu0)

        # Recombination
        'gamma_reduction': 0.1, # Factor to reduce Langevin recombination rate (-)
        'auger_coeffs': (0.0, 0.0), # Auger coefficients (C_n, C_p) [m^6/s] (set to 0 means no Auger)
        'spin_fraction_singlet': 0.25, # Fraction of e-h pairs forming singlets (-)

        # Exciton Dynamics Rates (units: 1/s or m^3/s or m^2/s)
        'kr': 1e7,             # Singlet radiative decay rate (1/s)
        'knr': 5e6,            # Singlet non-radiative decay rate (1/s)
        'k_isc': 2e7,          # Intersystem crossing (S->T) rate (1/s)
        'k_risc': 0.0,         # Reverse ISC (T->S) rate (1/s) (set to 0 means no RISC/TADF)
        'k_tph': 0.0,          # Triplet phosphorescent decay rate (1/s) (often negligible for fluorescence)
        'k_tnr': 1e5,          # Triplet non-radiative decay rate (1/s)
        'Ds': 1e-9,            # Singlet diffusion coefficient (m^2/s)
        'Dt': 1e-10,           # Triplet diffusion coefficient (m^2/s)

        # Quenching Rates (units: m^3/s)
        'kq_sn': 1e-19,        # Singlet quenching by electrons
        'kq_sp': 1e-19,        # Singlet quenching by holes
        'kq_tn': 1e-19,        # Triplet quenching by electrons
        'kq_tp': 1e-19,        # Triplet quenching by holes
        'k_ss': 1e-18,         # Singlet-singlet annihilation
        'k_tta': 5e-18,        # Triplet-triplet annihilation

        # Boundary Conditions & Background
        'n_cathode_bc': 1e23,  # Electron density at cathode (m^-3) for injection
        'p_anode_bc': 1e23,    # Hole density at anode (m^-3) for injection
        'n0_background': 1e6,  # Background electron density (m^-3) (used for initial state & decay BC)
        'p0_background': 1e6,  # Background hole density (m^-3) (used for initial state & decay BC)

        # Numerical Parameters
        'eps_phi': 1e-14,      # Artificial permittivity for Poisson eq. (F/m) (stabilization)
        'rtol': 1e-4,          # Relative tolerance for ODE solver
        'atol': 1e-7,          # Absolute tolerance for ODE solver (can be array later if needed)

        # Simulation Control - Continuation Steps
        't_ramp_step': 1e-7,       # Duration of voltage ramp in each step (s)
        't_stabilize_step': 5e-7,  # Additional time after ramp for stabilization (s)
        'num_t_points_step': 101,  # Number of output time points per step (-)
        'save_intermediate': True, # Save state after each voltage step?
        'solver_method': 'BDF',    # ODE solver for stiff problems

        # Simulation Control - Decay Phase
        't_off_decay': 1e-12,      # Time delay before voltage turns off in decay sim (s) (relative to start of decay sim)
        't_end_decay': 10e-6,      # Total duration of decay simulation (s)
        'num_t_points_decay': 1001,# Number of output time points for decay (-)
    }

    # Calculate derived parameters and add them to the dict (can be done inside runners too)
    base_params['dL'] = base_params['d'] / (base_params['N'] - 1)
    base_params['eps'] = base_params['eps_r'] * EPS0
    base_params['q'] = Q
    base_params['Vth'] = KB * base_params['T_dev'] / Q

    # --- Voltage Continuation Setup ---
    # Define the sequence of voltage points to simulate
    voltage_steps = np.linspace(0.0, 5.0, 11) # e.g., 0V to 5V in 11 steps (0, 0.5, ..., 5.0)
    print(f"Voltage steps: {voltage_steps}")

    current_y = None          # Holds the state vector between steps
    final_states = {}         # Dictionary to store final state for each voltage V_end
    save_dir = "oled_continuation_states" # Directory to save intermediate results
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist

    continuation_success = False # Flag to track if continuation finishes
    # --- Run Voltage Continuation Loop ---
    try:
        for i in range(len(voltage_steps) - 1):
            V_start = voltage_steps[i]
            V_end = voltage_steps[i+1]

            # Define filename for saving/loading intermediate state
            save_fname = os.path.join(save_dir, f"oled_state_V{V_end:.2f}.npz") if base_params['save_intermediate'] else None

            load_success = False
            # Attempt to load previously saved state if it exists
            if save_fname and os.path.exists(save_fname):
                try:
                    print(f"Attempting to load state for {V_end:.2f}V from {save_fname}...")
                    data = np.load(save_fname, allow_pickle=True)
                    # Check if the loaded voltage matches the expected end voltage
                    if np.isclose(data['V_end'], V_end):
                        current_y = data['y_final']
                        print(f"Successfully loaded state for {V_end:.2f}V.")
                        load_success = True
                    else:
                        print(f"Voltage mismatch in saved file (Expected {V_end:.2f}V, Found {data['V_end']:.2f}V). Recalculating.")
                except Exception as load_err:
                    print(f"Error loading {save_fname}: {load_err}. Recalculating.")

            # If loading failed or was skipped, run the simulation step
            if not load_success:
                 print(f"Running simulation step from {V_start:.2f}V to {V_end:.2f}V...")
                 current_y = run_1d_oled_sim_step(base_params, V_start, V_end, y0_input=current_y,
                                                  solver_method=base_params['solver_method'], save_filename=save_fname)

            # Store the final state of this step in the dictionary
            final_states[V_end] = current_y

        print("\n--- Voltage Continuation Finished Successfully ---")
        continuation_success = True

    except Exception as e:
        # Catch errors during the continuation loop
        print(f"\n--- Error during Voltage Continuation at step {V_start:.2f}V -> {V_end:.2f}V ---")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

    # --- Run Decay Simulation ---
    if continuation_success:
        # Proceed only if voltage continuation finished without errors
        target_voltage = voltage_steps[-1] # The final voltage reached

        if target_voltage in final_states:
            # Get the final steady state achieved at the target voltage
            y0_decay = final_states[target_voltage]
            try:
                # Run the decay simulation starting from this state
                decay_sol = run_1d_oled_decay(base_params, target_voltage, y0_decay,
                                              solver_method=base_params['solver_method'])

                # --- Plotting ---
                print("Plotting final steady state and transient decay results...")
                plot_final_state_and_decay(y0_decay, target_voltage, decay_sol, base_params)

            except Exception as e:
                # Catch errors during the decay simulation or plotting
                print("\n--- Error during Decay Simulation or Plotting ---")
                print(f"{type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            # This shouldn't happen if continuation succeeded, but check anyway
            print(f"Error: Final state for target voltage {target_voltage}V not found in results.")
    else:
        # Skip decay simulation if continuation failed
        print("Voltage continuation failed. Skipping decay simulation.")


# --- Plotting Function ---
def plot_final_state_and_decay(y_steady, voltage, decay_sol, params):
    """
    Generates plots to visualize the simulation results.

    Plots Generated:
    1. Steady-State Profiles vs. Position (at `voltage`):
       - Electrostatic Potential (`phi`)
       - Carrier Densities (`n`, `p`) on a log scale
       - Exciton Densities (`S`, `T`) on a log scale
    2. Transient Decay vs. Time (after turn-off at `t_off_decay`):
       - Total Electroluminescence (EL) intensity (integrated `kr * S`) on linear scale
       - Total Electroluminescence (EL) intensity on log-log scale
       - Total integrated Singlet and Triplet populations (`∫S dx`, `∫T dx`) on log-linear scale

    Args:
        y_steady (np.ndarray): The steady-state vector [phi, n, p, S, T] (size 5*N) achieved at `voltage`.
        voltage (float): The final steady-state voltage (V) corresponding to `y_steady`.
        decay_sol (scipy.integrate.OdeResult): The solution object returned by `run_1d_oled_decay`,
                                               containing the time evolution (`t`, `y`) during decay.
        params (dict): The simulation parameters dictionary, used for extracting N, dL, kr, d etc.
    """
    # --- Extract Data ---
    N = params['N']; dL = params['dL']; kr = params['kr']; d = params['d']
    x_nm = np.linspace(0, d * 1e9, N) # Spatial coordinate in nm

    # Steady-state profiles
    phi_s = y_steady[0*N : 1*N]
    n_s = y_steady[1*N : 2*N]
    p_s = y_steady[2*N : 3*N]
    S_s = y_steady[3*N : 4*N]
    T_s = y_steady[4*N : 5*N]
    # Apply floors for plotting (especially log scale)
    n_s = np.maximum(n_s, 1e4); p_s = np.maximum(p_s, 1e4)
    S_s = np.maximum(S_s, 0.0); T_s = np.maximum(T_s, 0.0)

    # Decay transient data
    t_decay = decay_sol.t # Time points from decay simulation (s)
    phi_d = decay_sol.y[0*N : 1*N, :]
    n_d = decay_sol.y[1*N : 2*N, :]
    p_d = decay_sol.y[2*N : 3*N, :]
    S_d = decay_sol.y[3*N : 4*N, :]
    T_d = decay_sol.y[4*N : 5*N, :]
    # Apply floors for calculations/plotting
    n_d = np.maximum(n_d, 1e4); p_d = np.maximum(p_d, 1e4)
    S_d = np.maximum(S_d, 0.0); T_d = np.maximum(T_d, 0.0)

    # Calculate total luminescence vs time during decay
    # Luminescence is proportional to kr * S(x, t) integrated over device thickness
    S_d_nonneg = np.maximum(S_d, 0.0) # Ensure S is non-negative before calculation
    # Sum over spatial dimension (axis=0), multiply by grid spacing dL for integration approximation
    lum_vs_t_decay = np.sum(kr * S_d_nonneg, axis=0) * dL

    # --- Create Plots ---
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice plot style
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # 2 rows, 3 columns of subplots
    fig.suptitle(f"OLED Steady State @ {voltage:.2f}V & Transient Decay (GDM={params.get('use_gdm', 'N/A')})", fontsize=14)
    t_final_decay_us = t_decay[-1] * 1e6 # Final decay time in microseconds for axis limits

    # -- Row 1: Steady State Profiles --
    # Plot Potential (phi)
    ax = axes[0, 0]
    ax.plot(x_nm, phi_s, label='phi(x)')
    ax.set(xlabel='x (nm)', ylabel='Potential (V)', title=f'Potential @ {voltage:.1f}V')
    ax.legend(); ax.grid(True)

    # Plot Carrier Densities (n, p) - Log Scale
    ax = axes[0, 1]
    ax.plot(x_nm, n_s, label='n(x) - Electrons')
    ax.plot(x_nm, p_s, label='p(x) - Holes', ls='--') # Dashed line for holes
    ax.set(xlabel='x (nm)', ylabel='Density (m$^{-3}$)', title=f'Carriers @ {voltage:.1f}V', yscale='log')
    ax.legend(); ax.grid(True, which='both') # Grid for major and minor ticks on log scale

    # Plot Exciton Densities (S, T) - Log Scale
    ax = axes[0, 2]
    ax.plot(x_nm, S_s, label='S(x) - Singlets')
    ax.plot(x_nm, T_s, label='T(x) - Triplets', color='purple', ls='--') # Different color/style for triplets
    ax.set(xlabel='x (nm)', ylabel='Density (m$^{-3}$)', title=f'Excitons @ {voltage:.1f}V', yscale='log')
    ax.legend(); ax.grid(True, which='both')

    # -- Row 2: Transient Decay Plots --
    # Plot EL Decay - Linear Scale
    ax = axes[1, 0]
    ax.plot(t_decay * 1e6, lum_vs_t_decay, label='Emission (kr*S integrated)') # Convert time to microseconds
    ax.set(xlabel='Time after turn-off (µs)', ylabel='Emission (arb. units)', title='EL Decay (Linear Scale)')
    ax.legend(); ax.grid(True)

    # Plot EL Decay - Log Scale
    ax = axes[1, 1]
    ax.plot(t_decay * 1e6, lum_vs_t_decay, label='Emission (kr*S integrated)')
    ax.set(xlabel='Time after turn-off (µs)', ylabel='Emission (arb. units)', title='EL Decay (Log Scale)', yscale='log', xscale='log')
    ax.legend(); ax.grid(True, which='both')
    # Adjust y-limits for log plot to avoid overly large range if luminescence drops very low
    min_lum = np.min(lum_vs_t_decay[lum_vs_t_decay > 0]) if np.any(lum_vs_t_decay > 0) else 1e-20
    ax.set_ylim(bottom=max(min_lum * 0.1, 1e-15 * np.max(lum_vs_t_decay))) # Set a reasonable lower bound

    # Plot Total Exciton Population Decay - Log Scale
    ax = axes[1, 2]
    # Integrate S and T densities over space at each time point
    total_S_vs_t = np.sum(S_d * dL, axis=0)
    total_T_vs_t = np.sum(T_d * dL, axis=0)
    ax.plot(t_decay * 1e6, total_S_vs_t, label='Total Singlets (∫S dx)')
    ax.plot(t_decay * 1e6, total_T_vs_t, label='Total Triplets (∫T dx)', color='purple', ls='--')
    ax.set(xlabel='Time after turn-off (µs)', ylabel='Integrated Density (m$^{-2}$)', title='Total Exciton Population Decay', yscale='log')
    ax.legend(); ax.grid(True, which='both')

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()


# --- Main Entry Point ---
if __name__ == '__main__':
    # This block executes when the script is run directly
    # Run the main simulation function which handles both continuation and decay
    main_voltage_continuation_and_decay()
```

