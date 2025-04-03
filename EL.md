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


code not fully functional
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import expit, erfc
import warnings
from tqdm.auto import tqdm # Import tqdm for progress bar
import multiprocessing # For parallel processing example
import time # For parameter sweep example timing
import os # For saving intermediate results

# --- Constants ---
Q = 1.602e-19; KB = 1.38e-23; EPS0 = 8.854e-12; HBAR = 1.054571817e-34

# --- Helper Functions (Bernoulli, GDM, Langevin - unchanged) ---
def bernoulli(x):
    x = np.asanyarray(x); b = np.zeros_like(x)
    small_mask = np.abs(x) < 1e-10; large_pos_mask = x > 700
    large_neg_mask = x < -700; valid_mask = ~(small_mask | large_pos_mask | large_neg_mask)
    b[small_mask] = 1.0 - 0.5 * x[small_mask] + x[small_mask]**2 / 12.0
    b[large_pos_mask] = x[large_pos_mask] * np.exp(-x[large_pos_mask])
    b[large_neg_mask] = -x[large_neg_mask]
    b[valid_mask] = x[valid_mask] / np.expm1(x[valid_mask])
    b[np.isnan(x)] = np.nan
    return b

def gaussian_disorder_mobility(E_abs, T, mu0, sigma_hop, C_gdm):
    kBT_eV = KB * T / Q;
    if kBT_eV < 1e-6: return np.zeros_like(E_abs) + 1e-18
    factor_T = (2.0 * sigma_hop / (3.0 * kBT_eV))**2
    E_safe = np.maximum(E_abs, 1e-10)
    mobility = mu0 * np.exp(-factor_T) * np.exp(C_gdm * np.sqrt(E_safe))
    return np.clip(mobility, 1e-18, 1e-1)

def calculate_langevin_gamma(mu_n_local, mu_p_local, eps):
    mu_n_safe = np.maximum(mu_n_local, 0.0); mu_p_safe = np.maximum(mu_p_local, 0.0)
    gamma_L = Q * (mu_n_safe + mu_p_safe) / eps
    return np.maximum(gamma_L, 0.0)

# --- Ramped Voltage (modified for continuation - unchanged) ---
def ramped_drive_voltage_continuation(t, params, V_start, V_end):
    t_ramp = params['t_ramp_step']; t_off = params.get('t_off', np.inf)
    if t < 0: return V_start
    elif t < t_ramp: return V_start + (V_end - V_start) * (t / t_ramp)
    elif t < t_off: return V_end
    else: return 0.0

# --- Initial Condition (Smoothed - unchanged) ---
def build_initial_condition_smoothed(N, params):
    phi0 = np.zeros(N); n0_bg = params.get('n0_background', 1e6)
    p0_bg = params.get('p0_background', 1e6); n_cathode_target = params['n_cathode_bc']
    p_anode_target = params['p_anode_bc']; n0 = np.full(N, n0_bg); p0 = np.full(N, p0_bg)
    num_smooth_points = min(5, N // 4)
    if num_smooth_points > 1:
        p_smooth_indices = np.arange(num_smooth_points)
        log_p_target = np.log10(max(p_anode_target, 1e-10))
        log_p_bg = np.log10(max(p0_bg, 1e-10))
        p0[p_smooth_indices] = 10**(np.linspace(log_p_target, log_p_bg, num_smooth_points)[::-1])
        p0[0] = p_anode_target
        n_smooth_indices = np.arange(N - num_smooth_points, N)
        log_n_bg = np.log10(max(n0_bg, 1e-10))
        log_n_target = np.log10(max(n_cathode_target, 1e-10))
        n0[n_smooth_indices] = 10**(np.linspace(log_n_bg, log_n_target, num_smooth_points))
        n0[-1] = n_cathode_target
    else:
         n0[-1] = n_cathode_target; p0[0] = p_anode_target
    S0 = np.zeros(N); T0 = np.zeros(N)
    return np.concatenate([phi0, n0, p0, S0, T0])

# --- Main PDE System Definition (unchanged) ---
def oled_1d_dde_equations_advanced_cont(t, y, params, V_start, V_end):
    # --- Unpack Parameters ---
    # Use .get() for safety, although they should be there now
    N = params.get('N'); dL = params.get('dL'); eps = params.get('eps'); q = params.get('q')
    T_dev = params.get('T_dev'); Vth = params.get('Vth')
    # Handle potential missing keys more gracefully during debugging if needed
    if any(v is None for v in [N, dL, eps, q, T_dev, Vth]):
        raise ValueError("Essential parameters missing from params dict in ODE function")

    mu0_n, sigma_hop_n, C_gdm_n = params['gdm_n']
    mu0_p, sigma_hop_p, C_gdm_p = params['gdm_p']
    use_gdm = params.get('use_gdm', True)
    gamma_reduction = params['gamma_reduction']
    C_n_aug, C_p_aug = params.get('auger_coeffs', (0.0, 0.0))
    spin_fraction_singlet = params['spin_fraction_singlet']
    kr = params['kr']; knr = params['knr']; k_isc = params['k_isc']
    k_risc = params.get('k_risc', 0.0)
    k_tph = params.get('k_tph', 0.0); k_tnr = params['k_tnr']
    Ds = params['Ds']; Dt = params['Dt']
    kq_sn = params['kq_sn']; kq_sp = params['kq_sp']
    kq_tn = params['kq_tn']; kq_tp = params['kq_tp']
    k_ss = params['k_ss']; k_tta = params['k_tta']
    n_cathode_bc_val = params['n_cathode_bc']
    p_anode_bc_val = params['p_anode_bc']
    eps_phi = params['eps_phi']

    # --- Reshape State Vector ---
    phi = y[0*N : 1*N]; n = y[1*N : 2*N]; p = y[2*N : 3*N]
    S = y[3*N : 4*N]; T = y[4*N : 5*N]
    n = np.maximum(n, 1e4); p = np.maximum(p, 1e4)
    S = np.maximum(S, 0.0); T = np.maximum(T, 0.0)

    dphi_dt=np.zeros(N); dn_dt=np.zeros(N); dp_dt=np.zeros(N); dS_dt=np.zeros(N); dT_dt=np.zeros(N)

    # --- Apply Boundary Conditions to State ---
    phi[0] = ramped_drive_voltage_continuation(t, params, V_start, V_end)
    phi[-1] = 0.0
    n[-1] = n_cathode_bc_val; p[0] = p_anode_bc_val
    S[0] = S[-1] = 0.0; T[0] = T[-1] = 0.0

    # --- Calculate Spatially Varying Parameters ---
    E = np.zeros(N); E[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dL)
    E[0] = -(phi[1] - phi[0]) / dL; E[-1] = -(phi[-1] - phi[-2]) / dL
    E_abs = np.abs(E)
    if use_gdm:
        mu_n_local = gaussian_disorder_mobility(E_abs, T_dev, mu0_n, sigma_hop_n, C_gdm_n)
        mu_p_local = gaussian_disorder_mobility(E_abs, T_dev, mu0_p, sigma_hop_p, C_gdm_p)
    else:
        mu_n_local = np.maximum(np.full(N, mu0_n), 1e-18)
        mu_p_local = np.maximum(np.full(N, mu0_p), 1e-18)
    Dn_local = mu_n_local * Vth; Dp_local = mu_p_local * Vth

    # --- Finite Difference Calculations ---
    dV_int = phi[:-1] - phi[1:]
    x_int = np.clip(dV_int / Vth, -500, 500)
    B_plus = bernoulli(x_int); B_minus = bernoulli(-x_int)
    Dn_int = 0.5 * (Dn_local[:-1] + Dn_local[1:])
    Dp_int = 0.5 * (Dp_local[:-1] + Dp_local[1:])
    Jn_int = (q * Dn_int / dL) * ( B_minus * n[1:] - B_plus * n[:-1] )
    Jp_int = (q * Dp_int / dL) * ( B_plus * p[:-1] - B_minus * p[1:] )

    # --- Calculate Divergences ---
    div_Jn = (Jn_int[1:] - Jn_int[:-1]) / dL; div_Jp = (Jp_int[1:] - Jp_int[:-1]) / dL

    # --- Calculate Rates ---
    n_int = n[1:-1]; p_int = p[1:-1]; S_int = S[1:-1]; T_int = T[1:-1]
    gamma_L_local = calculate_langevin_gamma(mu_n_local[1:-1], mu_p_local[1:-1], eps)
    R_bimol = gamma_reduction * gamma_L_local * (n_int * p_int); R_bimol = np.maximum(R_bimol, 0.0)
    R_auger = C_n_aug * n_int**2 * p_int + C_p_aug * n_int * p_int**2; R_auger = np.maximum(R_auger, 0.0)
    R_total_carrier_loss = R_bimol + R_auger
    G_S = spin_fraction_singlet * R_bimol; G_T = (1.0 - spin_fraction_singlet) * R_bimol
    d2S_dx2 = (S[2:] - 2*S_int + S[:-2]) / (dL**2); Diffusion_S = Ds * d2S_dx2
    d2T_dx2 = (T[2:] - 2*T_int + T[:-2]) / (dL**2); Diffusion_T = Dt * d2T_dx2
    Loss_S_decay = (kr + knr) * S_int; Loss_S_isc = k_isc * S_int
    Loss_S_quench_n = kq_sn * n_int * S_int; Loss_S_quench_p = kq_sp * p_int * S_int
    Loss_S_ssa = k_ss * S_int**2; Gain_S_risc = k_risc * T_int
    Gamma_loss_S_net = Loss_S_decay + Loss_S_isc + Loss_S_quench_n + Loss_S_quench_p + Loss_S_ssa - Gain_S_risc
    Loss_T_decay = (k_tph + k_tnr) * T_int; Loss_T_risc = k_risc * T_int
    Loss_T_quench_n = kq_tn * n_int * T_int; Loss_T_quench_p = kq_tp * p_int * T_int
    Loss_T_tta = k_tta * T_int**2; Gain_T_isc = Loss_S_isc
    Gamma_loss_T_net = Loss_T_decay + Loss_T_risc + Loss_T_quench_n + Loss_T_quench_p + Loss_T_tta - Gain_T_isc

    # --- Assemble Derivatives ---
    dn_dt[1:-1] = (1.0/q) * div_Jn - R_total_carrier_loss
    dp_dt[1:-1] = -(1.0/q) * div_Jp - R_total_carrier_loss
    dS_dt[1:-1] = G_S - Gamma_loss_S_net + Diffusion_S
    dT_dt[1:-1] = G_T - Gamma_loss_T_net + Diffusion_T

    # --- Poisson's Equation ---
    d2phi_dx2 = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / (dL**2)
    charge_term = q * (p[1:-1] - n[1:-1])
    dphi_dt[1:-1] = (eps * d2phi_dx2 - charge_term) / eps_phi

    # --- Fix Derivatives at Boundaries ---
    dphi_dt[0] = dphi_dt[-1] = 0.0; dn_dt[0] = dn_dt[-1] = 0.0
    dp_dt[0] = dp_dt[-1] = 0.0; dS_dt[0] = dS_dt[-1] = 0.0
    dT_dt[0] = dT_dt[-1] = 0.0

    dydt = np.concatenate([dphi_dt, dn_dt, dp_dt, dS_dt, dT_dt])
    return dydt

# --- Simulation Runner (CORRECTED derived parameter calculation) ---

def run_1d_oled_sim_step(params, V_start, V_end, y0_input=None, solver_method='BDF', save_filename=None):
    """
    Runs a single voltage step in the continuation method.
    Accepts an initial state y0_input. CORRECTED DERIVED PARAMS.
    """
    # !! **** CORRECTION: Calculate derived params here **** !!
    # Make a copy to avoid modifying the dictionary passed to subsequent steps if run in sequence non-parallel
    params = params.copy()
    params['dL'] = params['d'] / (params['N'] - 1)
    params['eps'] = params['eps_r'] * EPS0
    params['q'] = Q
    params['Vth'] = KB * params['T_dev'] / Q
    # Now dL, eps, q, Vth will be in the params dict passed to the ODE function
    # !! **************************************************** !!

    N = params['N']
    # --- Initial Condition ---
    if y0_input is None:
        # print("Building initial condition from scratch...") # Reduce verbosity
        y0 = build_initial_condition_smoothed(N, params)
        if V_start != 0.0:
             y0[0] = V_start
    else:
        # print(f"Using provided initial state y0_input.") # Reduce verbosity
        y0 = y0_input.copy()
        y0[0] = V_start
        y0[N-1] = 0.0

    # --- Time Stepping for this step ---
    t_ramp = params['t_ramp_step']
    stabilization_time = params.get('t_stabilize_step', 5 * t_ramp)
    t_span_step = (0, t_ramp + stabilization_time)
    num_t_points_step = params.get('num_t_points_step', 101)
    t_eval_step = np.linspace(t_span_step[0], t_span_step[1], num_t_points_step)
    max_step_val = params.get('max_step_init', np.inf)

    # print(f"Starting ODE step: {V_start:.2f}V -> {V_end:.2f}V (Ramp: {t_ramp:.1e}s, Stab: {stabilization_time:.1e}s)...") # Reduce verbosity

    sol = None
    # Wrap with tqdm - note the desc string format needs fixing if V_start/V_end are not defined here
    with tqdm(total=1, desc=f"Step {V_start:.2f}V->{V_end:.2f}V") as pbar:
        try:
            sol = solve_ivp(
                fun=oled_1d_dde_equations_advanced_cont,
                t_span=t_span_step,
                y0=y0,
                method=solver_method,
                t_eval=t_eval_step,
                args=(params, V_start, V_end), # Pass V_start, V_end
                rtol=params['rtol'],
                atol=params['atol'],
                max_step=max_step_val
            )
            pbar.update(1)
        except Exception as e:
            pbar.update(1)
            print(f"\nError during solve_ivp step {V_start:.2f}V->{V_end:.2f}V: {e}")
            # Decide if you want to raise or allow continuation loop to handle failure
            raise # Re-raise to stop the continuation loop on first error

    # print(f"ODE step finished: {sol.message if sol else 'Error'}") # Reduce verbosity
    if sol and not sol.success:
        # Warning is okay, but raise error if no steps were taken
        warnings.warn(f"ODE step {V_start:.2f}V->{V_end:.2f}V finished with warning: {sol.message}")
        if sol.t.size < 2 :
             raise RuntimeError(f"Solver failed to take any steps for {V_start:.2f}V->{V_end:.2f}V.")
    if not sol: # Should not happen if exception is raised, but for safety
         raise RuntimeError(f"Solver failed completely for {V_start:.2f}V->{V_end:.2f}V (no solution object).")


    final_state = sol.y[:, -1]

    if save_filename:
        try:
            np.savez_compressed(save_filename, y_final=final_state, V_end=V_end, params=params)
            # print(f"Saved final state for {V_end:.2f}V to {save_filename}") # Reduce verbosity
        except Exception as e:
            warnings.warn(f"Could not save intermediate state {save_filename}: {e}")

    # print(f"Step {V_start:.2f}V->{V_end:.2f}V completed.") # Reduce verbosity
    return final_state

# --- Main Execution (Voltage Continuation Loop - unchanged logic) ---
def main_voltage_continuation():
    print("--- Starting OLED Simulation with Voltage Continuation ---")
    base_params = {
        'N': 101, 'd': 100e-9, 'eps_r': 3.0, 'T_dev': 300.0,
        'gdm_n': (1e-7, 0.08, 3e-4), 'gdm_p': (1e-7, 0.08, 3e-4), 'use_gdm': True,
        'gamma_reduction': 0.1, 'auger_coeffs': (0.0, 0.0),
        'spin_fraction_singlet': 0.25, 'kr': 1e7, 'knr': 5e6, 'k_isc': 2e7, 'k_risc': 0.0,
        'k_tph': 0.0, 'k_tnr': 1e5, 'Ds': 1e-9, 'Dt': 1e-10,
        'kq_sn': 1e-19, 'kq_sp': 1e-19, 'kq_tn': 1e-19, 'kq_tp': 1e-19,
        'k_ss': 1e-18, 'k_tta': 5e-18,
        'n_cathode_bc': 1e23, 'p_anode_bc': 1e23,
        'n0_background': 1e6, 'p0_background': 1e6,
        'eps_phi': 1e-14, 'rtol': 1e-4, 'atol': 1e-7,
        't_ramp_step': 1e-7, 't_stabilize_step': 5e-7,
        'num_t_points_step': 101, 'save_intermediate': True,
        'solver_method': 'BDF',
    }
    voltage_steps = np.linspace(0.0, 5.0, 11) # 0.0, 0.5, ..., 5.0
    print(f"Voltage steps: {voltage_steps}")
    current_y = None
    final_states = {}
    save_dir = "oled_continuation_states" # Directory to save states
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist

    try:
        for i in range(len(voltage_steps) - 1):
            V_start = voltage_steps[i]
            V_end = voltage_steps[i+1]
            save_fname = os.path.join(save_dir, f"oled_state_V{V_end:.2f}.npz") if base_params['save_intermediate'] else None
            load_success = False
            if save_fname and os.path.exists(save_fname):
                try:
                    data = np.load(save_fname, allow_pickle=True)
                    if np.isclose(data['V_end'], V_end):
                        current_y = data['y_final']
                        print(f"Loaded previous state for {V_end:.2f}V from {save_fname}")
                        load_success = True
                    else: print(f"Saved state V mismatch ({data['V_end']:.2f} != {V_end:.2f}). Recalculating.")
                except Exception as load_err: print(f"Error loading {save_fname}: {load_err}. Recalculating.")

            if not load_success:
                 current_y = run_1d_oled_sim_step(base_params, V_start, V_end,
                                                  y0_input=current_y,
                                                  solver_method=base_params['solver_method'],
                                                  save_filename=save_fname)
            final_states[V_end] = current_y

        print("\n--- Voltage Continuation Finished Successfully ---")
        target_voltage = voltage_steps[-1]
        if target_voltage in final_states:
            print("Plotting final state from continuation...")
            plot_final_state(final_states[target_voltage], target_voltage, base_params)
        else: print("Final state not available for plotting.")

    except Exception as e:
        print(f"\n--- An error occurred during Voltage Continuation ---")
        print(f"Error Type: {type(e).__name__}"); print(f"Error Details: {e}")
        import traceback; print("\n--- Traceback ---"); traceback.print_exc(); print("-----------------\n")


# --- Plotting Helper (unchanged) ---
def plot_final_state(y_final, voltage, params):
    N = params['N']; dL = params['dL']; kr = params['kr']; d = params['d']
    x_nm = np.linspace(0, d * 1e9, N)
    phi = y_final[0*N : 1*N]; n = y_final[1*N : 2*N]; p = y_final[2*N : 3*N]
    S = y_final[3*N : 4*N]; T = y_final[4*N : 5*N]
    n = np.maximum(n, 1e4); p = np.maximum(p, 1e4)
    S = np.maximum(S, 0.0); T = np.maximum(T, 0.0)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Final State @ {voltage:.2f}V (GDM={params['use_gdm']})", fontsize=14)

    ax = axes[0]; ax.plot(x_nm, phi, label=f'phi(x)')
    ax.set(xlabel='Position x (nm)', ylabel='Potential (V)', title='Potential'); ax.legend(); ax.grid(True)
    ax = axes[1]; ax.plot(x_nm, n, label=f'n(x)')
    ax.plot(x_nm, p, label=f'p(x)', ls='--')
    ax.set(xlabel='Position x (nm)', ylabel='Density (m$^{-3}$)', title='Carriers', yscale='log'); ax.legend(); ax.grid(True, which='both')
    ax = axes[2]; ax.plot(x_nm, S, label=f'S(x)')
    ax.plot(x_nm, T, label=f'T(x)', color='purple', ls='--')
    ax.set(xlabel='Position x (nm)', ylabel='Density (m$^{-3}$)', title='Excitons', yscale='log'); ax.legend(); ax.grid(True, which='both')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


# --- Main Entry Point ---
if __name__ == '__main__':
    # Run the voltage continuation simulation
    main_voltage_continuation()
```
