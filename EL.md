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

###############################################################################
# 1) HELPER FUNCTIONS FOR INITIAL CONDITIONS AND BOUNDARY CONDITIONS
###############################################################################

def build_initial_condition(N):
    """
    Define initial guesses for [phi, n, p, S].
    We'll start with zero potential, low carrier densities, and no excitons.
    """
    phi0 = np.zeros(N)
    n0   = np.full(N, 1e12)   # small background electron density
    p0   = np.full(N, 1e12)   # small background hole density
    S0   = np.zeros(N)
    return np.concatenate([phi0, n0, p0, S0])

def ramped_drive_voltage(t, params):
    """
    Returns the device boundary voltage at x=0 (anode) as a function of time.
    - We ramp up from 0 to V_drive over time t_ramp
    - Then hold that value until t_off
    - After t_off, we drop to 0
    """
    V_drive   = params['V_drive']     # final drive voltage
    t_ramp    = params['t_ramp']      # ramp time
    t_off     = params['t_off']       # turn-off time

    if t < t_ramp:
        # ramp from 0 to V_drive linearly
        return V_drive * (t / t_ramp)
    elif t < t_off:
        # hold at V_drive
        return V_drive
    else:
        # after t_off, set to 0
        return 0.0

def apply_bc_phi(phi, t, params):
    """
    Dirichlet boundary conditions for phi at x=0 and x=d.
    - x=0 : phi(0) = ramped_drive_voltage(t)
    - x=N-1 : phi(N-1) = 0 (cathode reference)
    """
    V_anode = ramped_drive_voltage(t, params)
    phi[0] = V_anode
    phi[-1] = 0.0

def apply_bc_carriers(n, p, t, params):
    """
    Simple 'ohmic-like' boundary conditions:
    - At x=0 (anode): pinned hole density => p(0) = p_anode_bc
    - At x=N-1 (cathode): pinned electron density => n(N-1) = n_cathode_bc
    """
    p_anode_bc    = params['p_anode_bc']
    n_cathode_bc  = params['n_cathode_bc']

    p[0]    = p_anode_bc
    n[-1]   = n_cathode_bc

def apply_bc_excitons(S, t, params):
    """
    Excitons vanish at both electrodes: S(0)=0, S(N-1)=0.
    """
    S[0]    = 0.0
    S[-1]   = 0.0

###############################################################################
# 2) MAIN PDE (METHOD-OF-LINES) FUNCTION
###############################################################################

def ddp_equations(t, y, params):
    """
    Time-derivatives for [phi, n, p, S] in a 1D device using finite differences.

    y is a 1D array: [phi(0..N-1), n(0..N-1), p(0..N-1), S(0..N-1)].
    We'll reshape them internally, then compute derivatives.

    Poisson eqn is solved in a 'relaxed' manner:
      eps_phi * dphi/dt = - (d^2 phi/dx^2) - q/eps (p - n)

    Electron & Hole continuity eqn w/ drift-diffusion.
    Singlet exciton continuity eqn w/ optional diffusion, generation, decay.
    """
    # Unpack essential parameters
    N      = params['N']
    dL     = params['dL']
    eps    = params['eps']
    q      = params['q']

    mu_n   = params['mu_n']
    mu_p   = params['mu_p']
    D_n    = params['D_n']
    D_p    = params['D_p']

    gamma  = params['gamma']
    eta_exc= params['eta_exc']
    kr     = params['kr']
    knr    = params['knr']
    Ds     = params['Ds']
    eps_phi= params['eps_phi']

    # Reshape state variables
    phi = y[0:N]
    n   = y[N:2*N]
    p   = y[2*N:3*N]
    S   = y[3*N:4*N]

    # Prepare arrays for derivatives
    dphi_dt = np.zeros(N)
    dn_dt   = np.zeros(N)
    dp_dt   = np.zeros(N)
    dS_dt   = np.zeros(N)

    #-----------------------------------------------------------------------
    # (1) Poisson eq. with artificial time relaxation:
    #     eps_phi * dphi/dt = - d2phi/dx^2 - (q/eps) (p - n)
    for i in range(1, N-1):
        d2phi_dx2 = (phi[i+1] - 2*phi[i] + phi[i-1]) / (dL**2)
        charge_term = (q/eps)*(p[i] - n[i])
        dphi_dt[i] = -(d2phi_dx2 + charge_term) / eps_phi

    #-----------------------------------------------------------------------
    # (2) Compute fluxes for electrons & holes
    def flux_n(i):
        """
        Electron flux at boundary 'i' between cell i-1 and i.
        Jn = q [ mu_n * n_face * E_face - D_n * grad_n ]
        E_face = -(phi[i] - phi[i-1])/dL
        n_face = 0.5*(n[i] + n[i-1])
        grad_n = (n[i] - n[i-1])/dL
        """
        E_face = -(phi[i] - phi[i-1])/dL
        n_face = 0.5*(n[i] + n[i-1])
        grad_n = (n[i] - n[i-1]) / dL
        return q*(mu_n*n_face*E_face - D_n*grad_n)

    def flux_p(i):
        """
        Hole flux at boundary 'i' between cell i-1 and i.
        Jp = q [ - mu_p * p_face * E_face - D_p * grad_p ]
        E_face = -(phi[i] - phi[i-1])/dL
        p_face = 0.5*(p[i] + p[i-1])
        grad_p = (p[i] - p[i-1])/dL
        """
        E_face = -(phi[i] - phi[i-1])/dL
        p_face = 0.5*(p[i] + p[i-1])
        grad_p = (p[i] - p[i-1]) / dL
        return q*(-mu_p*p_face*E_face - D_p*grad_p)

    for i in range(1, N-1):
        # electron flux at left & right boundaries
        Jn_left  = flux_n(i)
        Jn_right = flux_n(i+1)
        dn_dx    = (Jn_right - Jn_left)/(dL*q)

        # hole flux
        Jp_left  = flux_p(i)
        Jp_right = flux_p(i+1)
        dp_dx    = (Jp_right - Jp_left)/(dL*q)

        # Bimolecular recombination
        recomb   = gamma*n[i]*p[i]

        dn_dt[i] = -dn_dx - recomb
        dp_dt[i] = -dp_dx - recomb

    #-----------------------------------------------------------------------
    # (3) Exciton PDE
    # dS/dt = Ds d2S/dx^2 + eta_exc*gamma*n*p - (kr+knr)*S
    for i in range(1, N-1):
        d2S_dx2 = (S[i+1] - 2*S[i] + S[i-1]) / (dL**2)
        gen     = eta_exc * gamma * n[i] * p[i]
        loss    = (kr + knr)*S[i]
        dS_dt[i] = Ds*d2S_dx2 + gen - loss

    #-----------------------------------------------------------------------
    # (4) Apply boundary conditions
    # Potential:
    apply_bc_phi(phi, t, params)
    dphi_dt[0]   = 0.0
    dphi_dt[-1]  = 0.0

    # Carriers:
    apply_bc_carriers(n, p, t, params)
    dn_dt[0]     = 0.0
    dn_dt[-1]    = 0.0
    dp_dt[0]     = 0.0
    dp_dt[-1]    = 0.0

    # Excitons:
    apply_bc_excitons(S, t, params)
    dS_dt[0]     = 0.0
    dS_dt[-1]    = 0.0

    # Pack up derivatives
    return np.concatenate([dphi_dt, dn_dt, dp_dt, dS_dt])

###############################################################################
# 3) MAIN SIMULATION FUNCTION
###############################################################################

def run_1d_oled_sim():
    #-----------------------
    # Define smaller device / mesh
    N = 51             # number of mesh points (fewer => less stiff, easier to solve)
    d = 50e-9          # 50 nm thickness (reduced from 100 nm)
    dL = d/(N-1)       # spatial step

    # Permittivity
    eps0 = 8.854e-12
    eps_r = 3.0
    eps = eps0*eps_r

    # Basic constants
    q   = 1.602e-19
    T   = 300.0        # temperature K
    kB  = 1.38e-23
    Vth = kB*T / q      # ~ 0.0259 eV at room temperature

    # Mobility
    mu_n = 1e-8   # m^2/Vs
    mu_p = 1e-8   # m^2/Vs
    # Diffusion (Einstein relation)
    D_n  = mu_n*Vth
    D_p  = mu_p*Vth

    # Recombination & exciton parameters
    gamma   = 1e-17     # reduce from 1e-16 to 1e-17
    eta_exc = 0.25
    kr      = 1e7
    knr     = 1e6
    Ds      = 1e-5      # exciton diffusion coefficient (slightly larger for test)

    # "Artificial" time-relaxation factor for Poisson eq
    eps_phi = 1e-8      # significantly increased from 1e-12

    # Drive voltage & time settings
    V_drive = 5.0
    t_ramp  = 1e-7   # ramp from 0 -> 5V over 0.1 microseconds
    t_off   = 3e-7   # after 0.3 microseconds, go to 0

    # Time range for the simulation
    t_end   = 2e-6   # total 2 microseconds
    # We'll pick 2001 points for plotting
    t_eval  = np.linspace(0, t_end, 2001)

    # Boundary pinned densities (lower => less extreme doping at contacts)
    n_cathode_bc = 1e14
    p_anode_bc   = 1e14

    # Pack parameters
    params = {
        'N': N,
        'dL': dL,
        'eps': eps,
        'q': q,
        'mu_n': mu_n,
        'mu_p': mu_p,
        'D_n': D_n,
        'D_p': D_p,
        'gamma': gamma,
        'eta_exc': eta_exc,
        'kr': kr,
        'knr': knr,
        'Ds': Ds,
        'eps_phi': eps_phi,

        'V_drive': V_drive,
        't_ramp':  t_ramp,
        't_off':   t_off,

        'n_cathode_bc': n_cathode_bc,
        'p_anode_bc':   p_anode_bc
    }

    # Build initial condition
    y0 = build_initial_condition(N)

    # Solve with solve_ivp, using BDF for stiff systems
    sol = solve_ivp(
        fun=lambda tt, yy: ddp_equations(tt, yy, params),
        t_span=(0, t_end),
        y0=y0,
        t_eval=t_eval,
        method='BDF',  
        rtol=1e-3,
        atol=1e-6,
        max_step=1e-8  # allow the solver to refine steps as needed
    )

    # Check success
    if not sol.success:
        print("Warning: ODE solver did not converge or ended early!")
        print(sol.message)

    # Ensure we have at least 2 points to plot
    if sol.t.size < 2:
        raise RuntimeError("Solver returned < 2 time points; cannot plot a decay.")

    # Extract solutions
    phi_sol = sol.y[0:N, :]
    n_sol   = sol.y[N:2*N, :]
    p_sol   = sol.y[2*N:3*N, :]
    S_sol   = sol.y[3*N:4*N, :]

    return sol.t, phi_sol, n_sol, p_sol, S_sol, params

###############################################################################
# 4) MAIN FUNCTION FOR RUNNING AND PLOTTING
###############################################################################

def main():
    # Run simulation
    t, phi_sol, n_sol, p_sol, S_sol, params = run_1d_oled_sim()
    N = params['N']
    x = np.linspace(0, params['dL']*(N-1), N)

    # Compute total emission:
    # L(t) ~ \int kr * S(x,t) dx (in 1D, sum over mesh cells)
    kr = params['kr']
    dL = params['dL']
    lum_vs_t = []
    for it in range(len(t)):
        S_t = S_sol[:, it]
        # local rate = kr * S(x), integrate across domain
        total_rate = np.sum(kr * S_t)*dL
        lum_vs_t.append(total_rate)
    lum_vs_t = np.array(lum_vs_t)

    # Plot results
    plt.figure(figsize=(10, 8))

    # (1) Potential at final time
    plt.subplot(2, 2, 1)
    plt.plot(x*1e9, phi_sol[:, -1], label='phi(x) final')
    plt.xlabel('x (nm)')
    plt.ylabel('Potential (V)')
    plt.title('Potential Distribution (final time)')
    plt.legend()

    # (2) Carrier distribution at final time
    plt.subplot(2, 2, 2)
    plt.plot(x*1e9, n_sol[:, -1], label='n(x)')
    plt.plot(x*1e9, p_sol[:, -1], label='p(x)')
    plt.yscale('log')
    plt.xlabel('x (nm)')
    plt.ylabel('Carrier Density (m^-3)')
    plt.title('Carriers at final time')
    plt.legend()

    # (3) Exciton distribution at final time
    plt.subplot(2, 2, 3)
    plt.plot(x*1e9, S_sol[:, -1], label='S(x)')
    plt.yscale('log')
    plt.xlabel('x (nm)')
    plt.ylabel('Exciton Density (m^-3)')
    plt.title('Excitons at final time')
    plt.legend()

    # (4) Light emission vs time
    plt.subplot(2, 2, 4)
    plt.plot(t*1e6, lum_vs_t, label='Emission vs time')
    plt.xlabel('Time (µs)')
    plt.ylabel('Emission (arb. units)')
    plt.title('Electroluminescence Decay')
    plt.legend()

    plt.tight_layout()
    plt.show()

###############################################################################
# 5) EXECUTE
###############################################################################

if __name__ == '__main__':
    main()
```
