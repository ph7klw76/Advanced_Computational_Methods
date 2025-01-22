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

$$
\frac{\partial n}{\partial t} = \frac{1}{q} \frac{\partial J_n}{\partial x} - R_\text{tot}(n,p) + \dots \tag{1}
$$

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

$$
\frac{\partial p}{\partial t} = -\frac{1}{q} \frac{\partial J_p}{\partial x} - R_\text{tot}(n,p) + \dots \tag{2}
$$

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

$$
\frac{\partial S}{\partial t} = R_\text{gen}(S)(n,p) - \Gamma_\text{loss}(S,n,p) + \dots \tag{3}
$$

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


