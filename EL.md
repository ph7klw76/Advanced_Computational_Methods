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
