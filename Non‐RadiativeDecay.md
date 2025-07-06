# A Quantum‐Mechanical Portrait of Non‐Radiative Decay

In an emissive molecule, excited‐state relaxation competes between radiative pathways (fluorescence or phosphorescence) and non‐radiative channels (internal conversion, IC, and intersystem crossing, ISC). From a device‐engineering standpoint, suppressing non‐radiative loss

$$
k_{nr} = k_{IC} + k_{ISC}
$$

is as crucial as maximizing the radiative rate, $k_r$, since the photoluminescence quantum yield

$$
\Phi_{PL} = \frac{k_r}{k_r + k_{nr}}
$$

directly maps loss into light‐emission efficiency.

## The Born–Oppenheimer Basis and Fermi’s Golden Rule

Under the Born–Oppenheimer approximation, electronic states $s$ (e.g.\ S₁) and $l$ (S₀) are each described by multidimensional harmonic potentials in the nuclear coordinates $Q_j$. The non‐radiative transition rate from an initial vibronic level $\lvert s, v_s \rangle$ to a reservoir of final states $\{\lvert l, v_l \rangle\}$ follows directly from Fermi’s Golden Rule:

$$
k_{nr} = \frac{2\pi}{\hbar} \sum_{s,v_s} \sum_{l,v_l} p(s,v_s) \lvert V_{sv_s,lv_l} \rvert^2 \delta(E_{sv_s} - E_{lv_l}),
$$

where $V$ is the non‐adiabatic coupling operator and $p(s,v_s)$ the population of the initial vibronic state. This form makes crystal‐clear that two ingredients govern IC:

- **Electronic coupling strength** $\lvert V \rvert^2$, incorporating both vibronic and spin–orbit matrix elements.
- **Vibrational overlap and energy matching**, encoded by the Franck–Condon factors and the delta‐function.

## The Displaced‐Oscillator Model and Vibrational Overlap

Robert Englman and Joshua Jortner’s seminal treatment recasts the nuclear dynamics in terms of dimensionless normal modes $q_j$ displaced by amounts $A_j$ between the two electronic surfaces:

$$
E_s = \sum_j \frac{1}{2} \hbar\omega_j q_j^2,\quad
E_\ell = \sum_j \frac{1}{2} \hbar\omega_j (q_j - A_j)^2 + \Delta E - \sum_j \hbar\omega_j A_j^2,
$$

where $\Delta E$ is the adiabatic energy gap and $\hbar\omega_j$ each mode’s quantum. The Huang–Rhys factor

$$
S_j = A_j^2
$$

quantifies the dimensionless displacement, and the reorganisation energy per mode is

$$
\lambda_j = S_j \hbar\omega_j.
$$

Together, these define the shift and breadth of the Franck–Condon envelope.

## Generating‐Function Formalism and the Englman–Jortner Law

Summing over an arbitrary number of modes is rendered tractable by the generating‐function approach, which yields a closed‐form expression for the vibrational “density of states” and thus for the IC rate. In the limit of weak electronic coupling (small overall displacement $G = \sum_j S_j$), the IC rate simplifies to an energy‐gap law of the form:

$$
k_{IC} \propto \exp\left[ -\frac{(\Delta E - \lambda)^2}{4\lambda k_B T} \right],
$$

where $\lambda = \sum_j \lambda_j$. At low temperature this reduces to a super‐exponential dependence on the gap, dominated by the highest‐energy promoting mode.

## Why High $\omega_i$ and Small $S_i$ Stifle Internal Conversion

Two complementary quantum‐mechanical effects conspire to suppress IC when coupling is confined to stiff, high‐frequency modes of small displacement:

### Energy‐matching sparsity

The $\delta$‐function in Fermi’s rule demands $\Delta E = n\hbar\omega_i$ for some integer $n$. A high $\hbar\omega_i$ (e.g.\ 1 500 cm⁻¹ ≈ 0.19 eV) is a “coarse” packet of vibrational energy. Matching a typical 2 eV gap requires $n \approx 10$, and only those rare, high‐order multiphonon terms carry non‐zero Franck–Condon weight. All intermediate baths of vibronic levels lie “off‐resonance”.

### Franck–Condon envelope collapse

The probability of accessing the $n$-phonon sideband scales as

$$
\frac{e^{-S_i} S_i^n}{n!}.
$$

When $S_i \ll 1$, even the first sidebands (necessary to bridge a gap of one quantum) are heavily down‐weighted; higher $n$ terms are factorially suppressed. Equivalently, the total reorganisation energy

$$
\lambda_i = S_i \hbar\omega_i
$$

is tiny, making the energy‐gap law exponent large and $k_{IC}$ vanishingly small.

Together, these effects mean that pushing vibrational coupling into high‐$\omega$, low‐$S$ territory erects a twin barrier—both coarse energy‐step mismatch and vanishing overlap—rendering IC extremely inefficient.

## Heavy‐Atom ISC as a Parallel Loss Channel

In molecules bearing heavy atoms (e.g.\ bromine in EHBIPOBr), strong spin–orbit coupling opens rapid intersystem crossing:

$$
k_{ISC} \propto \lvert H_{SO} \rvert^2,
$$

often exceeding $k_{IC}$ by an order of magnitude or more. This additional non‐radiative funnel further erodes $\Phi_{PL}$ unless the heavy atom is explicitly exploited for phosphorescence or TADF.

## From Formalism to Molecular Design

The mathematical portrait delivered by Englman–Jortner provides clear design handles:

- **Raise $\omega_{prom}$**. Rigidify conjugated backbones to stiffen key vibrational modes above ∼1 200 cm⁻¹.
- **Minimise displacements $S$**. Spread the excitation over a larger molecular scaffold or reduce mode‐specific distortions.
- **Avoid gratuitous heavy atoms** when high fluorescence yield is the goal.
- **Embed in rigid hosts** to quench low‐frequency lattice‐like modes that otherwise proliferate IC channels.

By tuning $\{\omega_i, S_i\}$ and controlling spin–orbit pathways, chemists can dial $k_{nr}$ down by one to two orders of magnitude, translating directly into percentage‐point gains in $\Phi_{PL}$ and device efficiency.
