##  Electronic coupling.

Electronic coupling is the quantum-mechanical interaction that mixes two charge-localized electronic states—typically a donor (D) and an acceptor (A)—and controls electron/energy transfer rates and the size of avoided crossings. In the most precise language, if one works in a diabatic basis of charge-localized states $\lvert D \rangle$ and $\lvert A \rangle$ (orthonormal by construction), the coupling is the off-diagonal Hamiltonian matrix element

$$
V \equiv H_{DA} = \langle D \lvert \hat{H} \rvert A \rangle.
$$


<img width="906" height="596" alt="image" src="https://github.com/user-attachments/assets/f3d1d35c-2557-4992-aec9-a63015fcaa2c" />

```text
Energy diagram for Electron Transfer including inner and outer sphere reorganization and electronic coupling:
The vertical axis is the free energy, and the horizontal axis is the "reaction coordinate"
– a simplified axis representing the motion of all the atomic nuclei (including solvent reorganization)
```


This definition is not merely notation: it is the unique scalar that sets the strength of state mixing and thus the splitting of adiabatic eigenenergies when the two diabatic surfaces come into resonance. For electron transfer, $V$ is the parameter that appears quadratically in nonadiabatic rate laws and linearly in the gap $2\lvert V \rvert$ at an avoided crossing for a symmetric system. Conceptually, diabatic states are constructed to have zero derivative (nonadiabatic) couplings, in contrast to adiabatic eigenstates of the Born–Oppenheimer Hamiltonian whose off-diagonal derivative couplings generally do not vanish; that is why diabatic states are preferred for defining $V$ unambiguously.


To see where the standard formulas come from, start with the two-state electronic Hamiltonian written in the $\{\lvert D \rangle, \lvert A \rangle\}$ basis:

$$
H = \begin{pmatrix}
\varepsilon_D & V \\
V & \varepsilon_A
\end{pmatrix}, \quad
\Delta \varepsilon \equiv \varepsilon_D - \varepsilon_A.
$$

Diagonalizing $H$ gives the adiabatic energies

$$
E_{\pm} = \frac{\varepsilon_D + \varepsilon_A}{2} \pm \sqrt{\left(\frac{\Delta \varepsilon}{2}\right)^2 + V^2}.
$$

The observable adiabatic splitting $\Delta E \equiv E_+ - E_-$ therefore satisfies

$$
\Delta E = 2\sqrt{\left(\frac{\Delta \varepsilon}{2}\right)^2 + V^2} \quad \Rightarrow \quad V = \frac{1}{2} \sqrt{(\Delta E)^2 - (\Delta \varepsilon)^2}.
$$

This exact identity underlies practical “adiabatic-to-diabatic” extraction of $V$: if one can compute or measure the two adiabatic energies $E_{\pm}$ and the diabatic bias $\Delta \varepsilon$, one can back out $V$ without any model-dependent fitting. In the symmetric case ($\varepsilon_D = \varepsilon_A$), the avoided-crossing gap is simply $2\lvert V \rvert$. This two-level diagonalization is standard material for two-state quantum systems. 

Real electronic structure calculations often begin from charge-localized (diabatic-like) determinants that are not exactly orthogonal. Then the generalized eigenproblem $Hc = ESc$ must be orthogonalized. Löwdin symmetric orthogonalization converts the pair $(H, S)$ to an orthonormal representation, in which the effective off-diagonal coupling—what one should report as “$V$”—is

$$
V_{\text{eff}} = \frac{H_{DA} - \frac{1}{2} S_{DA}(H_{DD} + H_{AA})}{\sqrt{1 - S_{DA}^2}},
$$

with $H_{ij} = \langle i \lvert \hat{H} \rvert j \rangle$ and $S_{ij} = \langle i \lvert j \rangle$. This relation follows directly from the Löwdin transformation and is widely implemented (sometimes called the “direct coupling” scheme using charge-localized states). It makes explicit how non-orthogonality modifies the naive $\langle D \lvert \hat{H} \rvert A \rangle$ matrix element.

There is a complementary, very practical route when your best-described states are adiabatic excited states from, e.g., TD-DFT or CI: the Generalized Mulliken–Hush (GMH) approach. GMH leverages how the dipole operator discriminates donor and acceptor character in a two-state subspace. If $\lvert 1 \rangle$ and $\lvert 2 \rangle$ are the two adiabatic states (with energy gap $\Delta E = E_2 - E_1$), $\Delta \mu = \langle 2 \lvert \hat{\mu} \rvert 2 \rangle - \langle 1 \lvert \hat{\mu} \rvert 1 \rangle$ is their difference dipole, and $\mu_{12} = \langle 1 \lvert \hat{\mu} \rvert 2 \rangle$ is the transition dipole (components taken along the donor–acceptor axis), then the GMH electronic coupling is

$$
V_{\text{GMH}} = \frac{\lvert \mu_{12} \rvert \, \Delta E}{\sqrt{(\Delta \mu)^2 + 4 \lvert \mu_{12} \rvert^2}}.
$$

This result is derived by requiring that the unitary rotation which “diabatizes” the two-state subspace simultaneously block-diagonalize the dipole operator (so that each diabatic state carries the donor or acceptor charge). GMH is nonperturbative within the subspace and has become a workhorse for coupling extraction when adiabatic data are more reliable than strictly diabatic states.

Once you have $V$, its physical role is clearest in dynamics and kinetics. In the weak-coupling, nonadiabatic regime (where hopping between diabats is rare compared with nuclear motion), Fermi’s golden rule gives the electron-transfer rate

$$
k_{\text{ET}} = \frac{2\pi}{\hbar} \lvert V \rvert^2 \, \rho_{\text{FC}}(T),
$$

where $\rho_{\text{FC}}(T)$ is the Franck–Condon weighted density of final states, encapsulating all nuclear reorganization statistics. Marcus theory provides the standard closed form for $\rho_{\text{FC}}$ under displaced harmonic parabolas, yielding

$$
k_{\text{ET}} = \frac{2\pi}{\hbar} \lvert V \rvert^2 \frac{1}{\sqrt{4\pi \lambda k_B T}} \exp\left[ -\frac{(\lambda + \Delta G^{\circ})^2}{4\lambda k_B T} \right],
$$

with reorganization energy $\lambda$ and driving force $\Delta G^{\circ}$. In the symmetric limit this expression smoothly connects to the Landau–Zener picture at an avoided crossing, where the probability of transition on a single passage depends exponentially on $V^2$. These relations make the status of $V$ as the kinetic “bottleneck parameter” completely explicit.




From a practical computational perspective, there are three tightly connected, verifiable routes that mirror what one would implement in code [https://github.com/ph7klw76/calculate_electronic_coupling/blob/main/electronic_coupling.md] and that all reduce to the same rigorous definitions above.

First, a direct-coupling (DC) evaluation: build charge-localized diabatic determinants (via constrained DFT, block-localized DFT, or fragment-orbital SCF), evaluate $H_{ij}$ and $S_{ij}$, and convert to $V_{\text{eff}}$ with the Löwdin formula quoted earlier. This is closest to the textbook definition

$$
V = \langle D \lvert \hat{H} \rvert A \rangle
$$

while correctly handling non-orthogonality.

Second, an adiabatic-splitting extraction: compute two adiabatic states as a function of a control coordinate that tunes donor–acceptor alignment (or compute site energies separately for $\Delta \varepsilon$) and use

$$
V = \frac{1}{2} \sqrt{(\Delta E)^2 - (\Delta \varepsilon)^2}.
$$

Third, a GMH calculation: take the adiabatic pair $\lvert 1 \rangle, \lvert 2 \rangle$, compute $\Delta \mu$ and $\mu_{12}$, and evaluate $V_{\text{GMH}}$ from the closed-form expression above.

Modern quantum-chemistry packages document and implement all three strategies, which provides a straightforward way to cross-validate a coupling obtained/


“Adiabatic” has a precise meaning in two common contexts, thermodynamics and quantum mechanics and the word signals “no exchange/mixing with the outside option” in each.

In thermodynamics, an adiabatic process is one with no heat exchanged with the surroundings: $\delta Q = 0$. If the process is also reversible (quasi-static with no dissipation), the entropy stays constant and you can show for an ideal gas that

$$
PV^{\gamma} = \text{const}
$$

with $\gamma = C_p / C_v$. Real, fast expansions or compressions can be adiabatic but not isentropic because they generate entropy internally; the defining feature remains the thermal isolation, not slowness.

In quantum mechanics and molecular physics, “adiabatic” describes evolution where the Hamiltonian changes slowly enough that a system prepared in an eigenstate remains in the corresponding instantaneous eigenstate. This is the adiabatic theorem: transitions to other eigenstates are suppressed when the timescale of change is long compared to $\hbar / \Delta E$, with $\Delta E$ the relevant energy gap. In chemistry we also talk about “adiabatic electronic states” (the Born–Oppenheimer picture): for each fixed nuclear geometry $R$, you solve the electronic Schrödinger equation to get eigenstates and energies $E_i(R)$. Those energy functions are the adiabatic potential-energy surfaces. Motion strictly on one surface, without electronic transitions to others, is adiabatic nuclear dynamics. When the off-diagonal “derivative couplings” $\tau_{ij}(R) = \langle \phi_i(R) \lvert \nabla_R \phi_j(R) \rangle$ are negligible, surfaces are effectively decoupled and the adiabatic approximation holds; when they are large—near avoided crossings or conical intersections—transitions become likely and the dynamics are nonadiabatic.

In short, adiabatic means “no exchange”: no heat with the environment in thermodynamics, and no population transfer between instantaneous eigenstates in quantum dynamics. In our previous discussion about electronic coupling, “adiabatic energies” were the eigenvalues $E_{\pm}$ obtained after diagonalizing the two-state Hamiltonian at a fixed geometry; their gap tells you how strongly the underlying diabatic (charge-localized) states are mixed.

