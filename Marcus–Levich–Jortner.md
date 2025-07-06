# Electron transfer (ET)

Electron transfer (ET) lies at the heart of countless chemical, biological and emerging quantum‐information processes. The Marcus–Levich–Jortner (MLJ) formalism provides a unified, non‐adiabatic rate theory that captures both the classical reorganization of an environment and the quantized nature of high‐frequency molecular vibrations. In what follows, we develop the MLJ expression in full detail, derive its purely quantum vibronic limit (Jortner’s rate), and then discuss how this framework informs the design and understanding of solid‐state quantum computing platforms.

## From Marcus to Marcus–Levich–Jortner

In classical Marcus theory the ET rate between a donor and acceptor is expressed by

$$
k_M = \frac{2\pi}{\hbar} |V|^2 \frac{1}{\sqrt{4\pi\lambda k_B T}} \exp\left[ -\frac{(\Delta G + \lambda)^2}{4\lambda k_B T} \right],
$$

where $V$ is the electronic coupling, $\lambda$ the total reorganization energy of the (classical) nuclear modes, $\Delta G$ the Gibbs free‐energy change, $k_B$ the Boltzmann constant and $T$ temperature. This formula follows from Fermi’s Golden Rule when all nuclear degrees of freedom are treated as a classical harmonic bath.

Levich’s insight was that intramolecular vibrations whose quanta $\hbar\omega$ greatly exceed $k_B T$ cannot be thermally populated and must be quantized. Jortner’s 1976 synthesis begins by partitioning the nuclear coordinates into a low‐frequency “outer‐sphere” bath (bulk solvent, soft lattice modes) of reorganization energy $\lambda_s$, and a single (or a few) high‐frequency “inner‐sphere” vibration(s) of frequency $\omega$ and reorganization energy $\lambda_\ell$. Defining the Huang–Rhys factor

$$
S = \frac{\lambda_\ell}{\hbar\omega},
$$

the MLJ rate emerges as the Poisson‐weighted sum over discrete vibrational quanta:

![image](https://github.com/user-attachments/assets/6e276783-4c3c-4473-b9e0-f2802ed87387)


Here, each term $m$ corresponds to an ET event exchanging exactly $m$ quanta of the high‐frequency mode. The Gaussian factor reflects classical fluctuations of the outer‐sphere modes, while the prefactor $\propto (\lambda_s T)^{-1/2}$ encodes their density of states. When $k_B T \gg \hbar\omega$ or $S \to 0$, the sum collapses to $m = 0$ and (1) reduces to the classical Marcus expression.

## The Quantum-Mode (Jortner) Limit

A further—and experimentally potent—limiting case occurs when the outer‐sphere reorganization is effectively “frozen” ($\lambda_s \to 0$), as in a rigid glass below its glass‐transition temperature or in a tightly packed solid matrix. Mathematically, the Gaussian in (1) becomes

$$
\lim_{\lambda_s \to 0} \frac{1}{\sqrt{4\pi\lambda_s k_B T}} \exp\left[ -\frac{x^2}{4\lambda_s k_B T} \right] = \delta(x),
$$

so that energy conservation is enforced exactly. Equation (1) then reduces to Jortner’s quantum‐mode rate,

![image](https://github.com/user-attachments/assets/9e6eb2bc-a924-4867-9407-51a59c7df14b)


in which only those vibrational channels satisfying $\Delta G = -m\hbar\omega$ contribute. Because the classical bath no longer broadens the transitions, the rate becomes temperature independent (aside from any residual phonon broadening) and exhibits sharp vibronic resonances.

For a typical C=C stretching mode at $\tilde{\nu} = 1500\,\text{cm}^{-1}$, one finds $\hbar\omega \simeq 0.185\,\text{eV}$ so that $k_B T / \hbar\omega \approx 0.14$ even at $T = 300\,\text{K}$. Thus the quantum‐mode condition $k_B T \ll \hbar\omega$ is already met in most laboratory settings, and the key requirement for (2) is simply a rigid host that suppresses $\lambda_s$.

## Relevance to Quantum Computing

In solid‐state quantum bits—whether charge qubits in quantum dots, donor‐acceptor molecular qubits, or superconducting circuits—the interplay between electronic states and vibrational (phonon) environments governs coherence times and gate fidelities. One often models decoherence and relaxation via variants of the spin–boson Hamiltonian, where the spectral density of phonons plays the role analogous to the reorganization energy in ET theory.

By applying MLJ formalism, one can quantify phonon‐assisted tunneling rates between qubit states. Here $\lambda_s$ maps onto the continuum of low‐frequency acoustic phonons, while discrete optical phonon modes or molecular vibrations correspond to the inner‐sphere quantum mode with frequency $\omega$. In the rigid‐matrix (low‐temperature) regime of quantum processors—typically operated at tens of millikelvin—the classical phonon bath is largely frozen ($\lambda_s \ll k_B T$), so relaxation occurs predominantly via resonant emission or absorption of high‐frequency quanta in a Jortner‐like fashion.

Concretely, if the qubit splitting $\Delta$ matches an integer multiple $m$ of a resonant phonon energy $\hbar\omega$, decoherence and energy relaxation are greatly enhanced, leading to “hot‐phonon” bottlenecks or latching phenomena. Conversely, detuning $\Delta$ off these vibronic resonances suppresses relaxation by orders of magnitude, mirroring the Jortner limit’s off‐resonant rate collapse. Designing qubit energy levels and their local vibrational environments to avoid such resonances is therefore a crucial strategy for extending coherence times in solid‐state devices.

## Outlook

The MLJ framework elegantly bridges the classical Marcus regime and the fully quantum‐mode Jortner limit, offering analytical clarity and physical insight into how discrete vibrations and continuous baths conspire to control non‐adiabatic transitions. In quantum information science, leveraging these ideas allows one to predict—and engineer—the phonon‐mediated relaxation pathways that ultimately limit qubit performance. As materials grow ever more ordered and devices operate ever colder, quantum‐mode effects will dominate, making Jortner’s $\delta$‐function resonances not merely an academic curiosity but a practical design principle for the next generation of quantum technologies.
