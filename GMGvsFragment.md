## Two types of Electronic coupling.

The Generalized Mulliken–Hush (GMH) approach gives you a coupling between electronic states, while the fragment-orbital Fock–overlap (FO, often called FO/DFT or FODFT-style) method gives you a coupling between specific molecular orbitals on different fragments. Both quantify “how strongly things interact,” but they live in different bases and answer different questions.

GMH starts from two interacting adiabatic states of the whole system, for example a locally excited state and a charge-transfer state, or the two mixed states in a mixed-valence complex. The key idea is to rotate these two states into a diabatic basis that diagonalizes the dipole operator along a chosen charge-transfer axis. In that dipole-diagonal basis, the off-diagonal Hamiltonian element is the desired state coupling $V_{DA}$. Practically, GMH needs only the energy gap between the two adiabatic states $\Delta E$, the difference between their state dipoles $\Delta \mu$, and the transition dipole $\mu_{12}$ projected along the charge-transfer direction. The defining relations are $\tan(2\theta) = 2\mu_{12}/\Delta\mu$ for the rotation angle and

$$
V_{DA} = \frac{\Delta E \lvert \mu_{12} \rvert}{\sqrt{(\Delta \mu)^2 + 4 \lvert \mu_{12} \rvert^2}}
$$

for the coupling. Because it works in a two-state subspace and depends only on energies and dipoles, GMH is model-independent and connects cleanly to spectroscopy and Marcus-type electron-transfer rates.

The fragment-orbital Fock–overlap method, by contrast, works directly with orbitals localized on fragments (for example, HOMO on fragment A and HOMO on fragment B) embedded in the dimer. You compute the dimer’s Fock matrix $F$ and overlap matrix $S$ in an atomic-orbital basis, project them onto the chosen fragment orbitals, and then correct for non-orthogonality. The resulting orbital-to-orbital transfer integral is

$$
t_{AB} = \frac{E_{AB} - \frac{1}{2}(E_{AA} + E_{BB}) S_{AB}}{1 - S_{AB}^2}
$$

where $E_{AA} = \langle \phi_A \lvert F \rvert \phi_A \rangle$, $E_{BB} = \langle \phi_B \lvert F \rvert \phi_B \rangle$, $E_{AB} = \langle \phi_A \lvert F \rvert \phi_B \rangle$, and $S_{AB} = \langle \phi_A \lvert \phi_B \rangle$. This quantity is ideal when you want hopping integrals for transport models, tight-binding parameterization, or to compare specific orbital pathways in molecular stacks and organic semiconductors.

Interpreting the numbers is straightforward once you keep the bases straight. GMH’s $V_{DA}$ is a state coupling: use it to discuss state mixing, to compare with intervalence charge-transfer spectra, or to plug into nonadiabatic (Marcus) rate expressions together with a reorganization energy. FO’s $t_{AB}$ is an orbital coupling: use it to build band or hopping models, to parameterize effective Hamiltonians, and to reason about how geometry and overlap change the ease of charge motion between specific sites. They can track each other when a state is dominated by a single pair of fragment orbitals, but they can diverge if multiple orbitals contribute, if overlap is large, or if your physical question is about states rather than orbitals (or vice versa).

Choosing between them depends on what you actually want to know. If your question is “how strongly do two states mix?”—for example, to explain a spectrum or to compute an electron-transfer rate—use GMH. If your question is “what is the hopping integral between these two orbitals?”—for example, to parameterize charge transport in a crystal or dimer—use the FO Fock–overlap approach. In practice, many projects benefit from doing both: FO gives orbital-level intuition and parameters for transport, while GMH provides a state-level coupling that ties directly to spectroscopy and kinetics.

Be mindful of common pitfalls. GMH assumes a two-state picture and depends on the chosen charge-transfer axis; check that only two states dominate and test the axis choice for robustness. FO results depend on how you define and orthogonalize fragment orbitals and on the magnitude of the overlap; use consistent localization and consider Löwdin orthogonalization or alternative projections as cross-checks. Neither method is “better” in general—the right one is the one that matches your physical question and the data you have.

The bottom line is simple. GMH is a dipole-based two-state rotation that yields a state coupling $V_{DA}$ for spectroscopy. FO (Fock–overlap) is a matrix-element projection that yields an orbital coupling $t_{AB}$ for transport and tight-binding models. Keep the basis and the question aligned, and your numbers will make clear physical sense.

## What question each method answers

| If you want to know… | Use | Why |
|----------------------|-----|-----|
| “How strongly do two electronic states (e.g., LE and CT) mix?” | GMH | Operates in a 2-state adiabatic subspace; gives a model-independent $V_{DA}$ tied to spectroscopy and Marcus theory. |
| “What is the hopping integral between specific orbitals on two fragments (e.g., HOMO$_A$ → HOMO$_B$)?” | FO (Fock–Overlap) | Works directly with fragment orbitals and the dimer’s Fock/overlap matrices; natural for transport integrals in organic semiconductors. |
