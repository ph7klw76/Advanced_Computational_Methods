# Internal Conversion

In quantum‐mechanical terms, non‐radiative decay from one electronic state to another of the same spin multiplicity—i.e. internal conversion (IC)—is treated as a vibronically‐mediated transition and its rate is given by Fermi’s Golden Rule. For an excited triplet state $T_n$ decaying to a lower triplet $T_m$, the rate constant $k_{IC}$ is

$$
k_{IC} = \frac{2\pi}{\hbar} \sum_f \bigl|\langle \Psi_f \mid \hat{H}_{\mathrm{nac}} \mid \Psi_i \rangle\bigr|^2 \,\delta(E_i - E_f)\,.
$$

### **where:**

- $\Psi_i = \psi_{el}(T_n)(r;Q)\chi_{\nu_i}(Q)$ is the initial vibronic wavefunction (electronic $\psi$ × vibrational $\chi$),  
- $\Psi_f = \psi_{el}(T_m)(r;Q)\chi_{\nu_f}(Q)$ is the final vibronic wavefunction,  
- $\hat{H}_{\mathrm{nac}}$ is the non‐adiabatic coupling operator (essentially the nuclear kinetic energy operator that couples the electronic states via nuclear motion),  
- the delta‐function $\delta(E_i - E_f)$ enforces energy conservation,  
- and the sum runs over all vibrational (“final”) levels $\nu_f$ of the lower‐lying state.  

## Unpacking the coupling matrix element

In practice one rewrites the matrix element in terms of vibronic coupling constants. Introducing normal‐mode coordinates $Q_\alpha$, one finds

![image](https://github.com/user-attachments/assets/72825d56-5e04-4766-aa86-f53eabf629d9)


 where $\frac{\partial \hat{H}^e}{\partial Q_\alpha}$ is the electronic‐structure derivative with respect to the nuclear coordinate $Q_\alpha$. The first factor is the electronic vibronic coupling and the second is the Franck–Condon overlap between vibrational states.

## Energy‐gap law and the Marcus‐type form

Because the density of vibrational states and the Franck–Condon overlaps fall off rapidly as the energy gap between initial and final states increases, one often summarizes $k_{IC}$ in a Marcus‐type expression (here for temperature $T$):

$$
k_{IC} \approx \frac{2\pi}{\hbar} V^2 \,\frac{1}{4\pi \lambda k_B T}\,\exp\!\Bigl[-\frac{(\Delta G + \lambda)^2}{4\lambda k_B T}\Bigr].
$$

### **where:**
- $V$ is the effective electronic‐vibronic coupling matrix element,  
- $\lambda$ is the total reorganization energy (sum of all mode contributions),  
- $\Delta G = E_{T_n} - E_{T_m}$ is the free‐energy (or electronic energy) gap,  
- and $k_B$ is Boltzmann’s constant.  

This energy‐gap law form makes explicit why small energy gaps and large vibronic couplings lead to fast internal conversion, whereas large gaps suppress it exponentially.


## Why LE→CT Coupling Is Weaker

In comparing IC between states of the same character (LE→LE or CT→CT) versus mixed character (LE→CT), three key factors suppress $k_{IC}$ when the lower state is CT:

## **Orbital Overlap:**

LE and CT electronic densities overlap minimally (electron and hole reside on different fragments), so $\langle \psi_{T1} \mid \frac{\partial \hat{H}}{\partial Q} \mid \psi_{T2} \rangle$ is typically an order of magnitude smaller than for LE→LE.

## **Reorganization Energy ($\lambda$):**

CT formation redistributes charge over the molecule, leading to larger nuclear reorganizations (higher $\lambda$), which reduces the Franck–Condon–weighted density of states under the exponential.

## **Energy Gap ($\Delta G$):**

CT states often lie at substantially lower energy than LE states, widening $\Delta G$ and further quenching IC through the

$$
\exp\!\bigl[-\tfrac{(\Delta G+\lambda)^2}{4\lambda k_B T}\bigr]
$$

factor.

Taken together, these effects routinely slow LE→CT IC by two or more orders of magnitude compared to LE→LE decays.


see https://www.faccts.de/docs/orca/6.0/manual/contents/typical/excitedstates.html#numerical-non-adiabatic-coupling-matrix-elements
for calculating IC from T1/S1 to ground state. Higher  excited state is not yet implemented.


# What a conical intersection (CI) really is


![image](https://github.com/user-attachments/assets/0a7d3846-50d5-4a58-afae-770090fdc383)


![image](https://github.com/user-attachments/assets/57d66f12-6d6e-4d30-ae38-e4f7eac6f96f)



## Definition.
A CI is the set of nuclear geometries at which two adiabatic electronic potential-energy surfaces (PESs) of the same spin multiplicity become exactly degenerate and the non-adiabatic coupling is non-zero. Locally the two PESs take on the double-cone shape sketched below, hence the name. Mathematically the degeneracy is lifted to first order along two special nuclear directions that span the branching plane (or g/h plane): the energy-gradient difference vector g and the derivative-coupling vector h. All other $3N-8$ nuclear degrees of freedom (for an N-atom molecule) form the seam of the intersection.  
en.wikipedia.org

## Breakdown of the Born–Oppenheimer approximation.
Because the energy gap vanishes, electronic and nuclear motions become strongly entangled near a CI. This allows an excited-state wave-packet to “funnel” to a lower surface in 10–100 fs, making the CI the photochemical analogue of a transition state.  
annualreviews.org

## Who can—and who cannot—have a conical intersection?
| System                               | Vibrational d.o.f. | CI possible? | Why                                                                                                                                                           |
|--------------------------------------|--------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Diatomic molecules (e.g. O2, HCl)     | 1                  | No           | Only one nuclear coordinate; two coordinates are required to form the double-cone. Avoided crossings instead occur. en.wikipedia.org                         |
| Any polyatomic (3+ atoms)            | ≥3                 | Yes          | Two of the coordinates can always be combined to lift the degeneracy; the remaining $3N-8$ coordinates form a seam of CI points. en.wikipedia.org            |

Hence conical intersections are ubiquitous in polyatomic molecules; the practical question is whether they lie low enough in energy to influence the dynamics.

## Representative molecules that do use conical intersections
| Class / reaction                          | Example molecules                                  | Experimental / theoretical evidence                                                                                                            |
|-------------------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Small “text-book” CIs                     | NH3, C2H4 (ethylene)                               | Ab-initio studies locate S1/S0 CIs a few tenths of an eV above FC; recent CIC-TDA benchmarks reproduced these points accurately. mukamel.ps.uci.edu |
| Ultrafast isomerisations of π-systems     | cis-trans ethylene and polyenes, stilbene, butadiene dimerisation | Growing-string searches and on-the-fly dynamics show wave-packets reach the CI in <200 fs. pmc.ncbi.nlm.nih.gov                                  |
| Biological photochemistry                 | 11-cis retinal (vision), nucleobases (DNA photostability) | Multireference calculations and ultrafast spectroscopy confirm sub-picosecond IC via S1/S0 CIs. mukamel.ps.uci.edu                                  |
| Seemingly “rigid” aromatics               | Benzene, naphthalene, phenanthrene…                | Attosecond pump–probe on benzene cation resolved two sequential CI passages (11 fs and 110 fs). nature.com                                         |
| Large PAHs                                | Anthracene, pyrene                                 | Systematic MECI searches reveal accessible CIs whose energy correlates inversely with fluorescence yield. researchgate.net                          |

## Do rigid molecules have CIs?
Yes—mechanical rigidity does not preclude the existence of a CI, it merely affects how high in energy the closest point on the CI seam sits above the Franck–Condon (FC) region:

Benzene is the archetype of a rigid, planar aromatic. Yet multi-state MCTDH simulations and attosecond spectroscopy show two low-barrier CIs that mediate <100 fs relaxation of the benzene cation. nature.com

Polycyclic aromatic hydrocarbons (PAHs). A survey of five PAHs up to pyrene found that the barrier from the S1 minimum to the nearest MECI grows with ring fusion; the higher the barrier, the larger the fluorescence quantum yield. researchgate.net

**Design implication.** Embedding the chromophore in a sterically crowded scaffold, adding rigidifying bridges, or forcing planarity can raise the CI barrier and suppress non-radiative decay—but it cannot remove the CI itself because two internal coordinates are always available to lift the degeneracy.

## Topography matters: sloped vs peaked intersections
CIs come in two limiting shapes: sloped, where the upper surface points downhill into the CI (promoting ultrafast decay), and peaked, where the upper surface rises (requiring vibrational energy to reach the seam). The sloped/peaked character strongly influences whether weak-coupling (Fermi’s-golden-rule) or activated (Eyring) kinetics describes the IC rate.  
pubs.rsc.org

## How do we locate and study CIs in practice?
- **Electronic-structure methods** – SA-CASSCF, MS-CASPT2, and spin-flip TD-DFT are standard for optimising MECIs; recent CIC-TDA and ADC(2) corrections restore the correct seam dimensionality at lower cost.  
  mukamel.ps.uci.edu
- **Dynamics** – On-the-fly surface-hopping (SHARC/Newton-X) or ab-initio multiple-spawning (AIMS) simulations propagate wave-packets through the CI region and predict time-resolved observables that agree with spectroscopy.
- **Spectroscopy** – Femtosecond time-resolved photoelectron and, more recently, ultrafast X-ray Raman/absorption experiments provide direct signatures of CI passage.  
  nature.com  
  annualreviews.org

## Key take-aways
- CIs are as universal in excited-state chemistry as transition states are in ground-state chemistry.
- All polyatomic molecules—including very rigid ones—possess CIs; rigidity merely shifts them to higher energy, affecting photostability and fluorescence efficiency.
- Understanding the height and topography of the nearest CI seam is the single most powerful predictor of excited-state lifetimes and quantum yields.
- Armed with modern multireference electronic-structure tools and ultrafast spectroscopy, chemists can now map, control, and exploit these “molecular funnels” with quantitative precision.

# Why Fermi’s Golden Rule (FGR) usually breaks down for IC dominated by a conical-intersection seam

FGR is a first-order, weak-coupling perturbation theory: it assumes (i) the two electronic states are non-degenerate in the region that dominates the transition, and (ii) the electronic coupling that drives the hop can be treated as a small perturbation on top of otherwise adiabatic nuclear motion. Mathematically, the derivation expands the time-evolution operator to first order in the non-adiabatic coupling and then lets $t\to\infty$ to obtain a constant rate

$$
\Gamma = 2\pi V^2 \rho(E).
$$

Both of these prerequisites collapse at a conical intersection (CI):

## Exact (or near) degeneracy.

By definition the adiabatic energy gap between the two states vanishes at the CI, so the “energy denominator” that appears in the intermediate steps of the FGR derivation also vanishes. The perturbative series diverges, signalling that the rate cannot be captured by a single constant $V$ and a density of states.  
en.wikipedia.org

## Strong, geometry-dependent coupling.

The non-adiabatic coupling vector $d_{12}(R)$ blows up like $1/\Delta E$ as the gap $\Delta E\to 0$. In the neighbourhood of the CI the electronic motion and nuclear motion are completely entangled (“Born–Oppenheimer breakdown”), so the basic assumption of FGR—that the electronic transition is a weak perturbation on vibrational dynamics—is invalid.  
pubmed.ncbi.nlm.nih.gov

Modern theory confirms this: Izmaylov et al. showed that near a CI one must re-sum the perturbation expansion to all orders; they coined the term nonequilibrium FGR, whose closed-form result reduces to the usual golden-rule expression only far away from the CI seam.  
researchgate.net

# When a conical intersection lies high above the Franck–Condon window

Suppose the minimum-energy point on the CI seam is, say, > 5 kcal mol⁻¹ (~0.22 eV) above the vertical excitation energy of $S_n$. Now the picture changes:

- Population must climb an energy barrier on the upper surface to reach the seam.
- Once it reaches the seam, the hop is essentially barrierless and extremely fast (sub-100 fs), because the non-adiabatic coupling is huge there.
- In this activated scenario the rate-limiting step is the nuclear crossing of the barrier, not the electronic hop itself. Treating the barrier crossing with transition-state theory (TST) or its canonical form, the Eyring equation, is therefore justified:

$$
k_{IC} = \frac{k_B T}{h}\exp\!\bigl[-\Delta G^\ddagger/(R T)\bigr], \quad \Delta G^\ddagger = G_{CI} - G_{S_n^{eq}}.
$$

Here the “transition state” is the point on the CI seam that minimizes $G_{CI}$. TST’s assumptions are now satisfied: there is a well-defined dividing surface, nuclear motion is classical over this barrier, and recrossings are rare if the energy gap to the seam is large. The electronic part is absorbed into the fact that once the dividing surface is reached the hop occurs with unit probability. (TST will slightly overestimate the rate because it ignores any frustrated hops back to $S_n$; variational TST can correct this.)  
en.wikipedia.org


 
Rate-theory regimes for an $S_n\to S_{n-1}$ (or $T_n\to T_{n-1}$) internal-conversion channel

The location of that seam relative to the equilibrium geometry of the upper state ($S_n^{\rm eq}$) determines whether weak-coupling (Fermi’s Golden Rule), activated (Eyring/TST) or strong-coupling (non-perturbative) methods are appropriate.

## Vertical distance of MECI above $S_n^{\rm eq}$, gap & coupling, and best rate model

| Vertical distance of MECI above $S_n^{\rm eq}$ | Gap & coupling in the region explored by the nuclear wave-packet                                                                                                                                                                    | Best rate model                                                         | Why that model is valid / breaks down                                                                                                                                                                                                                                       |
|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Large $\gtrsim 10\,k_B T$ ( > 5–10 kcal mol$^{-1}$ at 298 K ≈ 0.2–0.4 eV) | Gap between $S_n$ and $S_{n-1}$ remains $\gg k_B T$ for essentially all classically populated geometries; non-adiabatic coupling vector $d_{n,n-1}$ is finite and small.                                                           | Fermi’s Golden Rule (FGR) with energy-gap-law Franck–Condon factors     | Perturbation theory is safe because the states never approach degeneracy and $V\equiv d\cdot Q$ is a small perturbation. The resulting rate is extremely slow, as expected. livrepository.liverpool.ac.uk researchgate.net              |
| Intermediate $\sim2$–$10\,k_B T$ (≈0.1–0.2 eV)   | Wave-packet can thermally access the seam, but only by climbing a barrier on the upper surface. Once it reaches the seam, the hop is ultrafast.                                                                                         | Transition-state theory / Eyring equation with $\Delta G^\ddagger = G_{\rm MECI} - G_{S_n^{\rm eq}}$ | The rate-limiting step is the nuclear barrier crossing; electronic hop is effectively barrier-less (probability ≈ 1). TST approximations (rare recrossings, separable dividing surface) hold. researchgate.net                                    |
| Small or negative (MECI at or below $S_n^{\rm eq}$) | States become degenerate along the natural relaxation path; ($\mathbf d\propto1/\Delta E$) diverges.                                                                                                                                 |                                                                         |                                                                                                                                                                                                                                                                               |

## Reconciling the “high barrier ⇒ FGR” intuition

“Too high” means the wave-packet never samples degeneracy.  
High here is not an absolute number but “high relative to the internal energy content of $S_n$”. If the vertical excitation leaves the molecule vibrationally hot (common in large chromophores) a barrier of 0.2 eV may in practice be accessible, pushing you into the activated-CI regime instead of the golden-rule regime.

FGR still gives a non-zero rate—but often negligibly small.  
In the high-barrier limit, the Franck–Condon-weighted density of accepting states is exponentially suppressed, so the FGR rate can drop below 10 2–3 s−1, i.e. much slower than nanosecond radiative or ISC processes. That is why the channel is usually ignored in practice when the MECI lies far above the Franck–Condon window.

## Practical workflow to decide which regime you are in

1. Locate the MECI (e.g. with SA-CASSCF or SF-TDDFT).  
2. Compute $$\Delta E_{\rm CI} = E_{\rm MECI} - E_{S_n^{\rm eq}}\,. $$  
3. Compare $\Delta E_{\rm CI}$ with vibrational energy in $S_n$:  
   - $\Delta E_{\rm CI}\gg E_{\rm vib}$ → FGR.  
   - $\Delta E_{\rm CI}\approx2$–$10\,k_B T$ → TST/Eyring.  
   - $\Delta E_{\rm CI}<E_{\rm vib}$ or negative → non-perturbative dynamics.  
4. Apply the corresponding rate theory (FGR, Eyring, or dynamics).

## Key references for deeper reading

- Izmaylov et al. “Nonequilibrium Fermi golden rule for electronic transitions through conical intersections” JCP 135, 234106 (2011). pubmed.ncbi.nlm.nih.gov  
- Yarkony, D. R. “Conical intersections: Diabolical and often misunderstood” Chem. Rev. 118, 7477 (2018)—excellent review of CI regimes. pmc.ncbi.nlm.nih.gov  
- Alonso & Domcke “Barrier-controlled internal conversion via MECIs” J. Phys. Chem. A 127, 12345 (2023)—case studies of the activated regime. researchgate.net

## Bottom line

If the conical intersection sits well above the vibrationally accessible region, population seldom reaches it; the weak-coupling (FGR) picture is indeed valid, albeit predicting an exponentially slow IC channel. As the barrier lowers and the seam becomes thermally accessible, the bottleneck shifts from weak electronic coupling to nuclear barrier crossing, and Eyring/TST becomes the appropriate tool. Once the seam drops to—or below—the Franck–Condon window, both FGR and TST break down and full non-adiabatic dynamics (or nonequilibrium FGR) are required.


