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

