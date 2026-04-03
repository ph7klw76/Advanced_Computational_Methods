Superconductivity is a macroscopic quantum phenomenon characterized by the complete disappearance of electrical resistance and the expulsion of magnetic fields (the Meissner effect). From a quantum mechanical perspective, it is understood as a phase transition of a Fermi gas into a coherent condensate of fermion pairs, known as **Cooper pairs**.

The theoretical foundation is provided by the **BCS Theory** (Bardeen, Cooper, and Schrieffer).

---

### 1. The Cooper Instability
In a normal metal, electrons are fermions obeying the Pauli Exclusion Principle, filling a Fermi sea up to the Fermi energy $\epsilon_F$. Normally, electrons repel each other via the Coulomb interaction. However, in a superconductor, an attractive interaction is mediated by **lattice vibrations (phonons)**.

#### The Interaction Mechanism
When an electron moves through the lattice, it attracts nearby positive ions, creating a local region of increased positive charge density. A second electron is attracted to this "polarization cloud."

The effective interaction Hamiltonian for two electrons with momenta $\mathbf{k}$ and $-\mathbf{k}$ and opposite spins $\uparrow, \downarrow$ is given by:
$$V_{\text{eff}} = \sum_{\mathbf{k, k'}} V_{\mathbf{k, k'}} c^\dagger_{\mathbf{k}\uparrow} c^\dagger_{-\mathbf{k}\downarrow} c_{-\mathbf{k'}\downarrow} c_{\mathbf{k'}\uparrow}$$
Where $V_{\mathbf{k, k'}}$ is the interaction potential. BCS simplified this by assuming a constant attractive potential $-V$ within a thin shell around the Fermi surface:
$$V_{\mathbf{k, k'}} = \begin{cases} -V & \text{if } |\epsilon_{\mathbf{k}}| < \hbar\omega_D \\ 0 & \text{otherwise} \end{cases}$$
where $\omega_D$ is the Debye frequency.

---

### 2. The BCS Mean-Field Theory
To solve the many-body problem, we introduce the **order parameter** $\Delta$, which represents the vacuum expectation value of the pair operator (the "gap function"):
$$\Delta = -V \sum_{\mathbf{k}} \langle c_{-\mathbf{k}\downarrow} c_{\mathbf{k}\uparrow} \rangle$$
This $\Delta$ acts as a complex scalar field representing the macroscopic wavefunction of the condensate.

#### The BCS Hamiltonian
Applying the mean-field approximation, the Hamiltonian becomes quadratic:
$$H_{\text{BCS}} = \sum_{\mathbf{k}, \sigma} \xi_{\mathbf{k}} c^\dagger_{\mathbf{k}\sigma} c_{\mathbf{k}\sigma} - \sum_{\mathbf{k}} (\Delta c^\dagger_{\mathbf{k}\uparrow} c^\dagger_{-\mathbf{k}\downarrow} + \Delta^* c_{-\mathbf{k}\downarrow} c_{\mathbf{k}\uparrow})$$
where $\xi_{\mathbf{k}} = \epsilon_{\mathbf{k}} - \mu$ is the kinetic energy relative to the chemical potential.

#### Bogoliubov-Valatin Transformation
To diagonalize this Hamiltonian, we transform the electron operators into **Bogoliubon operators** ($\gamma_{\mathbf{k}}$), which are linear combinations of electrons and holes:
$$\begin{pmatrix} c_{\mathbf{k}\uparrow} \\ c^\dagger_{-\mathbf{k}\downarrow} \end{pmatrix} = \begin{pmatrix} u_{\mathbf{k}} & -v_{\mathbf{k}} \\ v_{\mathbf{k}} & u_{\mathbf{k}} \end{pmatrix} \begin{pmatrix} \gamma_{\mathbf{k}0} \\ \gamma^\dagger_{-\mathbf{k}1} \end{pmatrix}$$
The coefficients $u_{\mathbf{k}}$ and $v_{\mathbf{k}}$ must satisfy $|u_{\mathbf{k}}|^2 + |v_{\mathbf{k}}|^2 = 1$. Solving the Heisenberg equations of motion yields the **quasi-particle excitation energy**:
$$E_{\mathbf{k}} = \sqrt{\xi_{\mathbf{k}}^2 + |\Delta|^2}$$
This shows that there is a minimum energy $\Delta$ (the energy gap) required to create an excitation (break a Cooper pair).

---

### 3. The Gap Equation
The consistency condition for $\Delta$ is determined by the **BCS Gap Equation**:
$$\Delta = V \sum_{\mathbf{k}} \frac{\Delta}{2\sqrt{\xi_{\mathbf{k}}^2 + \Delta^2}} \tanh\left( \frac{\beta E_{\mathbf{k}}}{2} \right)$$
At $T = 0$, the $\tanh$ term goes to 1. Converting the sum to an integral over the density of states $N(0)$ at the Fermi level:
$$1 = V N(0) \int_{0}^{\hbar\omega_D} \frac{d\xi}{\sqrt{\xi^2 + \Delta_0^2}}$$
Solving for $\Delta_0$ gives the famous result:
$$\Delta_0 \approx 2\hbar\omega_D \exp\left( -\frac{1}{V N(0)} \right)$$
This non-perturbative result proves that superconductivity cannot be reached via standard perturbation theory in $V$.

---

### 4. Macroscopic Quantum Coherence
The Cooper pairs behave as composite bosons. Below $T_c$, they undergo a process analogous to Bose-Einstein Condensation (BEC), occupying a single quantum state.

The system can be described by a single macroscopic wavefunction:
$$
\Psi(\mathbf{r}) = \sqrt{n_s(\mathbf{r})} e^{i\theta(\mathbf{r})}
$$
where $n_s$ is the superfluid density and $\theta$ is the phase.

#### The Supercurrent
The current density $\mathbf{j}$ is derived from the quantum mechanical probability current:
$$
\mathbf{j} = \frac{q \hbar}{2m} n_s \left( \nabla \theta - \frac{q}{\hbar} \mathbf{A} \right)
\$$
where $\mathbf{A}$ is the vector potential and $q = 2e$.

#### Meissner Effect and London Equations
The rigidity of the wavefunction $\Psi$ implies that $\nabla \theta$ is constant (or zero in a simply connected superconductor). Taking the curl of the current equation:
$$\nabla \times \mathbf{j} = -\frac{n_s q^2}{m} \mathbf{B}$$
Combined with Maxwell's equations ($\nabla \times \mathbf{B} = \mu_0 \mathbf{j}$), this leads to:
$$\nabla^2 \mathbf{B} = \frac{1}{\lambda_L^2} \mathbf{B}, \quad \text{where } \lambda_L = \sqrt{\frac{m}{\mu_0 n_s q^2}}$$
This proves that the magnetic field $\mathbf{B}$ decays exponentially inside a superconductor over the **London penetration depth** $\lambda_L$, explaining the Meissner effect.

---

### Summary Table: Normal vs. Superconducting State

| Property | Normal Metal | Superconductor (BCS) |
| :--- | :--- | :--- |
| **Quasiparticles** | Electrons (Fermions) | Bogoliubons (Mixed Electron-Hole) |
| **Ground State** | Fermi Sea | Coherent Condensate of Cooper Pairs |
| **Energy Spectrum** | Continuous at $\epsilon_F$ | Energy Gap $2\Delta$ |
| **Wavefunction** | Many independent $\psi_i$ | Single macroscopic $\Psi$ |
| **Response to $\mathbf{B}$** | Penetration/Paramagnetism | Exponential Expulsion (Meissner) |<turn|>
