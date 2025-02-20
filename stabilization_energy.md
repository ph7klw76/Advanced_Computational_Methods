# Estimating the Stabilization Energy of a Charge‐Transfer (CT) State in a Solvent

Below is a broadly used conceptual and computational procedure to estimate the stabilization energy of a charge‐transfer (CT) state in a solvent. In other words, how much the solvent environment lowers (or sometimes raises) the energy of a CT excited state relative to what it would be in the gas phase. Although details can vary depending on the quantum chemistry software and the model of solvation, the outline below describes the key steps and rationale.

## 1. Understand the Definition of Stabilization Energy

For a charge‐transfer state $\psi_{\text{CT}}$, the stabilization energy due to the solvent can be broadly defined as:

$$
\Delta E_{\text{stabilization}} = [E_{\text{CT}}(\text{solution}) - E_{\text{GS}}(\text{solution})] - [E_{\text{CT}}(\text{gas}) - E_{\text{GS}}(\text{gas})],
$$

where:

- $E_{\text{CT}}(\text{solution})$ is the total energy of the CT excited state in solution,
- $E_{\text{GS}}(\text{solution})$ is the total energy of the ground state in solution,
- $E_{\text{CT}}(\text{gas})$ is the total energy of the CT state in the gas phase,
- $E_{\text{GS}}(\text{gas})$ is the total energy of the ground state in the gas phase.

The difference:

$$
[E_{\text{CT}}(\text{solution}) - E_{\text{CT}}(\text{gas})]
$$

represents how the solvent shifts the energy of the CT excited state, while:

$$
[E_{\text{GS}}(\text{solution}) - E_{\text{GS}}(\text{gas})]
$$

represents how the solvent shifts the ground-state energy. The net effect on the CT transition (the energy gap) is then given by taking the difference of these two shifts.

## 2. Choose a Solvation Model

There are generally two categories of solvation models:

### **Implicit (Continuum) Solvation Models**
- Examples: Polarizable Continuum Model (PCM), Conductor-like Polarizable Continuum Model (CPCM), SMD model, etc.
- The solvent is treated as a continuous polarizable medium characterized by its dielectric properties.
- These models are relatively fast and straightforward to apply.

### **Explicit Solvation Models**
- The solute molecule is surrounded by explicit solvent molecules, often in a molecular dynamics or Monte Carlo simulation.
- Can be combined with a QM/MM approach.
- More computationally expensive but can capture specific solvent–solute interactions (e.g., hydrogen bonding) more accurately.

For many routine calculations of CT state stabilization, an implicit solvation model (PCM or SMD) at the TD-DFT level is a common, verifiable approach.

## 3. Define a Computational Strategy

A step-by-step procedure for an implicit solvent approach is often as follows (using TD-DFT as an example):

### **3.1 Optimize the Ground State**

#### **Gas-phase geometry optimization of the ground state**
- Perform a geometry optimization at an appropriate level of theory (e.g., DFT with a chosen functional and basis set) in the gas phase.
- This yields $E_{\text{GS}}(\text{gas})$ and the optimized geometry $R_{\text{GS}}(\text{gas})$.

#### **Solution-phase geometry optimization of the ground state (if desired)**
- Some approaches keep the same geometry for both gas-phase and solution calculations (the “vertical” approach).
- Alternatively, you may optimize the ground state within the continuum (e.g., PCM) or with explicit solvent. This yields $E_{\text{GS}}(\text{solution})$ and the optimized geometry $R_{\text{GS}}(\text{soln})$.

> **Note**: If you want to measure the stabilization of an electronic transition vertically (i.e., using the ground-state geometry for both calculations), you may skip the geometry re-optimization in solution. However, for more refined results, optimizing both states in their respective environments can be helpful.

### **3.2 Calculate the Excited-State Energies**

#### **Excited state in gas phase**
- Using the geometry $R_{\text{GS}}(\text{gas})$ (or optionally an excited-state geometry for a more adiabatic approach), run a TD-DFT (or other appropriate excited-state method) calculation in the gas phase to get $E_{\text{CT}}(\text{gas})$.

#### **Excited state in solution**
- On the same geometry (if doing a vertical approach) or on an optimized excited-state geometry in solution (for an adiabatic approach), run a TD-DFT calculation with the chosen solvation model to get $E_{\text{CT}}(\text{solution})$.

Depending on which approach (vertical vs. adiabatic) you use, you will have either:

##### **Vertical transitions:**

$$
\Delta E_{\text{stabilization}} = (\Delta E_{\text{CT,vertical}}(\text{soln})) - (\Delta E_{\text{CT,vertical}}(\text{gas})),
$$

where:

$$
\Delta E_{\text{CT,vertical}}(\text{soln}) \equiv E_{\text{CT}}(\text{soln}) (R_{\text{GS}}) - E_{\text{GS}}(\text{soln}) (R_{\text{GS}}).
$$

##### **Adiabatic transitions (state-specific optimization):**

$$
\Delta E_{\text{stabilization}} = [(E_{\text{CT}}(\text{soln}) (R_{\text{CT}}(\text{soln})) - E_{\text{GS}}(\text{soln}) (R_{\text{GS}}(\text{soln})))] - [(E_{\text{CT}}(\text{gas}) (R_{\text{CT}}(\text{gas})) - E_{\text{GS}}(\text{gas}) (R_{\text{GS}}(\text{gas})))].
$$

## 4. Interpret the Results

Once you have the energies:

- **Ground state energy in gas phase**: $E_{\text{GS}}(\text{gas})$
- **Ground state energy in solution**: $E_{\text{GS}}(\text{solution})$
- **CT excited state energy in gas phase**: $E_{\text{CT}}(\text{gas})$
- **CT excited state energy in solution**: $E_{\text{CT}}(\text{solution})$

you can plug them into:

$$
\Delta E_{\text{stabilization}} = [(E_{\text{CT}}(\text{solution}) - E_{\text{GS}}(\text{solution})) - (E_{\text{CT}}(\text{gas}) - E_{\text{GS}}(\text{gas}))].
$$

- A **negative** $\Delta E_{\text{stabilization}}$ indicates that the solvent **lowers** the energy of the CT state relative to the gas phase (i.e., stabilizes the CT state).
- A **positive** $\Delta E_{\text{stabilization}}$ would indicate that solvation **raises** the energy gap (less common for classic polar solvents with a strong dipole but can happen depending on specific interactions).
