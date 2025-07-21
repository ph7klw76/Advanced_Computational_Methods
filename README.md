# Advanced Computational Methods

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/03fe4773-c529-4de7-8ccf-66e925084214" alt="WhatsApp Image" width="600" height="600" />
</div>

This project is more than a collection of computational techniques.  It is my attempt to **capture structure**, **clarify thought**, and **transmit pathways** 

This is where I:
- Think in public
- Teach myself through clarity of thought
- Lay the foundation for future frameworks

I build not to impress but to **preserve clarity**, in a world moving too fast with lots of noise to be able to think clearly.

You want to learn Python. Follow this path.

### This assumes that you have a good grasp of Python coding based on the passing standard of [MITx Computational Thinking using Python](https://www.edx.org/xseries/mitx-computational-thinking-using-python) or else take the course and get certified. Alternatively you learn and do exercises from [Introduction to Computer Science and Programming in Python](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)

This guide provides structured pathways for mastering advanced topics in scientific computing, machine learning, and quantum computing. In order to develop your metacognitive in programming, please click [here](metacognitive.md)  It covers essential methods in non-linear regression, foundational concepts in Nobel Physics 2024 topics, and machine learning models, including physics-informed approaches. In quantum optimization, it reviews algorithms like QAOA and VQE, alongside quantum machine learning techniques. Additional quantum methods address applications in chemistry. Finally, it includes data structures, algorithms, and Python design patterns for efficient computation, offering a comprehensive learning roadmap across these fields. It is important to you develop meta ability of how to learn in this field which can be found [here](Meta-Ability.md) as these topics are PhD level and requires an extensive pior knowledge before capable to particiapte in competitition. 



## To learn the magic of non-linear regression in scipy, follow the steps below:

### Scipy is a powerful libraray for scientists and engineers as it extend the ability of Numpy. It is recommended that you learn from the [website](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide)
1.  [Uncertainty_Gradients_Scientific_Parameter_Extraction](Uncertainty_Gradients_Scientific_Parameter_Extraction.md)
    Non-linear regression is often used in scientific contexts where parameter uncertainty plays a critical role. Gradients help in understanding the sensitivity of parameters.
2.  [Computational_Thinking](Computational_Thinking.md) How to think like a computer.
3.  [Computational_Thinking before Machine Learning](https://www.youtube.com/watch?v=V9Xy18YEK9M)
4.  [SciPy's _curve_fit](SciPy's_curve_fit.md) A common tool in SciPy, curve_fit allows fitting a curve to data by optimizing parameters based on least squares.
5.  [Jacobian_Matrix](Jacobian_Matrix.md) The Jacobian matrix represents the partial derivatives of functions, which is crucial in minimizing errors in non-linear regression.
6.  [Levenberg-Marquardt](Levenberg-Marquardt.md) his optimization method is particularly useful for non-linear least squares problems, balancing between gradient descent and Gauss-Newton methods.
7.  [Covariance_Matrix](Covariance_Matrix.md)  In non-linear regression, the covariance matrix is essential for quantifying uncertainty in parameter estimates.


## To learn the evolution of Nobel Physics 2024, follow the steps below:
1. [Spin_glass](spin_glass.md)  Spin glasses are disordered magnetic systems with complex interactions, leading to phenomena like frustration, critical in understanding certain types of matter.
2. [Hopfieldnetwork_spin glass](hopfieldnetwork_spinglass.md) (Unsupervised learning) An unsupervised learning model, Hopfield networks exhibit spin glass behaviors that model memory storage.U
3. [Boltzmann_Machines_spin_glass](Boltzmann_Machines_spin_glass.md) (Unsupervised learning) sed in unsupervised learning, Boltzmann machines also draw parallels with spin glasses, relevant in energy-based modeling.
4. [Backpropagation_Algorithm](backwardandforwardpropagation.md) (Unsupervised learning) A fundamental algorithm in neural networks that allows learning through adjusting weights, relevant in unsupervised contexts with unique variations.


## To learn Machine Learning, follow the steps below:

### Ideally you also take the lectures on [Machine Learning given by by Professor Andrew Ng](https://www.youtube.com/watch?v=gb262LDH1So&list=PLiPvV5TNogxIS4bHQVW4pMkj4CHA8COdX) for deeper understanding or you want to get certified at [Coursera](https://www.coursera.org/specializations/machine-learning-introduction)

1. [Generalized_Linear_Models](Generalized_Linear_Models.md) (Supervised learning) GLMs extend linear regression to non-linear relationships, common in supervised learning.
2. [k-Nearest_Neighbors](k-Nearest_Neighbors.md) (Supervised learning) A supervised learning method, k-NN uses proximity in feature space to make predictions.
3. [Ensemble Learning](Ensemble_Learning.md) (Supervised learning) Combines multiple models to improve predictive performance; commonly used in supervised learning.
4. [Artificial Neural Network](Artificial_Neural_Network.md) (Both) ANNs can be used for both supervised and unsupervised tasks, mimicking the structure of the human brain.
5. [Physics informed Machine Learning](Physics_informed_Machine_Learning.md) Integrates physical laws into machine learning models to inform and constrain predictions.
6. [Principal Component Analysis](PCA.md) Dimensionality reduction
7. [Physics-Guided Δ-Machine Learning](Physics-Guided-Machine-Learning.md)
8. [Graph Theory](graphtheory.md)
9. [Graph Neural Networks](GNN.md) (https://github.com/Wanlin-Cai/ML_GCN) (https://github.com/chemprop/chemprop?tab=readme-ov-file)
10. [Equivariant neural networks](Equivariantneuralnetworks.md) Modeling physical systems whose underlying laws respect symmetries
11. [Physics- Guided  Δ Equivariant Graph Neural Network](Equivariant_Graph_Convolutional.md)
12. Transfer Learning 
13. Meta-Learning and Few-Shot Learning

### There are numerous libraries which are useful including [Scikit-learn](https://scikit-learn.org/stable/), [Tensorflow](https://www.tensorflow.org/tutorials?authuser=1) and [Pytorch](https://pytorch.org/get-started/locally/)

## To learn Quantum Optimization using PennyLane library, follow the steps below:

### To learn more of quantum computing without strong quantum mechanics background, please also visit [here](https://www.youtube.com/watch?v=c0D8X4eN_Cg&list=PLnK6MrIqGXsJfcBdppW3CKJ858zR8P4eP&index=1)

1.  [Foundation of Quantum Computing](Foundation_QC.md) Basic principles and qubit operations form the groundwork.
2.  [Working with Quadratic Unconstrained Binary Optimization](Quadratic_Unconstrained_Binary_Optimization.md)  A formulation for optimization problems solvable by quantum computers.
3.  [Adiabatic Quantum Computing and Quantum Annealing](Adiabatic_Quantum_Computing_Quantum_Annealing.md) Techniques for optimization problems, using gradual transitions or energy minimization.
4.  [QAOA: Quantum Approximate Optimization Algorithm](Quantum_Approximate_Optimization_Algorithm.md)  A hybrid quantum-classical algorithm for combinatorial optimization.
5.  [GAS: Grover Adaptive Search](Grover_Adaptive_Search.md) An enhanced search algorithm that speeds up finding optimal solutions.
6.  [VQE: Variational Quantum Eigensolver](Variational_Quantum_Eigensolver.md) A method for finding eigenvalues in optimization and chemistry.
7.  [Quantum Phase Estimation](Quantum_Phase_Estimation.md) Essential for understanding phase shifts, critical in quantum algorithms.
8.  [Quantum singular value transformation and block-encoding](Quantum_Singular_Value_Transformation.md) Allows efficient transformations in data processing.
9.  [Hamiltonian Simulation Algorithms](Hamiltonian_Simulation_Algorithms.md) Simulating quantum systems' evolution for computational and physical insights.

## To learn Quantum Machine Learning using PennyLane library, follow the steps below:

### To learn more of Quantum Machine Learning visit the website [here](https://www.youtube.com/watch?v=QtWCmO_KIlg&list=PLmRxgFnCIhaMgvot-Xuym_hn69lmzIokg)

1. [What is Quantum Machine Learning](Quantum_Machine_Learning.md) ?  Covers how quantum mechanics intersects with traditional machine learning.
2. [Quantum Support Vector Machines](Quantum_Support_Vector_Machines.md) A quantum adaptation of SVMs for classification problems.
3. [Quantum Principal Component Analysis](QPCA.md) Dimensionality reduction using quantum techniques for data analysis.
4. [Quantum Neural Networks](Quantum_Neural_Networks.md) Neural networks leveraging quantum computations for complex tasks.
5. [Quantum Random Forest](QRF.md) Combining quantum and classical decision trees for predictions.
6. [Hamiltonian Learning](Hamiltonian_Simulation_Algorithms.md) Estimating quantum system parameters using measurements and optimizations.


## Then you have sufficient background to move to the Cookbook and Demostration

[PennyLane Quantum Cookbook](https://pennylane.ai/codebook)

[Visit PennyLane Quantum Computing Demonstrations](https://pennylane.ai/qml/demonstrations/)

## Creating User Interface and animation with Python
![image](https://github.com/user-attachments/assets/e3483476-497f-490b-801a-de13b51790b9)

where magic happens

1. [Example1](example.md)
2. [Pacman](https://github.com/ph7klw76/Advanced_Computational_Methods/blob/main/Pacman.py)
3. [Microwave](micorwave.md)
4. [Double_slit](doubleslit.py)

## Concurrency and Parallelism

[Concurrency and Parallelism](https://www.youtube.com/watch?v=S05-MZAJqNM)

1. [Speeding up calculation using Numba libraries with example in Physics](Numba.md)
   

## Agent-Based Modeling (ABM)

[Agent Based Modelling](https://www.youtube.com/watch?v=uAgxbrLoSxU&list=PLD4TWcPfbZO9HmaSutF_R2Y2RmiNDxvaP)

1. Introduction to Agent-Based Modeling
2. Agent-Based Modeling Frameworks (Mesa, NetLogo, Repast, AnyLogic)
3. Machine learning techniques with ABM
4. Hybrid models
5. Multi-agent systems
6. Evolutionary and adaptive agents


## Advanced Algorithms

[Advanced Algorithms](https://www.youtube.com/watch?v=omsr-55nG7s&list=PLMDFPuH4ZxUELJN4dBgchm2bqfQzOrPA6&index=1)

1.  Graphs 
2.  Weighted Graphs  
3.  Hash Tables
4.  Heaps
5.  Binary Trees and AVL Trees 
6.  Recursion
7.  Spatial Data Structures
8.  Genetic algorithms
9.  Principal Component Analysis
10. Finite Element Method
11. Quantum Monte Carlo
12. Graph Convolutional Networks (GCN)
13. Particle Swarm Optimization
14. Inverse Design Algorithms
15. t-Distributed Stochastic Neighbor Embedding
    

## Design Pattern in Python

[Design Pattern](https://www.youtube.com/watch?v=kNXDHjIkP_0&list=PLKWUX7aMnlEJzRvCXnwFEdk_WJDNjMDOo&index=1)

1. Introduction to Design Patterns
2. Creational Design Patterns
    Singleton Pattern
    Factory Method Pattern
    Abstract Factory Pattern
    Builder Pattern
    Prototype Pattern
4. Structural Design Patterns
    Adapter Pattern
    Bridge Pattern
    Composite Pattern
    Decorator Pattern
    Facade Pattern
    Flyweight Pattern
    Proxy Pattern
5. Behavioral Design Patterns
    Chain of Responsibility Pattern
    Command Pattern
    Interpreter Pattern
    Iterator Pattern
    Mediator Pattern
    Memento Pattern
    Observer Pattern
    State Pattern
    Strategy Pattern
    Template Method Pattern
    Visitor Pattern
6. Applying Design Patterns in Python
7. Real-World Applications
8. Advanced Topics 


## GPU Programming in Python

[GPU Programming](https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j&index=1)

1. Introduction to GPU Programming
2. Basics of Parallel Computing
3. Setting Up the Environment
4. Introduction to CUDA
5. Python Libraries for GPU Programming
6. Writing GPU Kernels
7. Performance Optimization
8. Advanced GPU Programming Techniques
9. Real-World Applications
10. Best Practices and Common Pitfalls
11. Future Trends in GPU Programming

## Advanced Numerical Methods

### A.Differential Equations
Finite Difference Methods (FDM):

Solves partial and ordinary differential equations (ODEs/PDEs) by discretizing them.
Application: Heat conduction, wave equations, diffusion problems.
Finite Element Methods (FEM):

Breaks a problem into smaller, simpler parts (finite elements).
Application: Stress analysis, quantum mechanics, electromagnetism.
Spectral Methods:

Uses orthogonal functions (e.g., Fourier, Chebyshev) for high-accuracy solutions to PDEs.
Application: Fluid dynamics, turbulence modeling.
Runge-Kutta Methods:

High-precision solutions for ODEs.
Application: Planetary motion, quantum trajectory calculations.

### B. Linear Algebra Techniques

LU and QR Decompositions:

Efficiently solve systems of linear equations.
Application: Eigenvalue problems in quantum mechanics.
Iterative Solvers (e.g., Conjugate Gradient, GMRES):

Solve large, sparse linear systems.
Application: Computational electrodynamics, lattice QCD.
Singular Value Decomposition (SVD):

Decomposes matrices for data reduction and system identification.
Application: Quantum state tomography, data compression.

### C. Monte Carlo Methods

Markov Chain Monte Carlo (MCMC):

Generates samples from a probability distribution.
Application: Statistical mechanics, Bayesian inference, path integrals.
Importance Sampling:

Improves efficiency in Monte Carlo simulations.
Application: High-dimensional integrals in quantum systems.

### D. Optimization and Variational Methods

Gradient-Based Optimization:

Algorithms like steepest descent and conjugate gradient.
Application: Energy minimization in molecular dynamics.
Simulated Annealing and Genetic Algorithms:

Global optimization methods.
Application: Protein folding, material design.
Variational Methods:

Approximate solutions to quantum mechanical systems using trial wavefunctions.
Application: Variational Monte Carlo, density functional theory.


### E. Spectral and Fourier Methods

Fast Fourier Transform (FFT):

Efficiently computes discrete Fourier transforms.
Application: Signal processing, solving PDEs in periodic systems.
Wavelet Transforms:

Decomposes signals into localized time-frequency components.
Application: Analyzing turbulent flows, quantum wavepackets.

### F.  Matrix Methods in Quantum Mechanics

Diagonalization Techniques:

Solves eigenvalue problems for Hamiltonians.
Application: Quantum spectra, vibrational modes.
Time Evolution Operators:

Methods like Crank-Nicolson for time-dependent quantum problems.
Application: Real-time dynamics in quantum mechanics.


### G.  Parallel Computing and High-Performance Methods

Domain Decomposition:

Divide problems for distributed computing.
Application: Molecular dynamics, climate models.
GPU Computing:

Accelerates large-scale simulations.
Application: AIMD, real-time quantum dynamics.
Sparse Matrix Operations:

Optimized for large, sparse systems.
Application: Finite element simulations, quantum lattice models.

## Quantum Chemistry
[Quantum Chemistry](https://www.youtube.com/watch?v=cd2Ua9dKEl8&list=PLcxq_TlK-AyjEdMxhhrte2b8PKe0AHpsr)

### A. Electronic Structure AND Excited-State Properties

1. [Linear Combination of Atomic Orbitals](gaussian.md) How we use basis set for complex atomic orbitals, just like fourier series

2. [Hartree-Fock_Theory](Hartree-Fock_Theory.md)  Foundation for understanding wavefunctions and molecular orbitals.

3. [Density Functional Theory and Time-dependent-DFT](DFT.md) Density Functional Theory (DFT): Crucial for predicting electronic properties of large organic molecules, such as HOMO-LUMO gaps and charge densities. Time-Dependent Density Functional Theory (TD-DFT): For studying optical absorption, emission spectra, and exciton behavior.

4. [Implicit solvation models](Implicit_Solvation.md)

5. [Explicit solvation models](explict_solvent.md)
   
6. Post-Hartree-Fock Methods: (e.g., MP2, CCSD) for more accurate calculations when needed, especially for excitonic effects in organic systems.

7. Configuration Interaction (CI): Useful for modeling excited states and singlet-triplet transitions in materials.

8. Multireference Methods: For systems with significant electron correlation, such as biradicals.

### B Charge Transport Mechanisms

1. [Reorganization Energy](Reorganization_Energy.md): Key parameter in charge hopping and mobility in organic semiconductors.
   
2. [Marcus Theory:](Marcus_Theory.md) For understanding electron and hole transport through hopping mechanisms.
   
4. [Marcus–Levich–Jortner Theory:](Marcus–Levich–Jortner.md) for low temperature 
   
5. [Landau–Zener (LZ) non-adiabatic transition probability:](LZ.md) for large electronic coupling

6.  [Marcus‑Rate Graphs & Exact Kinetic Monte Carlo (n‑Fold Way)](Marcus_Graph.md) A graph method to take into account of local ordering 
      
7.  Dynamic and static disorder
   
8.  Miller-Abrahams Hopping Model
   
9.  Variable Range Hopping (VRH) in Disordered Systems
   
10.  Polaron Hopping Mechanisms
    
11.  Crossover from Hopping to Band Transport
    
12.  Multiple Trapping and Release (MTR) Model
    
13.  Percolation Theory in Organic Films

### C. Molecular Interactions, Aggregation and Torsional Potential

1. [Non-Covalent Interactions](Non-Covalent_Interactions.md) π-π stacking, van der Waals forces, and hydrogen bonding in organic semiconductor.

2. H- and J-Aggregates
   
3. [Charge Transfer Complexes](Charge_Transfer_Complexes.md): Understanding donor-acceptor interactions in organic photovoltaic (OPV) and light-emitting devices.
   
4. [Torsional Potential](Torsion.md): Torsion Angle Definition and Conformational Energy,Dihedral Angles in Small Molecules and Polymers
Energy Barriers for Rotation and Conformational Isomerism

5. [Dispersion Correction](Dispersion.md)
   
6. Aggregation-Induced Emission (AIE) Mechanisms

7. [Estimating the Stabilization Energy of a Charge‐Transfer (CT) State in a Solvent](stabilization_energy.md)

### D.Spectroscopic Properties

1. [Selection Rules for Electronic Transitions](selection_rule.md)
   
2. [Vibrational Analysis](vibrational_analysis.md): Using quantum methods to predict and analyze IR and Raman spectra.
   
3. [UV-Vis and Fluorescence Spectroscopy](absorption_emission.md): Modeling absorption and emission processes.
   
4. [Reorganization Energy for rISC](reorganization.md)
   
6. [Franck-Condon (FC) intensities (IntensityFC) and Herzberg-Teller (HT) intensities ](FC_HT.md)

8. Spectral Line Shapes and Broadening Mechanisms
   
9. [Spin-Orbit Coupling](Spin–Orbit_Coupling.md): Relevant for phosphorescence in OLEDs.

10. Anti-Stokes Fluorescence
   
11. [Intersystem Crossing and Internal Conversion](ISC_IC.md)

12. [Quantum‐Mechanical Portrait of Non‐Radiative Decay](Non‐RadiativeDecay.md)
   
13. [Excited States via RPA, CIS and SF-TDA](Excited_States1.md)
    
14. [Excited State Methods: ROCIS, DFT/ROCIS, and EOM-CCSD](Excited_State_Methods2.md)
    
15. [Excited States via STEOM-CCSD, IH-FSMR-CCSD , PNO-based coupled cluster , DLPNO-STEOM-CCSD](Excited_state_method3.md)

### E. Photophysics and Photochemistry

1. [TADF](TADF.md)

2. [Photophysics Rate equations of organic molecules](TADF2.md)
   
3. [Spin-Vibronic-Pathways](Spin-Vibronic-Pathways.md)
   
4. [Photoinduced Electron Transfer: For understanding OPVs and photocatalytic systems](PET.md)
   
5. [Singlet and Triplet Exciton Dynamics](exciton_dynamics.md): For designing efficient OLED materials.
   
6. [Triplet-Triplet Annihilation (TTA)](TTA.md)
   
7. [Internal Conversion (IC)](IC.md)

8. [Biradical Photophysics](Diradical_%20character.md)
   
9. Singlet Exciton Fission

10. [Energy Transfer Mechanisms](Energy_transfer.md): Förster Resonance Energy Transfer (FRET) and Dexter transfer
   
11. [Two-Photon Absorption and PhotoPolymerisation](2PA.md) : Quantum Description and Mechanism

12. Charge Separation in Donor-Acceptor Systems
    
13. Charge Recombination Mechanisms

14. Third-Harmonic Generation (THG)
    
15. Exciton-Polariton Condensates

16. Metal-Enhanced Fluorescence

17. Strong Light-Matter Coupling
    
18. Spin-Polarized Luminescence
    
19. Excited-State Proton Transfer
    
20. [Transient Absorption](TA.md)

### F: OLED Device Dynamics

1:  [Electroluminescence Dynamics](EL.md)


## Molecular Dynamics

[Molecular Dynamics](https://www.youtube.com/watch?v=BnT6Onll1eQ&list=PLm8ZSArAXicKIzMfkR0Y0GVx9AA1zge-P)
[Molecular Dynamics in Python](https://www.youtube.com/watch?v=6gVoPVosXRs&list=PLP_iHNbRbB3eN3VhO76qiZlDZEyrU88rJ&index=1)

### A. [Fundamentals of Molecular Dynamics](Molecular_Dynamics.md)

Newton’s equations of motion.
Time integration algorithms (e.g., Verlet, leapfrog).
Periodic boundary conditions.
Temperature and pressure control (thermostats and barostats, e.g., Nosé-Hoover, Berendsen).

### B. [Force Fields and Potential Energy Surfaces](Force_Fields.md)

Parameterization of force fields for π-conjugated systems 
Bonded and non-bonded interactions:
Bond stretching, angle bending, dihedral torsions.
van der Waals interactions and electrostatics.
Customization of force fields for organic semiconductors.

### C. [Sampling Techniques](Advanced_Sampling_Techniques.md)
Enhanced sampling methods (e.g., metadynamics, umbrella sampling).
Free energy calculations (e.g., thermodynamic integration, FEP).
Importance sampling for rare event dynamics.

### D. Dynamic Properties

Molecular vibrations and phonon modes.
Diffusion coefficients and mean square displacement.
Time correlation functions (e.g., velocity autocorrelation).

### E.  Structural and Vibrational Analysis

Radial distribution functions (RDF) for structural ordering.
Pair correlation functions and packing density analysis.
Root-mean-square deviation (RMSD) and radius of gyration.
Normal mode analysis (NMA).
Vibrational density of states (VDOS).
Phonon spectra and their coupling with electronic states.

### F. Charge Transport Models

Marcus theory and nonadiabatic transitions.
Polaron formation and hopping mechanisms.
Kinetic Monte Carlo (KMC) simulations using MD data.


### G. Interface and Morphology Effects

Molecular dynamics at donor-acceptor interfaces.
Grain boundary and defect dynamics.
Role of surface energy and molecular orientation.


### H. Nonadiabatic Molecular Dynamics

Ehrenfest and surface hopping approaches.
Role of nonadiabatic coupling in charge transfer.
QM/MM methods for combining MD with quantum chemistry.

### I. Advanced Techniques

Reactive MD (ReaxFF) for studying chemical reactions.
Coarse-grained MD for large-scale morphology analysis.
Machine learning force fields for accelerated simulations.

[https://github.com/ChunHou20c/Monte-Carlo-simulation]
## Ab Initio Molecular Dynamics

### A. [Fundamentals of AIMD](AIMD.md)

Born-Oppenheimer approximation: Separation of electronic and nuclear motion.
Car-Parrinello molecular dynamics (CPMD): Simultaneous evolution of electronic and ionic degrees of freedom.
Difference between classical MD and AIMD.

### B. Quantum Mechanical Forces

Hellmann-Feynman forces.
Pulay corrections (for basis set incompleteness).
Forces from Hartree-Fock and post-Hartree-Fock methods.

### C. Electronic Structure Dynamics

Real-time propagation of electronic wavefunctions.
Electronic excitation and nonadiabatic effects.
Time-dependent DFT (TDDFT) for exciton dynamics.

### D. Vibrational Coupling and Phonons

Phonon modes from AIMD trajectories.
Vibrational energy transfer pathways.
Coupling of phonons with electronic states.

### E. Dynamic Disorder and Charge Transport

Time-dependent variation in electronic properties due to nuclear motion.
Impact of dynamic disorder on bandgap and charge mobility.
Marcus theory and nonadiabatic transitions in AIMD.

### F. Exciton Dynamics and Energy Transfer

Förster and Dexter energy transfer mechanisms.
Role of molecular vibrations in exciton recombination.

### G. Nonadiabatic Dynamics

Surface hopping methods (e.g., Tully’s algorithm).
Nonadiabatic couplings from AIMD trajectories.
Ehrenfest dynamics vs. surface hopping.


[Ab Initio Molecular Dynamics](https://www.youtube.com/watch?v=cGDiVSWpXLc)




