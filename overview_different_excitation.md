# Understanding Excited States: A Comparative Analysis of Computational Approaches

Excited-state calculations play a crucial role in understanding the electronic structure and optical properties of molecules and materials. Various quantum chemical methods have been developed to accurately describe excited states, each with its strengths and limitations. In this article, we explore several computational approaches, comparing their theoretical foundations, computational cost, and applicability to different systems.

---

## 1. Excited States via RPA, CIS, TD-DFT, and SF-TDA

### **Random Phase Approximation (RPA)**

RPA is a post-Hartree-Fock method that describes electronic excitations in the linear response formalism. While computationally expensive, it captures electron correlation effects more accurately than time-dependent density functional theory (TD-DFT). RPA is widely used in solid-state physics and materials science.

### **Configuration Interaction Singles (CIS)**

CIS is a basic wavefunction-based method where single excitations from the ground state determinant are considered. While computationally inexpensive, it lacks correlation effects, leading to systematic overestimation of excitation energies. It is often used for qualitative insight and as a benchmark for more advanced methods.

### **Time-Dependent Density Functional Theory (TD-DFT)**

TD-DFT extends ground-state DFT to describe excited states through linear response theory. It offers a good balance between accuracy and computational efficiency, making it popular in organic electronics, photochemistry, and biophysics. However, it struggles with charge-transfer excitations and Rydberg states due to limitations in commonly used exchange-correlation functionals.

### **Spin-Flip Tamm-Dancoff Approximation (SF-TDA)**

SF-TDA addresses challenges in systems with strong electronic correlation by allowing excitations between spin states. It is particularly useful for treating biradicals and conical intersections, offering an improvement over traditional TD-DFT in multi-reference systems.

---

## 2. Excited States via ROCIS and DFT/ROCIS

### **Restricted Open-Shell Configuration Interaction Singles (ROCIS)**

ROCIS improves upon CIS by including open-shell configurations, making it suitable for transition metal complexes and radical species. It effectively captures spin-dependent effects and provides a more accurate description of excited-state properties compared to conventional CIS.

### **DFT/ROCIS**

DFT/ROCIS combines ROCIS with density functional approximations to include correlation effects at a reduced computational cost. It is widely used in spectroscopy simulations, such as X-ray absorption and UV-Vis spectra of transition metal complexes.

---

## 3. Excited States via MC-RPA

### **Multiconfigurational RPA (MC-RPA)**

MC-RPA extends RPA by incorporating multi-configurational reference states, making it suitable for highly correlated systems. It is particularly useful in photochemistry and transition metal systems where electron correlation plays a significant role. However, it is computationally demanding and typically applied to small molecular systems.

---

## 4. Excited States via EOM-CCSD

### **Equation-of-Motion Coupled Cluster Singles and Doubles (EOM-CCSD)**

EOM-CCSD is one of the most accurate wavefunction-based methods for excited states, providing a balanced description of dynamical and static correlation effects. It is widely used for studying photochemical reactions, charge-transfer processes, and core-excited states. Despite its high accuracy, the computational cost scales as $\mathcal{O}(N^6)$, limiting its applicability to medium-sized systems.

---

## 5. Excited States via STEOM-CCSD

### **Similarity-Transformed EOM-CCSD (STEOM-CCSD)**

STEOM-CCSD modifies the EOM-CCSD approach by using a similarity transformation to improve numerical stability and reduce computational cost. It is particularly useful for studying valence and Rydberg excitations with improved efficiency compared to standard EOM-CCSD.

---

## 6. Excited States via IH-FSMR-CCSD

### **Internally-Contracted Fock-Space Multi-Reference Coupled Cluster (IH-FSMR-CCSD)**

IH-FSMR-CCSD is a multi-reference extension of coupled-cluster theory, designed for systems with strong correlation effects. It is well-suited for transition metals, radicals, and conical intersections. While highly accurate, its computational expense makes it feasible only for small molecules.

---

## 7. Excited States using PNO-based Coupled Cluster

### **Pair-Natural Orbital (PNO)-Based CC Methods**

PNO-based coupled-cluster methods significantly reduce computational cost by truncating excitations in a natural orbital basis. These methods retain the high accuracy of CCSD while enabling calculations on larger systems, making them suitable for photochemistry and organic electronics.

---

## 8. Excited States via DLPNO-STEOM-CCSD

### **Domain-Based Local Pair Natural Orbital STEOM-CCSD (DLPNO-STEOM-CCSD)**

DLPNO-STEOM-CCSD further improves efficiency by combining PNO approximations with STEOM-CCSD. This allows near-coupled-cluster accuracy at a fraction of the computational cost, making high-level excited-state calculations feasible for large systems.

---

## Conclusion

The choice of method for excited-state calculations depends on the trade-off between **accuracy and computational cost**. **TD-DFT** remains the go-to method for large-scale simulations, while **EOM-CCSD and STEOM-CCSD** provide highly accurate results for smaller systems. Emerging techniques such as **PNO-based coupled cluster methods** offer promising pathways for extending high-accuracy methods to larger molecules.
