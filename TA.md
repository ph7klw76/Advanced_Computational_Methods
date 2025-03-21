
### **Transient Absorption (TA) Spectroscopy and Third-Order Polarization**

Transient Absorption (TA) spectroscopy arises from the third-order polarization in the context of nonlinear optical response theory. To understand this, let's break down the concepts involved.

---

## **1. Nonlinear Optical Response**

In general, when a system (such as a molecule) interacts with an external electric field, the system’s polarization (which describes the response of the system to the electric field) is related to the applied field. For a linear response, the polarization is simply proportional to the electric field, as described by:

$$
P(t) = \chi^{(1)} E(t)
$$

Where:

- $P(t)$ is the polarization  
- $E(t)$ is the electric field  
- $\chi^{(1)}$ is the linear susceptibility

However, in nonlinear optics, the polarization becomes a more complicated function of the electric field, and higher-order susceptibilities come into play. The general expression for the polarization at the $n$-th order in the electric field is:

$$
P(t) = \sum_n \chi^{(n)} E(t)^n
$$

In the case of third-order polarization, the polarization is related to the cube of the electric field:

$$
P^{(3)}(t) = \chi^{(3)} E(t)^3
$$

This third-order term plays a central role in nonlinear processes, such as third-order susceptibilities that govern phenomena like third-harmonic generation, four-wave mixing, and Transient Absorption.

---

## **2. Third-Order Polarization and Its Role in TA**

The key to understanding TA spectroscopy is that it originates from the third-order polarization of the system. The response of the system to two laser pulses (the pump and the probe) is a third-order nonlinear process. Let’s go through the steps to understand this in the context of TA.

### **a. Pump and Probe Pulses**

In time-resolved transient absorption spectroscopy, we typically apply a pump pulse (which excites the system to an excited state) and a probe pulse (which measures the absorption of the system at different time delays after the pump). The time delay between the pump and probe is denoted by $\tau$.

- The pump pulse excites the system from the ground state to an excited state.  
- The probe pulse is used to monitor the absorption of the system as a function of time after the excitation.

### **b. Third-Order Nonlinear Process**

The interaction of the system with these two pulses is a third-order nonlinear process. This means that the polarization response of the system to these pulses involves the third power of the electric field.

In general, the third-order polarization $P^{(3)}(t)$ can be expressed as a time-dependent function involving the electric fields of the pump and probe pulses:

$$
P^{(3)}(t) \propto \int_{-\infty}^{t} dt_1 \int_{-\infty}^{t_1} dt_2 \int_{-\infty}^{t_2} dt_3 \,
E_{\text{pump}}(t_1) E_{\text{pump}}(t_2) E_{\text{probe}}(t_3) \, \chi^{(3)}(t_1, t_2, t_3)
$$

Where $\chi^{(3)}(t_1, t_2, t_3)$ is the third-order susceptibility, which is related to the system's response to the applied fields and contains the information about the transition between quantum states, including:

- Excited-state absorption (ESA)
- Ground-state bleaching (GSB)
- Stimulated emission (SE)

### **c. Generation of Transient Absorption Signal**

The Transient Absorption signal arises from the probe pulse interacting with the system after it has been excited by the pump pulse. The induced polarization at the probe frequency is what determines the signal we measure.

The signal that we observe is proportional to the imaginary part of the third-order polarization, which reflects the absorption of light by the system:

$$
S(\omega, \tau) \propto \Im \left[ \int dt \, e^{i \omega t} P^{(3)}(t) \right]
$$

Here:

- $\omega$ is the frequency of the probe pulse  
- $\tau$ is the time delay between the pump and probe pulses

Thus, the third-order polarization gives rise to the time-resolved absorption spectrum by describing how the system absorbs light at various time delays. The system's dynamics are encoded in this signal, such as the relaxation, reorganization, and decay of the excited state populations.

### **d. Physical Interpretation of the Polarization Components**

To fully appreciate this, it's important to recognize that the third-order polarization describes different pathways for the system's evolution after the pump pulse:

- **Ground-State Bleach (GSB):** This occurs when the system absorbs the probe pulse but the probe finds the system in the ground state (after it has been excited by the pump). This pathway can lead to a depletion of ground-state population.

- **Stimulated Emission (SE):** Here, the probe pulse induces emission from the excited state back to a lower state (often back to the ground state). This pathway generally results in a negative signal (indicating the system is emitting light).

- **Excited-State Absorption (ESA):** This occurs when the system absorbs the probe pulse while it is in an excited state. The probe excites the system further into a higher excited state.

The sum of these contributions (GSB, SE, and ESA) leads to the transient absorption spectrum.

---
To calculate ESA for singlet using ORCA

```text
! DEF2-SVP CPCM(Toluene)
%TDDFT
	NROOTS 100
	IROOT 1
        IRootMult Singlet
        DOTRANS True
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.06159
END
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 1 S0.xyz
```

To calculate ESA for Triplet using ORCA

```text
! DEF2-SVP CPCM(Toluene)
%SCF
	HFTYP UKS
END
%TDDFT
	NROOTS 100
	IROOT 3
            DOTRANS True
END
%method
        method dft
        functional HYB_GGA_XC_LRC_WPBEH
	ExtParamXC "_omega" 0.06159
END
%maxcore 5000
%pal nprocs 16 end
* XYZFILE 0 3 S0.xyz
```

## **3. Conclusion**

Transient Absorption arises from the third-order polarization because it is the result of the system's nonlinear response to the applied pump and probe fields. The third-order nonlinear interaction provides a detailed mapping of the system's excited-state dynamics, allowing us to probe the system’s behavior on ultrafast time scales, often in the femtosecond to picosecond range.

The polarization we observe at the probe frequency is directly related to the absorption characteristics of the system and reflects contributions from ground-state bleaching, stimulated emission, and excited-state absorption. Each of these processes encodes important information about the molecular system, which can be extracted through careful analysis of the time-resolved data.
