# Learning Electronic Coupling with Δ-Equivariant Graph Networks

## Abstract.

Electronic coupling $V_{ref}$ between two molecules is a key ingredient in charge‐transfer rates, photophysics, and materials design. Here we walk through an end-to-end Python implementation that (1) reads dimer geometries and quantum‐chemical descriptors, (2) constructs an equivariant message-passing graph neural network using e3nn ($\ell_{\text{max}} = 3$), (3) leverages Δ-learning to predict the correction from a cheap baseline $V_0$ to the target $V_{ref}$, and (4) trains with a distance-weighted loss and one-cycle learning‐rate schedule.

## 1. Background: Why Δ-Learning & Equivariance?

**Electronic Coupling $V_{ref}$.** Obtained from e.g. Gaussian‐STO-3G Mulliken overlap integrals or more expensive post­Hartree–Fock. We denote the baseline STO-3G result as $V_0$, and the “true” reference coupling (e.g. from a larger basis or higher‐level method) as $V_{ref}$.

**Δ-Learning.** Instead of learning $V_{ref}$ from scratch, we teach the network to predict

$$
\Delta V = V_{ref} - V_0
$$

which often has smaller magnitude and smoother dependence on geometry, making learning faster and more accurate.

**Equivariance with e3nn.** Electronic couplings depend on molecular orientation in space. An equivariant network ensures that if you rotate the entire dimer, the learned features and ultimately $V_{ref}$ respond correctly (i.e. scalar outputs stay invariant, vector/tensor features rotate accordingly). This dramatically improves data efficiency versus plain fingerprints.

## 2. Data Preparation

### 2.1. PDB Parsing
```python
def read_pdb(path: Path):
    elems, xyz = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                token = line[76:78].strip() or line[12:16].strip()
                atom = ''.join(ch for ch in token if ch.isalpha()).capitalize()
                elems.append(PT[atom])
                xyz.append([…, …, …])
    return torch.tensor(elems), torch.tensor(xyz)
```
