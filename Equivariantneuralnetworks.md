# Introduction

Equivariant neural networks have emerged as a powerful paradigm for modeling physical systems whose underlying laws respect symmetries. In three-dimensional space, many scientific tasks molecular property prediction, force field modeling, and electronic structure learning require architectures that are equivariant under the Euclidean group E(3): rotations, translations, and inversions. The e3nn (“E3 Equivariant Neural Networks”) library provides building blocks for constructing such architectures in PyTorch, enabling models that inherently respect these symmetries. In this post, we delve into the rationale and mathematical foundations of e3nn, illustrate its core components, and build intuitive understanding of how equivariance is enforced at every layer.

## Why Equivariance Matters

### The Curse of Ignoring Symmetry

A standard neural network treats atom coordinates merely as node features; its output generally changes unpredictably if you rotate or translate the input geometry. Yet physical observables transform predictably:

- Scalar properties (e.g., total energy) are invariant under rotations: no matter how you rotate a molecule, its energy stays the same.
- Vector properties (e.g., dipole moment) rotate just like arrows: if you spin the molecule, the dipole arrow spins with it.
- Tensorial quantities (e.g., polarizability) follow higher-order transformation rules.

Failing to bake in these symmetries forces the network to learn them from data—dramatically increasing the number of examples needed and risking physically inconsistent predictions.

### Equivariance as a Design Principle

An equivariant function  satisfies

$$
f(g \cdot x) = \rho(g) f(x)
$$

for every symmetry operation $g$ (e.g., a rotation) and its representation $\rho(g)$ on the output. Concretely, rotating the input positions by a rotation matrix $R$ leads to rotating vector outputs by the same $R$, while leaving scalar outputs unchanged.

## Mathematical Foundations (Intuitive)

To make equivariance concrete, we break down features into pieces that we know how to rotate—much like expressing a 3D shape in terms of simple building blocks.

### 1. Scalars and Vectors: Our Building Blocks

- Scalars ($\ell=0$): Just a number, like temperature at a point. Rotating the system does nothing to a scalar; it stays the same.
- Vectors ($\ell=1$): Like an arrow with 3 components. If you apply a rotation matrix $R$ to coordinates, the vector transforms by multiplying: $v \mapsto Rv$.

**Analogy**: Imagine a wind arrow drawn on a globe. If you spin the globe, the arrow spins accordingly—its length and shape stay the same, only its orientation changes.

### 2. Higher‑Order Tensors: Beyond Arrows

- Rank‑2 tensors ($\ell=2$): Think of them as ellipsoids (like a stretched sphere). Under rotation, every axis of the ellipsoid rotates together.

In general, an object of degree $\ell$ has $2\ell + 1$ components. For $\ell=2$, we have 5 independent numbers describing the ellipsoid.

### 3. Direct Sum of Irreducible Pieces

In e3nn, each atomic feature is a stack (direct sum) of several of these pieces:

For example, "4x0e + 8x1o + 2x2e" means:

- Four scalar-even features (4×1 numbers),
- Eight vector-odd features (8×3 numbers),
- Two rank‑2-even features (2×5 numbers).

This combined feature has length $4 + 8 \cdot 3 + 2 \cdot 5 = 38$. Each piece rotates exactly as physics dictates.

### 4. Combining Features: The Clebsch–Gordan Analogy

To let features interact—like combining two arrows or fusing arrows with ellipsoids—we use the Clebsch–Gordan tensor product. Mathematically, combining an $\ell_1$ piece with an $\ell_2$ piece can produce all $\ell$ values between $|\ell_1−\ell_2|$ and $\ell_1+\ell_2$.

**Simple Example: Adding two arrows ($\ell_1=1$, $\ell_2=1$):**

- You can form a scalar by dotting them: $a \cdot b$ ($\ell=0$).
- You can form another arrow ($\ell=1$) by crossing them: $a \times b$ (gives three components).
- You can form an ellipsoid‑like quantity ($\ell=2$) by taking the symmetric part: $\text{sym}(a \otimes b)$.

Each of these combinations transforms correctly under rotation because they follow the rules of vector algebra you learned in undergraduate physics.

### 5. Nonlinearity and Learnable Mixing

In a standard neural net you apply a ReLU or sigmoid. In an equivariant net, you apply nonlinearities that respect rotation:

- **Mix components of the same $\ell$**: You can apply an ordinary neural network to the scalar coefficients of each irrep independently.
- **Recombine with CG products**: You use distance‑based weights (from a small MLP on the distance) multiplied by spherical harmonics of the direction vector. This ensures messages passed along edges carry geometric information in an equivariant way.

## Core Components of e3nn

- **Irreps**: Specify how many scalars, vectors, tensors, etc., your features contain.
- **Linear (Tensor-Product) Layers**: Perform all allowed Clebsch–Gordan couplings at once to mix features.
- **Radial Networks**: Small MLPs that take inter-atomic distance $r$ and output weights for combining features.
- **Message Passing**: Use the above to pass equivariant messages along edges, then pool or read out.

## Intuition and Practical Tips

- **Data Efficiency**: Hard‑coding symmetry means less data needed to learn physical laws.
- **Physical Interpretability**: Features align with scalars, arrows, and ellipsoids you know from physics.
- **Modularity**: You can adjust how many arrows or ellipsoids to include based on your problem’s complexity.

```python
from e3nn.o3 import Irreps, FullyConnectedNet, Linear
from e3nn.nn.models.gate_points_message_passing import MessagePassing

# Define irreps
node_irreps = Irreps("8x0e + 8x1o")  # 8 scalars, 8 vectors
edge_irreps = Irreps("1x0e + 1x1o")  # distance scalar + direction vector

# Radial network: r -> weights
radial = FullyConnectedNet([1, 64, len(edge_irreps)], activations=torch.relu)

# Equivariant message-passing layer
conv = MessagePassing(
    irreps_node_input=node_irreps,
    irreps_node_output=node_irreps,
    irreps_edge_attr=edge_irreps,
    radial=radial,
    sc_dict={(0,0,0):True, (1,1,0):True, (1,1,2):True}
)

# Forward: conv(pos, node_features, edges, edge_attr)
```

Here, combining an arrow with another arrow or a scalar uses exactly the vector operations you know (dot, cross) and extends them to higher‑order features—all while guaranteeing correct rotation behavior.


## Energy Prediction (Scalar Output)

This example builds a simple equivariant message-passing network to predict a scalar energy from atomic positions and numbers.

```python
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn.o3 import Irreps, FullyConnectedNet, Linear
from e3nn.nn.models.gate_points_message_passing import MessagePassing

# 1. Define irreps
node_irreps = Irreps("4x0e + 8x1o")     # 4 scalars + 8 vectors per atom
edge_irreps = Irreps("1x0e + 1x1o")     # distance scalar + direction vector

# 2. Radial network: maps distance -> weights for tensor products
def make_radial():
    return FullyConnectedNet([1, 32, len(edge_irreps)], activation=torch.relu)

# 3. Equivariant MessagePassing block
def make_conv():
    return MessagePassing(
        irreps_node_input=node_irreps,
        irreps_node_output=node_irreps,
        irreps_edge_attr=edge_irreps,
        radial=make_radial(),
        sc_dict={(0, 0, 0): True, (1, 1, 0): True, (1, 1, 2): True}
    )

class EnergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.readout = torch.nn.Linear(node_irreps.dim, 1)

    def forward(self, data):
        x = data.z.view(-1,1).float()  # atomic numbers as scalar feature
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # First equivariant conv
        x = self.conv1(pos, x, edge_index, edge_attr)
        # Second conv
        x = self.conv2(pos, x, edge_index, edge_attr)
        # Global pooling: average over atoms
        x = torch_scatter.scatter_mean(x, data.batch, dim=0)
        # Predict scalar energy
        energy = self.readout(x)
        return energy.view(-1)

# Example usage with a single methane molecule
atoms = [(6, (0,0,0)), (1, (0.63,0.63,0.63)), (1, (-0.63,-0.63,0.63)), (1, (-0.63,0.63,-0.63)), (1, (0.63,-0.63,-0.63))]
Z = torch.tensor([el for el, _ in atoms], dtype=torch.long)
pos = torch.tensor([coord for _, coord in atoms], dtype=torch.float)
# build edges (e.g., complete graph)
from torch_geometric.nn import radius_graph
edge_index = radius_graph(pos, r=2.0)
# compute edge_attr: radial and spherical harmonics (example with zeros)
edge_attr = torch.zeros(edge_index.size(1), edge_irreps.dim)

data = Data(z=Z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
loader = DataLoader([data], batch_size=1)

model = EnergyModel()
for batch in loader:
    energy = model(batch)
    print("Predicted energy (a.u.):", energy.item())
```

## Dipole Moment Prediction (Vector Output)

For vector outputs, we include an ℓ=1 output irrep so the network predicts a 3D vector (e.g., dipole):

```python
from e3nn.o3 import Irreps
from e3nn.nn.models.gate_points_message_passing import MessagePassing
import torch
from torch_geometric.data import Data, DataLoader

# Node features: scalars only
node_irreps = Irreps("8x0e")
# Edge features: scalar + vector
edge_irreps = Irreps("1x0e + 1x1o")
# Output: one vector (dipole)
out_irreps = Irreps("1x1e")

# Build equivariant conv layers as before
# ... reuse make_radial and make_conv with updated irreps ...

class DipoleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = make_conv()  # defined with new irreps
        # linear readout mapping node irreps -> output vector
        self.readout = Linear(node_irreps, out_irreps)

    def forward(self, data):
        x = data.z.float().view(-1,1)
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.conv(pos, x, edge_index, edge_attr)
        # pool to get molecular feature
        from torch_scatter import scatter_mean
        x = scatter_mean(x, data.batch, dim=0)
        # predict dipole vector
        dipole = self.readout(x)  # shape (batch, 3)
        return dipole

# Example usage omitted for brevity; similar to energy example.
```
more [example](https://www.youtube.com/watch?v=q9EwZsHY1sk)

![image](https://github.com/user-attachments/assets/7dd7f7b4-5394-4ca5-a974-bd5355786cb8)

