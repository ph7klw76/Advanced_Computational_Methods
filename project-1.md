# 1) Purpose
It preprocesses dimer systems (two molecules A and B) into batched PyTorch Geometric graphs and saves them with metadata to a single file (e.g., `cached_graphs.pt`). Inputs are per-system rows in `meta.xlsx` plus per-molecule PDB files and per-dimer NPZ files. Outputs are `Data` objects (one per row) with atom features, edges (including distances and spherical harmonics), global features, and training targets.

Inline math is enclosed with single `$`, as seen in the molecule labels $A$ and $B$.

For example, if the spherical harmonics are used in the edge features, they may take the form:

$$
Y_\ell^m(\theta, \phi)
$$

Where:
- $\ell$ is the degree (angular momentum),
- $m$ is the order,
- $\theta$ and $\phi$ are the spherical coordinate angles.

Distances $d_{ij}$ between atoms $i$ and $j$ may be included in the edge attributes as well.

The overall graph structure is represented using batched `Data` objects, where each graph corresponds to a dimer system described in a row of `meta.xlsx`.

# 2) Key inputs & units

Directory layout:
in the chosen root_dir:

- meta.xlsx (table of dimers)
- IDA.pdb, IDB.pdb (cartesians for molecules A and B)
- an npz file per dimer (properties)

meta.xlsx required columns: molecule a, molecule b, npz, r, θ, v(0), vref.

Units & conversions:

- Distance: R is given in nm, converted to Å by ×10, then multiplied by an internal scale (e.g., 0.1) to define model units.
- Dipole in Debye → $e·\text{Å}$ via ×0.20819434, then ×(internal scale).
- Quadrupole in Debye·Å → $e·\text{Å}^2$ via the same factor, then ×(internal scale) $^2$.

These factors are defined near the top and used uniformly, ensuring you can independently verify unit handling.

# 3) High-level pipeline (per row of meta.xlsx)

Read molecules IDA.pdb and IDB.pdb, parse element symbols robustly (using ATOM/HETATM, the PDB element column if present, then fallbacks from the atom name), and extract atomic numbers Z and positions X. The parser enforces shape/format checks and throws clear errors on malformed coordinates. 

Center each monomer at its own centroid (subtract per-molecule mean). This removes arbitrary absolute origins and keeps local geometry intact. 

Scale to internal units by multiplying all positions by internal_angstrom_scale. This lets you pick numerically convenient units for the model (e.g., 0.1 so bond lengths ~1.0). 

Place the dimer by translating molecule B along the +z axis by R_internal = R_nm × 10 Å/nm × internal_angstrom_scale

Angle $\theta$ is stored as a global scalar (in radians) rather than actively rotating atoms here. (A rotation helper exists but isn’t used in this script.) 

Load NPZ properties: Mulliken populations (s, p, d columns), participation ratio pr, dipole vector (3), and quadrupole (6 independent comps). It validates shapes and falls back to zeros only for a very specific dipole edge-case (scalar encoded NPZ). 

Build node features: for each atom i, 

atomic_node_attr[i] = [Mulliken_s, Mulliken_p, Mulliken_d]

It also checks that Mulliken rows match the atom count. 
Field features (node-invariant “per-dimer” features):
Dipole → scaled internal units (3 numbers).
Quadrupole → trace-free tensor, then mapped to 5 real spherical components $l=2$:

$$
\text{tr} = \frac{Q_{xx} + Q_{yy} + Q_{zz}}{3}, \quad Q_{\alpha\alpha}^t = Q_{\alpha\alpha} - \text{tr}
$$

$$
c_0 = \frac{2Q_{zz}^t - Q_{xx}^t - Q_{yy}^t}{\sqrt{6}}, \quad 
c_1 = 2Q_{xz}, \quad 
c_2 = 2Q_{yz}
$$

$$
c_3 = \frac{Q_{xx}^t - Q_{yy}^t}{2}, \quad 
c_4 = 2Q_{xy}
$$

The final field_node_features is 3 (dipole) + 5 (quadrupole) = 8 values. (A general SO(3) linear transform helper transform_by_matrix_preprocess exists for rotating such irreps features, but it’s not invoked here.)

Global features (u): [R_internal, theta_radians, V0, pr]. These are graph-level scalars helpful for conditioning the model. 

Targets:

$$
y = \Delta V = V_\text{ref} - V_0 \quad \text{(the learning target)}
$$

y_true_vref = Vref (stored for reference/evaluation).

This design lets a downstream model learn corrections while still tracking the absolute reference. 

Neighborhood graph: builds undirected edges with radius_graph using a physical cutoff × internal scale → an internal cutoff. Edge attributes include:

distance (1 scalar per edge)

spherical harmonics (up to $l=3$, “4x0e+4x1o+4x2e+4x3o”) computed from relative vectors and pre-normalized, giving an equivariant basis for message passing. If there are no edges, empty tensors are created with correct shapes/dtypes. 

# 4) What gets saved (schema)

The script aggregates all per-row Data objects into a list and writes:

```python
{
  "graphs": [Data, Data, ...],  # one per meta.xlsx row
  "metadata": {
     "INT_ANGSTROM_SCALE": internal_angstrom_scale,
     "internal_cutoff_val": physical_cutoff * internal_angstrom_scale,
     "physical_cutoff_val": physical_cutoff,
     "IRREPS_EDGE_SH_PRECOMPUTE": "4x0e+4x1o+4x2e+4x3o"
  }
}
```

Each Data contains (names as stored):

```text
z_atoms (N,): torch.long atomic numbers
pos_atoms (N,3): positions in internal units
atomic_node_attr (N,3): Mulliken [s,p,d]
edge_index_atoms (2,E): source→target indices
edge_attr_atoms (E,1): distances
edge_attr_sh (E, L): spherical harmonics (L = dim of specified irreps)
field_node_features (8,): dipole(3) + real-sph quadrupole(5)
u (4,): [R_internal, θ(rad), V0, pr]
y (1,): ΔV = Vref − V0
y_true_vref (1,): Vref
R_internal_val (1,)
book-keeping: id_a, id_b, npz_file, num_atomic_nodes
All of this is written via torch.save(...) to the specified output path.
```

# 5) Error handling & validations

Missing files: raises on missing meta.xlsx, missing PDBs, or missing NPZ.

Shape checks: enforces Mulliken vs atom counts, dipole (3,), quadrupole (6,), and ensures edge tensors are well-typed.

Parsing safeguards: PDB element symbol extraction uses the standard element column when present; otherwise, it falls back to well-defined heuristics on the atom name (handling one- and two-letter symbols).

Graceful skipping: during caching, any row that throws an exception is logged and skipped, and the run continues. If nothing succeeds, the script refuses to write an empty file.
These behaviors are all visible in the code paths of read_pdb_preprocess, __getitem__, and the main loop in build_cache(...). 


# 6) Design rationale 

Center each monomer: eliminates dependence on absolute coordinates; only relative placement (R, θ) matters.

Internal scaling: keeps numeric ranges stable for learning and for equivariant bases.

Cutoff graph: mirrors local chemical interactions; radius_graph is standard for molecular GNNs.

Spherical harmonics features: essential for rotation-equivariant message passing (e3nn/EGNN-style). Precomputing them at preprocessing time reduces runtime overhead later.

$\Delta V$ target: learning a correction to V0 is often more stable than predicting Vref directly, while still retaining Vref for metrics. 


# 7) How to run it 

Update the parameters at the bottom (or call build_cache(...) from another script):

```python
python preprocess_and_cache.py
```

With:

param_root_dir_str: path containing meta.xlsx, PDBs, and NPZs
param_physical_cutoff: e.g., 12.0 (Å)
param_internal_angstrom_scale: e.g., 0.1
param_outfile: e.g., cached_graphs.pt
The script logs how many rows it found, iterates with a progress bar, and writes the file if at least one graph succeeds.


# Flow chart

```text
┌─────────────────────────────────────────────────────────────────┐
│                         INPUTS (per row)                        │
│  meta.xlsx: { id_a, id_b, npz_file, R_nm, theta, V0, Vref, pr } │
│  PDB files: IDA.pdb, IDB.pdb                                     │
│  NPZ file:  mulliken_s,p,d ; dipole(3) ; quadrupole(6) ; pr      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               1) READ & PARSE MOLECULES (A and B)               │
│  - Extract element symbols → atomic numbers Z_A, Z_B  (N_A,N_B) │
│  - Extract Cartesian coords X_A, X_B                  (N_A,3)…  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│     2) CENTER & SCALE (internal units for numerical stability)  │
│  - Center each monomer:  X_A←X_A-mean(X_A) ; X_B←X_B-mean(X_B)  │
│  - Scale by internal_angstrom_scale                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            3) PLACE DIMER USING GLOBAL GEOMETRY (R,θ)           │
│  - Convert R_nm → R_internal (nm→Å→internal)                    │
│  - Translate B:  X_B ← X_B + (0,0,R_internal)                   │
│  - Store θ (radians) as a global scalar (no rotation applied)   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│             4) LOAD PER-DIMER FIELDS & PROPERTIES (NPZ)         │
│  - Mulliken populations → per-atom features (s,p,d)             │
│  - Dipole(3)  in Debye → internal units                         │
│  - Quadrupole(6) in Debye·Å → trace-free → 5 real-sph comps     │
│  - pr (participation ratio), V0, Vref                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5) BUILD ATOMIC NODE FEATURES                 │
│  - Concatenate monomers:                                        │
│      Z = [Z_A ; Z_B]                     → z_atoms        (N,)  │
│      X = [X_A ; X_B]                     → pos_atoms     (N,3)  │
│      Mulliken per-atom s,p,d             → atomic_node   (N,3)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            6) NEIGHBOR GRAPH (local chemical structure)         │
│  - radius_graph(X, cutoff_internal) → edge_index_atoms   (2,E)  │
│  - dist per edge ||x_j - x_i||         → edge_attr_atoms  (E,1) │
│  - unit vectors per edge → spherical harmonics up to l=3        │
│                                  → edge_attr_sh          (E,L)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     7) GLOBAL/FIELD FEATURES                    │
│  - field_node_features = [dipole(3), quadrupole_sph(5)]  (8,)   │
│  - u = [R_internal, theta, V0, pr]                        (4,)   │
│  - Targets: y = (Vref - V0) (ΔV),  y_true_vref = Vref     (1,), (1,) │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│      8) PACKAGE INTO A TORCH GEOMETRIC Data OBJECT (per row)    │
│  Data = {                                                       │
│    z_atoms (N,), pos_atoms (N,3), atomic_node_attr (N,3),       │
│    edge_index_atoms (2,E), edge_attr_atoms (E,1), edge_attr_sh (E,L),│
│    field_node_features (8,), u (4,), y (1,), y_true_vref (1,),  │
│    R_internal_val (1,), bookkeeping ids                         │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│        9) ACCUMULATE ALL Data OBJECTS → SAVE CACHE FILE         │
│  graphs = [Data_1, Data_2, …]                                   │
│  metadata = {INT_ANGSTROM_SCALE, physical_cutoff, internal_cutoff,│
│              IRREPS spec for SH}                                │
│  torch.save({"graphs": graphs, "metadata": metadata}, outfile)  │
└─────────────────────────────────────────────────────────────────┘
```



