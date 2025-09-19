# 1) Purpose & high-level idea

The program trains a rotation-aware graph neural network (GNN) to predict a correction $\Delta V$ to a baseline value $V_0$ so that the final prediction of $V_\text{ref}$ is $V_0 + \Delta V$. Each sample is a dimer (two molecules A and B) represented as a point cloud of atoms turned into a message-passing graph. Besides atom-level features, each graph also includes one “field node” per dimer carrying global multipole features (dipole & quadrupole) that interacts with all atoms. Training uses distance-weighted L1 on $V_\text{ref}$, evaluation uses MAPE.

# 2) Constants, units, and irreps (rotational structure)

Float & matmul precision. Default dtype is float32; if available, float32 matmul precision is set to “high”. This stabilizes training without resorting to float64. 

Unit system. Coordinates are handled in Å physically but internally rescaled by INT_ANGSTROM_SCALE = 0.1 so typical bond lengths (≈1 Å) become ≈0.1 in model units. Dimer separation R comes in nm and is converted nm → Å (×10) → internal (×0.1). Dipoles convert Debye → $e·\text{Å}$ (×0.20819434) and are also scaled by 0.1; quadrupoles (Debye·Å) become $e·\text{Å}^2$ and scaled by $0.1^2$. These choices keep numbers in friendly ranges for the equivariant network. 

Irreps (SO(3) representations).

Edge spherical harmonics (geometry basis): "4x0e+4x1o+4x2e+4x3o" up to $l=3$.

Scalar globals $u = [R, \theta, V_0, PR]$: "4x0e".

Field-node features: dipole $l=1$ (odd parity) + quadrupole $l=2$ (even parity) → "1x1o + 1x2e".
These determine how vectors/tensors rotate and how tensor products are constructed. 

# 3) Data layer

You can train from raw files (on-the-fly build) or from a prebuilt cache.

3.1 DimerDataset (raw → graph, with optional augmentation)

This dataset reads one row per dimer from meta.xlsx, loads two PDBs (A and B), and one NPZ (per-dimer properties). It constructs a PyTorch Geometric Data object with:

Atoms.

z_atoms (N,): atomic numbers (H…Ca supported in the built-in PT) parsed robustly from ATOM/HETATM and PDB element/name heuristics.

pos_atoms (N,3): centered (per monomer), scaled (×0.1), with B translated along +z by the requested internal $R$. Optional jitter (Gaussian noise in internal units) and whole-dimer random rotation when augment=True. 

Per-atom attributes.
atomic_node_attr (N,3) = Mulliken populations [s,p,d] from mulliken_spd in NPZ (d=0 if absent). This is later embedded by a tiny MLP. 

Field node features (per graph, not per atom).
field_node_features (8,): concatenation of the dipole vector (converted & scaled) and the quadrupole decomposed into 5 real spherical $l=2$ components after removing the trace (explicit formula in code). If augment=True, these features are rotated consistently with the same rotation matrix used for positions. 

Global scalars.

```text
u (4,) = [R_{\text{internal}}, \theta_{\text{rad}}, V_0, \mathrm{PR}]
```
$\theta$ is stored (in radians) but no atom rotation by $\theta$ is applied here

Targets.

```text
y = (V_{\text{ref}} - V_0)
```

```text
y_true_vref = V_{\text{ref}}
```
(for eval/plots). Also stores R_internal_val for loss weighting. 


Edges (atom–atom).
edge_index_atoms (2,E) from radius_graph(pos, r=internal_cutoff).
edge_attr_atoms (E,1) holds interatomic distances. Spherical harmonics for atoms are not precomputed here (only in the cached/jitter path or on-the-fly in the model), which keeps preprocessing light. 

3.2 CachedDimerDataset (fast load, consistent augmentation)

Loads a torch.save cache containing graphs (list of Data) plus metadata. Options:

Rotation augmentation: rotates positions and transforms pre-stored field_node_features and, if present, edge_attr_sh accordingly.

Jitter augmentation: adds noise, then rebuilds the atom–atom graph and recomputes distances and spherical harmonics so edges remain consistent with the perturbed geometry.
It also reads the cached internal cutoff from metadata to guarantee consistency. 


# 4) Model architecture: DeltaCoupling
## 4.1 Inputs & packing

Each mini-batch is a Data Batch. The model expects:

z_atoms (ΣN,), pos_atoms (ΣN,3), atomic_node_attr (ΣN,3), edge_index_atoms (2,ΣE), and per-graph field_node_features (B, 8), u (B,4). Here B is number of graphs in the batch, ΣN total atoms. 

## 4.2 Node feature layout (irreps)

The learnable node state is one big vector with irreps:

```text
node_features_irreps =  "8x0e  +  1x1o  +  1x2e"
                         ^^^^^     ^^^      ^^^
                       scalars   dipole   quadrupole
```

Atomic nodes: only scalars part is filled initially using an embedding of atomic number Z into 8 scalar channels (nn.Embedding).

Field node (one per graph): only 1x1o (dipole) and 1x2e (quadrupole) slices are filled from field_node_features.
All atoms of the batch and all field nodes are concatenated into a single tensor x_combined. Positions are similarly concatenated: pos_combined = [pos_atoms ; pos_field_nodes], with pos_field_nodes set to the centroid of atoms per graph (one point per graph). A batch vector tracks which graph each row belongs to. 

## 4.3 Graph edges used by the model

The model uses two edge sets:

Atom–atom: taken from the dataset (edge_index_atoms). Spherical harmonics for these edges are either provided (cached jitter path) or computed on-the-fly from pos_atoms. Distances are likewise taken or computed.

Atom–field + field–atom: the model constructs these edges itself so every atom is connected to the single field node of its graph (both directions). Relative positions are from atom to the field node’s position and vice versa; from those it computes spherical harmonics and distances. This is the coupling mechanism that lets the model mix global multipoles with local atomic geometry. 

The two edge sets are concatenated into edge_index_combined, with matching SH features edge_attr_sh_combined and scalar distances edge_attr_distances_combined. 


## 4.4 Convolution block (PointsConvolutionIntegrated)

Each layer performs an equivariant message passing step with three pieces: 


Distance expansion through radial Bessel basis

RadialBesselBasisLayer(num_rbf, cutoff):

$$
\phi_k(r) = \frac{\sin(\omega_k r)}{\omega_k r} \times \frac{1}{2}(1 + \cos(\pi r / \text{cutoff})) \quad \text{for } r \le \text{cutoff}
$$

with 

$$
\omega_k = \frac{k\pi}{\text{cutoff}}
$$

Output has shape (E, num_rbf). This becomes input to a small MLP to produce edge-wise weights for the tensor product (see step 3). 


Two self/linear maps via FullyConnectedTensorProduct (FCTP)

sc: a self connection (residual-like) transforming node features by combining them with node attributes (the embedded Mulliken [s,p,d]) to produce the same irreps as input/output.

lin1: a “pre-mixing” transform of node features (again with node attributes) before they are sent into the edge message op.
These are equivariant linear maps aware of irreps. 


Edge message via learnable Tensor Product (TP)

For each valid path $l_\text{in} \times l_\text{edge} \rightarrow l_\text{out}$ allowed by irreps, the code builds instructions to a tensor product tp. The weights of the TP are not static: a small MLP FullyConnectedNet (input = RBF features) outputs tp.weight_numel scalars that parameterize the tensor product per edge, letting the network modulate messages by interatomic distance.

After computing per-edge messages, it scatter-means them into receivers (edge_dst).

A second FCTP (lin2) maps the aggregated representation, then a learned scalar gate alpha (tanh, produced by FCTP to a scalar irreps) modulates the update:

$$
\text{out} = \underbrace{\text{sc}(x, \text{attr})}_{\text{self}} + \underbrace{\alpha \odot \text{lin2}(\text{aggregate}, \text{attr})}_{\text{message}}
$$

Every two conv layers, a standard residual add is applied (if shapes match). 


After num_conv_layers such blocks, node features are pooled per graph (mean over batch_combined) to get a graph embedding $h$. Then the scalar globals $u = [R, \theta, V_0, \mathrm{PR}]$ are concatenated and passed to a plain MLP → $\Delta V$ (shape (B,)). Final $V_\text{ref}$ during eval is:

$$
\hat{V}_\text{ref} = V_0 + \Delta V
$$

# 5) Losses and metrics

Training loss: distance_weighted_l1_on_vref_loss computes L1 error in Vref space (not ΔV), weighted by 

<img width="464" height="33" alt="image" src="https://github.com/user-attachments/assets/53e58e16-3d37-488f-8ea2-34601f1a3a8d" />


Farther dimers (larger $R$) are emphasized, which is sensible for interaction curves. This uses R_internal_val stored per graph. 


Eval metric: mape_loss_fn on $V_\text{ref}$, with small epsilon in the denominator. Reported as percent. 


# 6) Training/eval engine (run_epoch)

Accepts a DataLoader, the model, and optional optimizer/scheduler.

Ensures data.u has shape (B,4) even if loaded as a flat vector (handles a variety of edge cases cleanly).

AMP (automatic mixed precision): On CUDA, uses torch.amp.autocast with bfloat16 for GPUs with capability ≥ 8.0, else float16; gradients are scaled with GradScaler. This gives speed without sacrificing much stability. 


Forward pass: model returns ΔV, then it reconstructs $\hat{V}_\text{ref}$ for metric computation.

Backward pass: only when training; guards against NaN/Inf losses and uses optimizer.zero_grad(set_to_none=True) + scaled update. LR scheduler steps each batch.

Aggregation: returns mean loss (train) or mean metric (eval) weighted by number of valid graphs actually processed. Any batch that errors is caught, logged, and skipped (robust long-running training). 


# 7) Plotting (plot_regression)

Collects all predicted $V_\text{ref}$ vs. targets over a loader and draws a scatter with the ideal diagonal. It performs the same careful shape handling for u as the training loop and will skip inconsistent batches rather than crash. Saves a PNG if save_path is given. 


# 8) main() orchestration

Arguments & defaults. If no CLI args are present (common when running in an IDE), it uses sane defaults: epochs=1500, batch_size=5, physical_cutoff=12 Å (→ internal 1.2), num_conv_layers=6, num_rbf=50, lr=1e-3, augmentations on, etc. root_dir defaults to a Windows path under .../molecule. 

Cache management.

Looks for <root_dir>/<cache_file_name> (default cached_graphs.pt).

If missing, it tries to build the cache by calling build_cache(root_dir, physical_cutoff, ...) from a preprocess_and_cache.py module. (Note: the file name in your environment should match this import; otherwise it will print a helpful error.)

If cache exists, it validates metadata: internal cutoff and scaling must match current settings, or the program exits with a clear instruction to regenerate the cache (prevents training with inconsistent geometry). 

Dataset choice.

Training: if apply_augmentations=True and augment_cached_train=False, it uses DimerDataset (fresh augmentations each epoch). Otherwise it uses CachedDimerDataset (faster start, optional rotation/jitter on cached graphs).

Validation/Test: always CachedDimerDataset without augmentation for stability. 

Split logic. Robust splitting that works even for tiny datasets:

n<10: hand-crafted edge cases ensure at least training samples and, if possible, a small val/test.

n≥10: 80/10/10 (rounded and clamped) with defensive corrections if counts do not add up.
The result is three Subsets and matching DataLoaders with sensible num_workers, pin_memory, and persistent_workers based on device. 

Model build & compile.

Instantiates DeltaCoupling with the selected RBF/cutoff/num layers.

Optionally sets memory_format=torch.channels_last on CUDA (a mild perf tweak).

Tries torch.compile: on Windows uses "aot_eager"; elsewhere prefers "inductor" on CUDA. Falls back to eager if compilation fails (non-fatal). 

Optimization setup.

Adam(lr) and, if training data exist, OneCycleLR with steps = epochs*len(train_loader) (min 20 steps), warmup pct_start=0.3.

Early stopping: watches val MAPE, patience 250 epochs. Saves best checkpoint on improvement. Logs every 10 epochs (and the first/last). 

Finalization.

Loads best checkpoint (if present).

Evaluates on test (if any), plots test_final_delta_learning.png.

Evaluates on val, plots val_final_delta_learning.png.

Saves final_model_delta_learning.pt regardless. All artifacts go under <root_dir>/{plots,models}. 


