# From Geometry to Predictions: A Computational Thinking Walkthrough of an Equivariant Graph Convolutional  Neural Networking for electronic coupling prediction


```python
from __future__ import annotations

import math
import sys
import argparse
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import  scatter_mean
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_dtype(torch.float32)
DEFAULT_TORCH_DTYPE = torch.get_default_dtype()

if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision("high")

PHYSICAL_UNIT_ANGSTROM = 1.0
INT_ANGSTROM_SCALE = 0.1

NM_TO_PHYSICAL_ANGSTROM = 10.0
DEBYE_TO_E_PHYSICAL_ANGSTROM = 0.20819434

SCALAR_GLOBAL_FEATURES_ORDER = ["R", "theta", "V0", "PR"]
V0_SCALAR_GLOBAL_INDEX = SCALAR_GLOBAL_FEATURES_ORDER.index("V0")
SCALAR_GLOBAL_IRREPS_STR = f"{len(SCALAR_GLOBAL_FEATURES_ORDER)}x0e"
SCALAR_GLOBAL_IRREPS_DIM = o3.Irreps(SCALAR_GLOBAL_IRREPS_STR).dim

FIELD_NODE_IRREPS = o3.Irreps("1x1o + 1x2e")
FIELD_NODE_IRREPS_DIM = FIELD_NODE_IRREPS.dim

IRREPS_EDGE_SH_PRECOMPUTE = o3.Irreps("4x0e+4x1o+4x2e+4x3o")

RUN_FROM_IDE = len(sys.argv) == 1

SCRIPT_DIR_STR = "C:/Users/Woon/Documents/DICC/HK/HK7/acceptor/low-level"
SCRIPT_DIR = Path(SCRIPT_DIR_STR)
DEFAULT_SPYDER_ROOT_DIR = SCRIPT_DIR / "molecule"

DEFAULT_EPOCHS = 1500
DEFAULT_BATCH = 5
DEFAULT_PHYSICAL_CUTOFF = 12.0
DEFAULT_LR = 1e-3
DEFAULT_NUM_RBF = 50
DEFAULT_NUM_CONV_LAYERS = 6
PHYSICAL_JITTER_STRENGTH = 0.02
MAPE_EPSILON = 1e-7
BOOST_EXP  = 6
BOOST_BASE = 1.0

PT: dict[str,int] = {}
for Z, sym in enumerate(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca".split(), 1):
    PT[sym] = Z

@torch.jit.script
def random_rotation_matrix(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    q = torch.randn(4, device=device, dtype=dtype)
    norm = q.norm() + 1e-8
    q = q / norm
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R00 = 1 - 2*(y*y + z*z); R01 = 2*(x*y - w*z);   R02 = 2*(x*z + w*y)
    R10 = 2*(x*y + w*z);   R11 = 1 - 2*(x*x + z*z); R12 = 2*(y*z - w*x)
    R20 = 2*(x*z - w*y);   R21 = 2*(y*z + w*x);   R22 = 1 - 2*(x*x + y*y)
    
    row0 = torch.stack([R00, R01, R02]); row1 = torch.stack([R10, R11, R12]); row2 = torch.stack([R20, R21, R22])
    R_candidate = torch.stack([row0, row1, row2])
    if torch.linalg.det(R_candidate) < 0: R_candidate[:, 2] *= -1
    return R_candidate

@torch.no_grad()
def transform_by_matrix(
    irreps: o3.Irreps,
    feats: torch.Tensor,
    rotation: torch.Tensor,
    *,
    check: bool = False, 
) -> torch.Tensor:
    if rotation.shape != (3, 3):
        raise ValueError("`rotation` must have shape (3, 3)")
    
    original_shape = feats.shape
    if feats.ndim == 0 or feats.shape[-1] != irreps.dim:
        raise ValueError(f"Last dimension of feats ({feats.shape[-1]}) must match irreps.dim ({irreps.dim})")

    if feats.ndim > 1:
        feats_matrix = feats.reshape(-1, irreps.dim)
    else:
        feats_matrix = feats.unsqueeze(0)

    D = irreps.D_from_matrix(rotation)
    if check and not torch.allclose(D @ D.T, torch.eye(D.shape[0], dtype=D.dtype, device=D.device), atol=1e-4):
        print("Warning: D @ D.T is not close to identity in transform_by_matrix.")
    
    transformed_feats_matrix = torch.matmul(feats_matrix, D.T)
    
    if feats.ndim > 1:
        return transformed_feats_matrix.reshape(*original_shape[:-1], irreps.dim)
    else:
        return transformed_feats_matrix.squeeze(0)

def decompose_quadrupole_to_real_spherical(q_vec: torch.Tensor) -> torch.Tensor:
    if q_vec.ndim < 1 or q_vec.shape[-1] != 6:
        raise ValueError(f"Expected last dimension = 6, got {q_vec.shape}")
    Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = q_vec[...,0], q_vec[...,1], q_vec[...,2], q_vec[...,3], q_vec[...,4], q_vec[...,5]
    tr = (Qxx + Qyy + Qzz) / 3.0
    Qxx_t, Qyy_t, Qzz_t = Qxx - tr, Qyy - tr, Qzz - tr
    c0 = (2 * Qzz_t - Qxx_t - Qyy_t) / math.sqrt(6.0)
    c1, c2 = math.sqrt(2.0) * Qxz, math.sqrt(2.0) * Qyz
    c3, c4 = (Qxx_t - Qyy_t) / math.sqrt(2.0), math.sqrt(2.0) * Qxy
    return torch.stack([c0, c1, c2, c3, c4], dim=-1).to(q_vec.device)

def read_pdb(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    elems = []
    xyz = []
    if not path.exists(): raise FileNotFoundError(f"PDB missing: {path}")
    with path.open() as fh:
        for line_idx, line in enumerate(fh):
            if line.startswith(("ATOM  ", "HETATM")):
                atom_name_field = line[12:16].strip()
                elem_symbol = ""
                if len(line) >= 78:
                    elem_field_std = line[76:78].strip()
                    if elem_field_std:
                        cap_elem_field_std = elem_field_std[0].upper() + (elem_field_std[1:].lower() if len(elem_field_std)>1 else "")
                        if cap_elem_field_std in PT: elem_symbol = cap_elem_field_std
                if (not elem_symbol or elem_symbol not in PT) and atom_name_field:
                    if len(atom_name_field) >= 2 and atom_name_field[0].isalpha():
                        potential_sym_2_cand = atom_name_field[:2].capitalize() if atom_name_field[1].islower() else atom_name_field[0].capitalize() + atom_name_field[1].lower()
                        if potential_sym_2_cand in PT: elem_symbol = potential_sym_2_cand
                    if not elem_symbol and atom_name_field[0].isalpha():
                        potential_sym_1 = atom_name_field[0].capitalize()
                        if potential_sym_1 in PT: elem_symbol = potential_sym_1
                if elem_symbol and elem_symbol in PT:
                    elems.append(PT[elem_symbol])
                    try: xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except ValueError: raise ValueError(f"Could not parse coordinates for atom {elem_symbol} in {path} at line {line_idx+1}: {line.strip()}")
    if not elems: raise ValueError(f"No known atoms found or parsed in PDB: {path}")
    return torch.tensor(elems, dtype=torch.long), torch.tensor(xyz, dtype=DEFAULT_TORCH_DTYPE)

class RadialBesselBasisLayer(torch.nn.Module):
    def __init__(self, num_rbf: int, cutoff: float, learnable_freqs: bool = False, device=None, dtype=None):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        freq_dtype = torch.float64
        final_dtype = dtype if dtype is not None else DEFAULT_TORCH_DTYPE
        frequencies = torch.arange(1, num_rbf + 1, device=device, dtype=freq_dtype) * (math.pi / self.cutoff)
        if learnable_freqs: self.frequencies = nn.Parameter(frequencies.to(dtype=final_dtype))
        else: self.register_buffer('frequencies', frequencies)

    def _cosine_cutoff(self, distances: torch.Tensor) -> torch.Tensor:
        mask = (distances <= self.cutoff).float()
        return mask * (0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.numel() == 0: return torch.empty((0, self.num_rbf), device=distances.device, dtype=distances.dtype)
        distances_shaped = distances[:, None] if distances.ndim == 1 else distances
        current_frequencies = self.frequencies.to(device=distances_shaped.device, dtype=distances_shaped.dtype)
        freq_dist = distances_shaped * current_frequencies.unsqueeze(0)
        rbf_values = torch.where(torch.abs(freq_dist) < 1e-6, torch.ones_like(freq_dist), torch.sin(freq_dist) / freq_dist)
        cutoff_factor = self._cosine_cutoff(distances_shaped)
        return cutoff_factor * rbf_values

class DimerDataset(Dataset):
    def __init__(self, root: Path | str , internal_cutoff: float, augment: bool = False, physical_jitter_strength: float = 0.0):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.root_path = Path(root)
        self.internal_cutoff = internal_cutoff
        self.augment = augment
        self.physical_jitter_strength = physical_jitter_strength
        self.internal_jitter_strength = physical_jitter_strength * INT_ANGSTROM_SCALE

        meta_file_path = self.root_path / "meta.xlsx"
        if not meta_file_path.exists(): raise FileNotFoundError(f"meta.xlsx missing in {self.root_path}. Looked for: {meta_file_path}")
        self.df = pd.read_excel(meta_file_path)
        self.df.columns = self.df.columns.str.strip().str.lower()
        required_cols = ["molecule a", "molecule b", "npz", "r", "θ", "v(0)", "vref"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in meta.xlsx. Found columns: {self.df.columns.tolist()}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ida = int(row["molecule a"])
        idb = int(row["molecule b"])
        npz_filename = str(row["npz"])
        pdb_a_path = self.root_path / f"{ida}.pdb"
        pdb_b_path = self.root_path / f"{idb}.pdb"
        Za, Xa_phys = read_pdb(pdb_a_path)
        Zb, Xb_phys = read_pdb(pdb_b_path)
        npz_path = self.root_path / npz_filename
        if not npz_path.exists(): raise FileNotFoundError(f"NPZ file '{npz_filename}' not found at {npz_path}.")
        with np.load(npz_path) as extra_data:
            mulliken_spd_np = extra_data['mulliken_spd'].astype(np.float32)
            participation_ratio = float(extra_data['participation_ratio'])
            dipole_vector_np = extra_data['dipole_vector'].astype(np.float32)
            if dipole_vector_np.shape != (3,):
                if dipole_vector_np.shape == () and np.isscalar(dipole_vector_np.item()): dipole_vector_np = np.zeros(3,dtype=np.float32)
                else: raise ValueError(f"Dipole in {npz_filename} not a 3-vector: shape is {dipole_vector_np.shape}")
            quadrupole_6_vec_np = extra_data['quadrupole'].astype(np.float32)
            if quadrupole_6_vec_np.shape != (6,): raise ValueError(f"Quadrupole in {npz_filename} not a 6-vector: shape is {quadrupole_6_vec_np.shape}")
        
        Xa_phys, Xb_phys = Xa_phys.to(DEFAULT_TORCH_DTYPE), Xb_phys.to(DEFAULT_TORCH_DTYPE)
        Xa_centered_phys, Xb_centered_phys = Xa_phys - Xa_phys.mean(0,keepdim=True), Xb_phys - Xb_phys.mean(0,keepdim=True)
        Xa_internal, Xb_internal = Xa_centered_phys * INT_ANGSTROM_SCALE, Xb_centered_phys * INT_ANGSTROM_SCALE
        
        R_nm_original = float(row["r"])
        R_physical_Angstrom = R_nm_original * NM_TO_PHYSICAL_ANGSTROM
        R_internal_shift_val = R_physical_Angstrom * INT_ANGSTROM_SCALE
        Xb_internal += torch.tensor([0., 0., R_internal_shift_val], dtype=DEFAULT_TORCH_DTYPE)
        pos_atoms_internal, Z_atoms = torch.cat([Xa_internal, Xb_internal]), torch.cat([Za, Zb])
        num_total_atoms = Z_atoms.shape[0]

        if mulliken_spd_np.shape[0]!=num_total_atoms: raise ValueError(f"Mulliken spd count {mulliken_spd_np.shape[0]} in {npz_filename} != PDB atoms {num_total_atoms}")
        s_pop = torch.from_numpy(mulliken_spd_np[:,0]).to(DEFAULT_TORCH_DTYPE).unsqueeze(-1)
        p_pop = torch.from_numpy(mulliken_spd_np[:,1]).to(DEFAULT_TORCH_DTYPE).unsqueeze(-1)
        d_pop = torch.from_numpy(mulliken_spd_np[:,2]).to(DEFAULT_TORCH_DTYPE).unsqueeze(-1) if mulliken_spd_np.shape[1] > 2 else torch.zeros_like(s_pop)
        node_attr_tensor=torch.cat([s_pop,p_pop,d_pop],dim=1)

        dipole_Debye = torch.from_numpy(dipole_vector_np).to(DEFAULT_TORCH_DTYPE)
        dipole_e_Angstrom = dipole_Debye * DEBYE_TO_E_PHYSICAL_ANGSTROM
        dipole_internal = dipole_e_Angstrom * INT_ANGSTROM_SCALE

        quadrupole_Debye_Ang = torch.from_numpy(quadrupole_6_vec_np.flatten()).to(DEFAULT_TORCH_DTYPE)
        quadrupole_e_Angstrom2 = quadrupole_Debye_Ang * DEBYE_TO_E_PHYSICAL_ANGSTROM
        quadrupole_internal_prespherical = quadrupole_e_Angstrom2 * (INT_ANGSTROM_SCALE**2)
        
        field_node_features_unrotated = torch.cat([dipole_internal, decompose_quadrupole_to_real_spherical(quadrupole_internal_prespherical)])
        
        theta_rad = math.radians(float(row["θ"]))
        V0_val = float(row["v(0)"])
        Vref_val = float(row["vref"])
        scalar_global_features = torch.tensor([R_internal_shift_val, theta_rad, V0_val, participation_ratio], dtype=DEFAULT_TORCH_DTYPE)
        delta_v_target = torch.tensor([Vref_val-V0_val],dtype=DEFAULT_TORCH_DTYPE)
        y_true_vref_for_loss_and_plot = torch.tensor([Vref_val],dtype=DEFAULT_TORCH_DTYPE)
        
        pos_to_augment = pos_atoms_internal.clone()
        rotation_mat = torch.eye(3,dtype=DEFAULT_TORCH_DTYPE,device=pos_to_augment.device)
        
        if self.augment:
            if self.internal_jitter_strength > 0:
                 pos_to_augment += torch.randn_like(pos_to_augment) * self.internal_jitter_strength
            rotation_mat = random_rotation_matrix(device=pos_to_augment.device, dtype=DEFAULT_TORCH_DTYPE)
            pos_to_augment = pos_to_augment @ rotation_mat.T
        
        final_field_node_features = field_node_features_unrotated
        if self.augment:
            dip_part_dim = o3.Irreps("1x1o").dim
            dip_part = field_node_features_unrotated[:dip_part_dim]
            quad_part = field_node_features_unrotated[dip_part_dim:]
            final_field_node_features = torch.cat([
                transform_by_matrix(o3.Irreps("1x1o"),dip_part,rotation_mat, check=False), 
                transform_by_matrix(o3.Irreps("1x2e"),quad_part,rotation_mat, check=False)
            ])
            
        edge_index_atoms = radius_graph(pos_to_augment,r=self.internal_cutoff,loop=False,flow='source_to_target',max_num_neighbors=1000)
        if edge_index_atoms.dtype != torch.long: edge_index_atoms = edge_index_atoms.long()
        edge_attr_distances_atoms = (pos_to_augment[edge_index_atoms[1]] - pos_to_augment[edge_index_atoms[0]]).norm(dim=1,keepdim=True) if edge_index_atoms.numel()>0 else torch.empty((0,1),dtype=DEFAULT_TORCH_DTYPE,device=pos_to_augment.device)
        
        data_dict = dict(z_atoms=Z_atoms, pos_atoms=pos_to_augment,
                    atomic_node_attr=node_attr_tensor,
                    edge_index_atoms=edge_index_atoms, edge_attr_atoms=edge_attr_distances_atoms,
                    field_node_features=final_field_node_features,
                    u=scalar_global_features,
                    y=delta_v_target, y_true_vref=y_true_vref_for_loss_and_plot,
                    R_internal_val=torch.tensor([R_internal_shift_val], dtype=DEFAULT_TORCH_DTYPE),
                    id_a=torch.tensor([ida]), id_b=torch.tensor([idb]),
                    npz_file=npz_filename,
                    num_atomic_nodes=torch.tensor([num_total_atoms], dtype=torch.long))
        return Data(**data_dict)

class CachedDimerDataset(Dataset):
    def __init__(self, cache_file: str | Path, augment_with_rotations: bool = False, physical_jitter_strength: float = 0.0):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.augment_with_rotations = augment_with_rotations
        self.physical_jitter_strength = physical_jitter_strength
        self.internal_jitter_strength = physical_jitter_strength * INT_ANGSTROM_SCALE
        
        loaded_data = torch.load(cache_file, weights_only=False)
        if isinstance(loaded_data, dict) and 'graphs' in loaded_data and 'metadata' in loaded_data:
            self.graphs: list[Data] = loaded_data['graphs']
            self.metadata: dict[str, Any] = loaded_data['metadata']
        elif isinstance(loaded_data, list):
            self.graphs: list[Data] = loaded_data
            self.metadata = {}
            print("Warning: Loaded legacy cache file without metadata. Consider re-generating the cache.")
        else:
            raise TypeError(f"Unknown cache file format for {cache_file}")

        if not isinstance(self.graphs, list):
            self.graphs = [self.graphs]
        
        self.internal_cutoff_from_cache = self.metadata.get('internal_cutoff_val', DEFAULT_PHYSICAL_CUTOFF * INT_ANGSTROM_SCALE)


    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx].clone()
        if self.augment_with_rotations or self.internal_jitter_strength > 0:
            pos_to_augment = data.pos_atoms.clone()
            rotation_mat = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=pos_to_augment.device)
            
            if self.internal_jitter_strength > 0:
                 pos_to_augment += torch.randn_like(pos_to_augment) * self.internal_jitter_strength
            
            if self.augment_with_rotations:
                rotation_mat = random_rotation_matrix(device=pos_to_augment.device, dtype=DEFAULT_TORCH_DTYPE)
                pos_to_augment = pos_to_augment @ rotation_mat.T
            
            data.pos_atoms = pos_to_augment

            if hasattr(data, 'field_node_features'):
                dip_part_dim = o3.Irreps("1x1o").dim
                dip_part = data.field_node_features[:dip_part_dim]
                quad_part = data.field_node_features[dip_part_dim:]
                data.field_node_features = torch.cat([
                    transform_by_matrix(o3.Irreps("1x1o"), dip_part, rotation_mat, check=False),
                    transform_by_matrix(o3.Irreps("1x2e"), quad_part, rotation_mat, check=False)
                ])
            
            if self.internal_jitter_strength > 0: 
                data.edge_index_atoms = radius_graph(data.pos_atoms, r=self.internal_cutoff_from_cache, loop=False, flow='source_to_target', max_num_neighbors=1000)
                if data.edge_index_atoms.dtype != torch.long: data.edge_index_atoms = data.edge_index_atoms.long()

                if data.edge_index_atoms.numel() > 0:
                    rel_pos_aa = data.pos_atoms[data.edge_index_atoms[1]] - data.pos_atoms[data.edge_index_atoms[0]]
                    data.edge_attr_sh = o3.spherical_harmonics(
                        IRREPS_EDGE_SH_PRECOMPUTE, rel_pos_aa, normalize=True, normalization='component'
                    )
                    data.edge_attr_atoms = rel_pos_aa.norm(dim=1, keepdim=True)
                else:
                    data.edge_attr_sh = torch.empty((0, IRREPS_EDGE_SH_PRECOMPUTE.dim), dtype=DEFAULT_TORCH_DTYPE, device=data.pos_atoms.device)
                    data.edge_attr_atoms = torch.empty((0,1), dtype=DEFAULT_TORCH_DTYPE, device=data.pos_atoms.device)
            elif self.augment_with_rotations and hasattr(data, 'edge_attr_sh') and data.edge_attr_sh.numel() > 0 :
                data.edge_attr_sh = transform_by_matrix(IRREPS_EDGE_SH_PRECOMPUTE, data.edge_attr_sh, rotation_mat, check=False)
                if hasattr(data,'edge_index_atoms') and data.edge_index_atoms.numel() > 0 and (not hasattr(data, 'edge_attr_atoms') or data.edge_attr_atoms is None):
                      data.edge_attr_atoms = (data.pos_atoms[data.edge_index_atoms[1]] - data.pos_atoms[data.edge_index_atoms[0]]).norm(dim=1, keepdim=True)

        return data

@compile_mode("script")
class PointsConvolutionIntegrated(torch.nn.Module):
    def __init__(self, irreps_node_input, irreps_node_attr, irreps_edge_sh, irreps_node_output,
                 fc_hidden_dims, num_rbf: int, rbf_cutoff: float, learnable_rbf_freqs: bool = False):
        super().__init__()
        self.irreps_node_input=o3.Irreps(irreps_node_input)
        self.irreps_node_attr=o3.Irreps(irreps_node_attr)
        self.irreps_edge_sh=o3.Irreps(irreps_edge_sh)
        self.irreps_node_output=o3.Irreps(irreps_node_output)
        self.rbf_basis=RadialBesselBasisLayer(num_rbf,rbf_cutoff,learnable_rbf_freqs, dtype=DEFAULT_TORCH_DTYPE)
        fc_input_dim=num_rbf
        self.sc=FullyConnectedTensorProduct(self.irreps_node_input,self.irreps_node_attr,self.irreps_node_output)
        self.lin1=FullyConnectedTensorProduct(self.irreps_node_input,self.irreps_node_attr,self.irreps_node_input)
        
        tp_output_channels_list_for_tp = []
        instructions_for_tp = []
        for i_in1, (mul_1, ir_1) in enumerate(self.irreps_node_input):
            for i_in2, (mul_2, ir_2) in enumerate(self.irreps_edge_sh):
                for ir_out_candidate in ir_1 * ir_2:
                    if ir_out_candidate in self.irreps_node_output or ir_out_candidate.l == 0:
                        i_out = len(tp_output_channels_list_for_tp)
                        tp_output_channels_list_for_tp.append( (mul_1, ir_out_candidate) )
                        instructions_for_tp.append( (i_in1, i_in2, i_out, "uvu", True) )

        if not instructions_for_tp:
            raise ValueError(f"No valid paths for TP from {self.irreps_node_input} x {self.irreps_edge_sh} to {self.irreps_node_output} or scalar.")
        
        irreps_tp_direct_output = o3.Irreps(tp_output_channels_list_for_tp)

        self.tp = o3.TensorProduct(
            self.irreps_node_input, self.irreps_edge_sh, irreps_tp_direct_output, instructions_for_tp,
            internal_weights=False, shared_weights=False
        )
        
        fc_neurons_full_list = [fc_input_dim] + fc_hidden_dims + [self.tp.weight_numel]
        self.fc = FullyConnectedNet(fc_neurons_full_list, F.silu)
        self.lin2 = FullyConnectedTensorProduct(self.tp.irreps_out.simplify(), self.irreps_node_attr, self.irreps_node_output)
        self.alpha = FullyConnectedTensorProduct(self.tp.irreps_out.simplify(), self.irreps_node_attr, "0e")
        with torch.no_grad(): self.alpha.weight.zero_()
        if not (self.alpha.irreps_out.lmax==0 and self.alpha.irreps_out.dim==1): raise AssertionError(f"Alpha FCTP output is not scalar. Got: {self.alpha.irreps_out}")

    def forward(self, node_input, node_attr, edge_sh_attr,
                edge_scalar_distances, batch_info_for_scatter) -> torch.Tensor:
        edge_src = batch_info_for_scatter['edge_src']
        edge_dst = batch_info_for_scatter['edge_dst']
        
        expanded_edge_scalars = self.rbf_basis(edge_scalar_distances)
        weight = self.fc(expanded_edge_scalars)
        
        node_self_connection = self.sc(node_input, node_attr)
        node_features_after_lin1 = self.lin1(node_input, node_attr)

        if edge_src.numel() > 0:
            if node_features_after_lin1.shape[0] == 0: raise ValueError("edge_src non-empty, node_features_after_lin1 0 nodes.")
            gathered_node_features = node_features_after_lin1[edge_src]
            
            edge_message_features = self.tp(gathered_node_features, edge_sh_attr, weight)
            aggregated_node_features = scatter_mean(edge_message_features, edge_dst, dim=0, dim_size=node_input.shape[0])
        else:
            aggregated_node_features = torch.zeros((node_input.shape[0], self.tp.irreps_out.dim),
                                                   device=node_input.device, dtype=node_input.dtype)
        
        node_conv_out_before_alpha = self.lin2(aggregated_node_features, node_attr)
        alpha_scalars = torch.tanh(self.alpha(aggregated_node_features, node_attr))
        m = self.sc.output_mask
        alpha_gate = (1 - m) + alpha_scalars * m
        return node_self_connection + alpha_gate * node_conv_out_before_alpha

class DeltaCoupling(nn.Module):
    SCALAR_GLOBAL_IRREPS_STR_CLS = SCALAR_GLOBAL_IRREPS_STR
    FIELD_NODE_IRREPS_STR_CLS = "1x1o + 1x2e"

    def __init__(self, max_Z: int = max(PT.values()) + 1, num_rbf: int = DEFAULT_NUM_RBF,
                 internal_rbf_cutoff: float = DEFAULT_PHYSICAL_CUTOFF * INT_ANGSTROM_SCALE, 
                 learnable_rbf_freqs: bool = False, num_conv_layers: int = DEFAULT_NUM_CONV_LAYERS):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        self.node_features_irreps = o3.Irreps("8x0e + 1x1o + 1x2e")
        self.atomic_scalar_embed_dim = o3.Irreps("8x0e").dim
        self.embed_atomic_scalar = nn.Embedding(max_Z + 1, self.atomic_scalar_embed_dim)
        self.field_node_irreps = o3.Irreps(DeltaCoupling.FIELD_NODE_IRREPS_STR_CLS)
        self.field_node_dim = self.field_node_irreps.dim
        self.slice_for_atomic_scalar, self.slice_for_field_1o, self.slice_for_field_2e = None, None, None
        current_offset = 0
        for _mul, ir in self.node_features_irreps:
            dim = ir.dim * _mul ; s = slice(current_offset, current_offset + dim)
            if ir.l==0 and ir.p==1 and self.slice_for_atomic_scalar is None and dim==self.atomic_scalar_embed_dim: self.slice_for_atomic_scalar=s
            elif ir.l==1 and ir.p==-1 and self.slice_for_field_1o is None and dim==o3.Irreps("1x1o").dim: self.slice_for_field_1o=s
            elif ir.l==2 and ir.p==1 and self.slice_for_field_2e is None and dim==o3.Irreps("1x2e").dim: self.slice_for_field_2e=s
            current_offset += dim
        self.input_node_attr_irreps = o3.Irreps("3x0e")
        self.node_attr_embedding_dim = 16
        self.embedded_node_attr_irreps = o3.Irreps(f"{self.node_attr_embedding_dim}x0e")
        self.node_attr_mlp = FullyConnectedNet([self.input_node_attr_irreps.dim,32,self.embedded_node_attr_irreps.dim],F.silu)
        self.scalar_global_irreps = o3.Irreps(DeltaCoupling.SCALAR_GLOBAL_IRREPS_STR_CLS)
        self.irreps_edge_sh_for_conv = IRREPS_EDGE_SH_PRECOMPUTE
        fc_hidden_dims = [32, 32]
        self.convs = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            conv = PointsConvolutionIntegrated(
                irreps_node_input=self.node_features_irreps, irreps_node_attr=self.embedded_node_attr_irreps,
                irreps_edge_sh=self.irreps_edge_sh_for_conv, irreps_node_output=self.node_features_irreps,
                fc_hidden_dims=fc_hidden_dims, 
                num_rbf=num_rbf, rbf_cutoff=internal_rbf_cutoff, learnable_rbf_freqs=learnable_rbf_freqs)
            self.convs.append(conv)
        mlp_input_dim = self.node_features_irreps.dim + self.scalar_global_irreps.dim
        self.mlp = nn.Sequential(nn.Linear(mlp_input_dim,128), nn.SiLU(), nn.Linear(128,64), nn.SiLU(), nn.Linear(64,32), nn.SiLU(), nn.Linear(32,1))

    def _build_atom_field_edges(self, ptr, pos_atoms, pos_field_nodes, device, dtype):
        num_graphs = len(ptr) - 1
        if num_graphs == 0 or pos_atoms.numel() == 0 :
            return (torch.empty((2,0), dtype=torch.long, device=device),
                    torch.empty((0, self.irreps_edge_sh_for_conv.dim), dtype=dtype, device=device),
                    torch.empty((0,1), dtype=dtype, device=device))

        atoms_per_graph = ptr[1:] - ptr[:-1]
        graph_indices_for_atoms = torch.repeat_interleave(torch.arange(num_graphs, device=device), atoms_per_graph)
        
        atom_indices_global = torch.arange(ptr[-1], device=device)
        field_node_indices_global_repeated = ptr[-1] + graph_indices_for_atoms

        src_af = atom_indices_global
        dst_af = field_node_indices_global_repeated
        
        src_fa = field_node_indices_global_repeated
        dst_fa = atom_indices_global

        src_combined = torch.cat([src_af, src_fa])
        dst_combined = torch.cat([dst_af, dst_fa])
        
        rel_pos_af = pos_field_nodes[graph_indices_for_atoms] - pos_atoms[atom_indices_global]
        rel_pos_full = torch.cat([rel_pos_af, -rel_pos_af])

        sh = o3.spherical_harmonics(self.irreps_edge_sh_for_conv, rel_pos_full, normalize=True, normalization='component')
        dist = rel_pos_full.norm(dim=1, keepdim=True)

        return torch.stack([src_combined, dst_combined]), sh, dist

    def forward(self, data: Batch) -> torch.Tensor:
        Z_atoms, pos_atoms, atomic_node_attr_input = data.z_atoms, data.pos_atoms, data.atomic_node_attr
        scalar_globals_u_in, field_node_features_batch_in = data.u, data.field_node_features
        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') and data.num_graphs is not None else 0
        num_atomic_nodes_total = Z_atoms.shape[0]
        
        current_field_node_features_batch = field_node_features_batch_in
        current_scalar_globals_u = scalar_globals_u_in
        device = pos_atoms.device # Ensure consistent device for new tensors

        if not hasattr(data, 'field_node_features') or data.field_node_features is None:
            current_field_node_features_batch = torch.empty(num_graphs, self.field_node_dim, device=device, dtype=DEFAULT_TORCH_DTYPE)
        elif current_field_node_features_batch.ndim == 1:
            if num_graphs == 0 and current_field_node_features_batch.numel() == self.field_node_dim:
                num_graphs = 1
                if hasattr(data, 'num_graphs'): data.num_graphs = 1
                current_field_node_features_batch = current_field_node_features_batch.unsqueeze(0)
            elif num_graphs == 1 and current_field_node_features_batch.numel() == self.field_node_dim:
                current_field_node_features_batch = current_field_node_features_batch.unsqueeze(0)
            elif num_graphs > 0 and current_field_node_features_batch.numel() == num_graphs * self.field_node_dim:
                current_field_node_features_batch = current_field_node_features_batch.reshape(num_graphs, self.field_node_dim)
            elif num_graphs == 0 and current_field_node_features_batch.numel() == 0:
                 current_field_node_features_batch = current_field_node_features_batch.reshape(0, self.field_node_dim)
            else:
                pass
        
        if not hasattr(data, 'u') or data.u is None:
            current_scalar_globals_u = torch.empty(num_graphs, self.scalar_global_irreps.dim, device=device, dtype=DEFAULT_TORCH_DTYPE)
        elif current_scalar_globals_u.ndim == 1:
            num_graphs_from_data_attr = data.num_graphs if hasattr(data, 'num_graphs') and data.num_graphs is not None else num_graphs
            if num_graphs_from_data_attr == 0 and current_scalar_globals_u.numel() == self.scalar_global_irreps.dim:
                if hasattr(data, 'num_graphs') and data.num_graphs == 0 : data.num_graphs = 1 
                current_scalar_globals_u = current_scalar_globals_u.unsqueeze(0)
            elif num_graphs_from_data_attr == 1 and current_scalar_globals_u.numel() == self.scalar_global_irreps.dim:
                current_scalar_globals_u = current_scalar_globals_u.unsqueeze(0)
            elif num_graphs_from_data_attr > 0 and current_scalar_globals_u.numel() == num_graphs_from_data_attr * self.scalar_global_irreps.dim:
                current_scalar_globals_u = current_scalar_globals_u.reshape(num_graphs_from_data_attr, self.scalar_global_irreps.dim)
            elif num_graphs_from_data_attr == 0 and current_scalar_globals_u.numel() == 0:
                 current_scalar_globals_u = current_scalar_globals_u.reshape(0, self.scalar_global_irreps.dim)
            else:
                pass
        
        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') and data.num_graphs is not None else num_graphs
        
        x_atomic_scalar_embedded = self.embed_atomic_scalar(Z_atoms)
        x_atoms = torch.zeros(num_atomic_nodes_total,self.node_features_irreps.dim,device=device,dtype=DEFAULT_TORCH_DTYPE)
        if num_atomic_nodes_total > 0 : x_atoms[:, self.slice_for_atomic_scalar] = x_atomic_scalar_embedded
        
        x_field_nodes = torch.zeros(num_graphs,self.node_features_irreps.dim,device=device,dtype=DEFAULT_TORCH_DTYPE)
        if num_graphs > 0:
            dip_part_dim = o3.Irreps("1x1o").dim
            x_field_nodes[:, self.slice_for_field_1o] = current_field_node_features_batch[:,:dip_part_dim]
            x_field_nodes[:, self.slice_for_field_2e] = current_field_node_features_batch[:,dip_part_dim:]
        x_combined = torch.cat([x_atoms, x_field_nodes], dim=0)

        current_ptr_val = data.ptr if hasattr(data,'ptr') and data.ptr is not None else torch.tensor([0,num_atomic_nodes_total],device=device,dtype=torch.long)
        if num_atomic_nodes_total == 0 and num_graphs > 0: current_ptr_val = torch.zeros(num_graphs+1,device=device,dtype=torch.long)
        
        atom_batch_indices = data.batch if hasattr(data,'batch') and data.batch is not None else torch.zeros(num_atomic_nodes_total, dtype=torch.long, device=device)
        if num_atomic_nodes_total == 0 : atom_batch_indices = torch.empty(0, dtype=torch.long, device=device)

        pos_field_nodes = scatter_mean(pos_atoms, atom_batch_indices, dim=0, dim_size=num_graphs) if num_atomic_nodes_total > 0 else torch.zeros(num_graphs, 3, device=device, dtype=DEFAULT_TORCH_DTYPE)
        if num_graphs > 0 and pos_field_nodes.shape[0] != num_graphs :
            temp_pos_field = torch.zeros(num_graphs, 3, device=device, dtype=DEFAULT_TORCH_DTYPE)
            if pos_field_nodes.shape[0] > 0: temp_pos_field[:pos_field_nodes.shape[0]] = pos_field_nodes
            pos_field_nodes = temp_pos_field

        pos_combined = torch.cat([pos_atoms, pos_field_nodes], dim=0)
        
        batch_atoms_val = atom_batch_indices

        batch_field_nodes = torch.arange(num_graphs,device=device,dtype=torch.long)
        batch_combined = torch.cat([batch_atoms_val, batch_field_nodes], dim=0)
        
        embedded_attr_atoms = self.node_attr_mlp(atomic_node_attr_input)
        embedded_attr_field = torch.zeros(num_graphs,self.embedded_node_attr_irreps.dim,device=device,dtype=DEFAULT_TORCH_DTYPE)
        node_attr_combined = torch.cat([embedded_attr_atoms, embedded_attr_field], dim=0)

        edge_index_atoms = data.edge_index_atoms
        
        if hasattr(data, 'edge_attr_sh') and data.edge_attr_sh is not None:
            edge_attr_sh_atoms = data.edge_attr_sh
        elif edge_index_atoms.numel() > 0:
            rel_pos_aa = pos_atoms[edge_index_atoms[1]] - pos_atoms[edge_index_atoms[0]]
            edge_attr_sh_atoms = o3.spherical_harmonics(
                self.irreps_edge_sh_for_conv, rel_pos_aa, normalize=True, normalization='component'
            )
        else:
            edge_attr_sh_atoms = torch.empty((0, self.irreps_edge_sh_for_conv.dim), device=device, dtype=DEFAULT_TORCH_DTYPE)
        
        if hasattr(data,'edge_attr_atoms') and data.edge_attr_atoms is not None:
            edge_attr_distances_atoms_val = data.edge_attr_atoms
        elif edge_index_atoms.numel() > 0:
             edge_attr_distances_atoms_val = (pos_atoms[edge_index_atoms[1]] - pos_atoms[edge_index_atoms[0]]).norm(dim=1,keepdim=True)
        else:
            edge_attr_distances_atoms_val = torch.empty((0,1), device=device, dtype=DEFAULT_TORCH_DTYPE)
        
        edge_index_af, sh_af, dist_af = self._build_atom_field_edges(current_ptr_val, pos_atoms, pos_field_nodes, device=device, dtype=DEFAULT_TORCH_DTYPE)

        if edge_index_atoms.numel() > 0:
            edge_index_combined = torch.cat([edge_index_atoms, edge_index_af], dim=1)
            edge_attr_sh_combined = torch.cat([edge_attr_sh_atoms, sh_af], dim=0)
            edge_attr_distances_combined = torch.cat([edge_attr_distances_atoms_val, dist_af], dim=0)
        else:
            edge_index_combined = edge_index_af
            edge_attr_sh_combined = sh_af
            edge_attr_distances_combined = dist_af
        
        x_conv_out = x_combined
        for i_conv in range(self.num_conv_layers):
            residual_input = x_conv_out 
            batch_info_conv = {'edge_src': edge_index_combined[0], 'edge_dst': edge_index_combined[1]}
            x_conv_out = self.convs[i_conv](
                x_conv_out, node_attr_combined, edge_attr_sh_combined,
                edge_attr_distances_combined, batch_info_conv
            )
            if (i_conv+1)%2==0 and x_conv_out.shape==residual_input.shape: x_conv_out += residual_input
        
        x_scattered = scatter_mean(x_conv_out, batch_combined, dim=0, dim_size=num_graphs)
        
        if num_graphs > 0:
            h = torch.cat([x_scattered, current_scalar_globals_u], dim=1)
        else:
            h = torch.empty((0, self.node_features_irreps.dim + self.scalar_global_irreps.dim), device=device, dtype=DEFAULT_TORCH_DTYPE)
        
        delta_v_pred = self.mlp(h)
        return delta_v_pred.squeeze(-1)

def mape_loss_fn(pred_vref,target_vref,epsilon=MAPE_EPSILON):
    abs_error=torch.abs(target_vref-pred_vref)
    abs_target=torch.abs(target_vref)
    if abs_target.numel() == 0: return torch.tensor(0.0, device=pred_vref.device if pred_vref.numel() > 0 else abs_target.device, dtype=DEFAULT_TORCH_DTYPE) * 100.0
    return torch.mean(abs_error/(abs_target+epsilon))*100.0

def distance_weighted_l1_on_vref_loss(pred_delta_v, target_delta_v, R_internal_batch, V0_batch, graph_cutoff_for_weighting):
    if pred_delta_v.numel() == 0: return torch.tensor(0.0, device=pred_delta_v.device, dtype=DEFAULT_TORCH_DTYPE, requires_grad=True)
    pred_vref, target_vref = V0_batch + pred_delta_v, V0_batch + target_delta_v
    R_internal_batch = R_internal_batch.to(pred_vref.device)
    if R_internal_batch.ndim == 1: R_internal_batch = R_internal_batch.unsqueeze(-1)
    if pred_vref.ndim == 1: pred_vref = pred_vref.unsqueeze(-1)
    if target_vref.ndim == 1: target_vref = target_vref.unsqueeze(-1)
    abs_error_vref = torch.abs(target_vref - pred_vref)
    weights =  BOOST_BASE + (R_internal_batch / graph_cutoff_for_weighting) ** BOOST_EXP
    return torch.mean(weights * abs_error_vref)

def run_epoch(loader,model,optimizer=None,scheduler=None,device="cpu",
              loss_for_backward_fn=None, metric_fn_for_eval=mape_loss_fn,
              is_training:bool=False, dist_weight_cutoff:float=DEFAULT_PHYSICAL_CUTOFF * INT_ANGSTROM_SCALE,
              use_amp: bool = True):
    model.train(is_training)
    total_loss_display, n_graphs_processed = 0.0, 0
    if loader is None or len(loader) == 0: return float('nan')
    
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))

    for batch_idx, data_obj in enumerate(loader):
        try:
            data_obj = data_obj.to(device)
            num_graphs_in_this_batch = data_obj.num_graphs if hasattr(data_obj,'num_graphs') and data_obj.num_graphs is not None else 0
            
            current_u = data_obj.u
            if not hasattr(data_obj, 'u') or data_obj.u is None:
                current_u = torch.empty(num_graphs_in_this_batch, SCALAR_GLOBAL_IRREPS_DIM, device=device, dtype=DEFAULT_TORCH_DTYPE)
            elif current_u.ndim == 1:
                if num_graphs_in_this_batch == 0 and current_u.numel() == SCALAR_GLOBAL_IRREPS_DIM:
                    num_graphs_in_this_batch = 1
                    if hasattr(data_obj, 'num_graphs'): data_obj.num_graphs = 1
                    current_u = current_u.unsqueeze(0)
                elif num_graphs_in_this_batch == 1 and current_u.numel() == SCALAR_GLOBAL_IRREPS_DIM:
                    current_u = current_u.unsqueeze(0)
                elif num_graphs_in_this_batch > 0 and current_u.numel() == num_graphs_in_this_batch * SCALAR_GLOBAL_IRREPS_DIM:
                     current_u = current_u.reshape(num_graphs_in_this_batch, SCALAR_GLOBAL_IRREPS_DIM)
                elif num_graphs_in_this_batch == 0 and current_u.numel() == 0 :
                    current_u = current_u.reshape(0, SCALAR_GLOBAL_IRREPS_DIM)
                else:
                    pass
            data_obj.u = current_u
            
            v0_from_u = data_obj.u[:, V0_SCALAR_GLOBAL_INDEX] if data_obj.u.numel()>0 and data_obj.u.shape[0] > 0 else torch.empty(0,device=device, dtype=DEFAULT_TORCH_DTYPE)
            
            amp_dtype = torch.float16
            if device.type == 'cuda' and use_amp:
                capability = torch.cuda.get_device_capability(device)
                if capability[0] >= 8: amp_dtype = torch.bfloat16
            
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == 'cuda'):
                with torch.set_grad_enabled(is_training):
                    pred_delta_v = model(data_obj)
                    actual_processed_graphs = pred_delta_v.shape[0]
                    
                    if actual_processed_graphs == 0 and num_graphs_in_this_batch == 0: continue
                    if actual_processed_graphs != num_graphs_in_this_batch:
                        if not (num_graphs_in_this_batch==0 and actual_processed_graphs==0):
                            if actual_processed_graphs == 0 : continue
                    
                    pred_vref_eval = v0_from_u[:actual_processed_graphs] + pred_delta_v
                    target_vref_eval = data_obj.y_true_vref.squeeze(-1)[:actual_processed_graphs]
                    loss_opt = torch.tensor(0.0,device=device,dtype=DEFAULT_TORCH_DTYPE,requires_grad=is_training)
                    if is_training and optimizer and loss_for_backward_fn:
                        target_delta_v_train = data_obj.y.squeeze(-1)[:actual_processed_graphs]
                        r_internal_loss = data_obj.R_internal_val.squeeze(-1)[:actual_processed_graphs]
                        loss_opt = loss_for_backward_fn(pred_delta_v, target_delta_v_train, r_internal_loss, v0_from_u[:actual_processed_graphs], dist_weight_cutoff)
            
            metric_val = float('nan')
            if metric_fn_for_eval and pred_vref_eval.numel()>0:
                metric_val = metric_fn_for_eval(pred_vref_eval.float(), target_vref_eval.float())
                if torch.is_tensor(metric_val): metric_val = metric_val.item()

            if is_training and optimizer and loss_opt.requires_grad and actual_processed_graphs>0:
                if not (torch.isnan(loss_opt) or torch.isinf(loss_opt)):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss_opt).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else: print(f"Warning: NaN/Inf loss at batch {batch_idx}. Skipping backprop.")
                if scheduler: scheduler.step()

            if actual_processed_graphs > 0:
                val_to_add = loss_opt.item() if is_training and not (torch.isnan(loss_opt) or torch.isinf(loss_opt)) else \
                             (metric_val if not is_training and not (math.isnan(metric_val) or math.isinf(metric_val)) else 0.0)
                num_valid_in_metric = actual_processed_graphs if (is_training and not (torch.isnan(loss_opt) or torch.isinf(loss_opt))) or \
                                                            (not is_training and not (math.isnan(metric_val) or math.isinf(metric_val))) else 0
                if num_valid_in_metric > 0:
                    total_loss_display += val_to_add * num_valid_in_metric
                    n_graphs_processed += num_valid_in_metric
        except Exception as e: print(f"Error batch {batch_idx}: {e}"); import traceback; traceback.print_exc(); continue
    return total_loss_display / n_graphs_processed if n_graphs_processed > 0 else float('nan')

def plot_regression(model,loader,device,title="Regression Plot",save_path=None):
    model.eval(); all_preds_vref,all_targets_vref=[],[]
    if loader is None or not hasattr(loader,'dataset') or len(loader.dataset)==0: print(f"No plot: {title}, loader/dataset empty."); return
    with torch.no_grad():
        for batch_idx, data_obj in enumerate(loader):
            try:
                data_obj=data_obj.to(device)
                num_graphs_in_this_batch = data_obj.num_graphs if hasattr(data_obj,'num_graphs') and data_obj.num_graphs is not None else 0
                
                current_u = data_obj.u
                if not hasattr(data_obj, 'u') or data_obj.u is None:
                    current_u = torch.empty(num_graphs_in_this_batch, SCALAR_GLOBAL_IRREPS_DIM, device=device, dtype=DEFAULT_TORCH_DTYPE)
                elif current_u.ndim == 1:
                    if num_graphs_in_this_batch == 0 and current_u.numel() == SCALAR_GLOBAL_IRREPS_DIM:
                        num_graphs_in_this_batch = 1 # Update local num_graphs for this iteration
                        # Note: data_obj.num_graphs is not updated here as it's just for local processing in plot
                        current_u = current_u.unsqueeze(0)
                    elif num_graphs_in_this_batch == 1 and current_u.numel() == SCALAR_GLOBAL_IRREPS_DIM:
                        current_u = current_u.unsqueeze(0)
                    elif num_graphs_in_this_batch > 0 and current_u.numel() == num_graphs_in_this_batch * SCALAR_GLOBAL_IRREPS_DIM:
                         current_u = current_u.reshape(num_graphs_in_this_batch, SCALAR_GLOBAL_IRREPS_DIM)
                    elif num_graphs_in_this_batch == 0 and current_u.numel() == 0 :
                        current_u = current_u.reshape(0, SCALAR_GLOBAL_IRREPS_DIM)
                    else:
                        # If it's still 1D and doesn't fit known patterns, it's problematic for plotting this batch.
                        print(f"Plotting Warning: u shape {current_u.shape} for batch_idx {batch_idx} with num_graphs {num_graphs_in_this_batch} is unexpected. Skipping this batch for plot.")
                        continue # Skip to the next data_obj in the loader
                data_obj.u = current_u # Assign reshaped u back to data_obj for this scope

                # Assertions to ensure correctness (can be removed after debugging)
                if num_graphs_in_this_batch > 0:
                    if not (data_obj.u.ndim == 2 and \
                            data_obj.u.shape[0] == num_graphs_in_this_batch and \
                            data_obj.u.shape[1] == SCALAR_GLOBAL_IRREPS_DIM):
                        print(f"Plotting Warning: data_obj.u shape validation failed after reshape. Shape: {data_obj.u.shape}, Expected: ({num_graphs_in_this_batch}, {SCALAR_GLOBAL_IRREPS_DIM}). Skipping batch.")
                        continue
                elif num_graphs_in_this_batch == 0:
                     if not (data_obj.u.ndim == 2 and data_obj.u.shape[0] == 0 and data_obj.u.shape[1] == SCALAR_GLOBAL_IRREPS_DIM):
                        print(f"Plotting Warning: data_obj.u shape validation failed for empty batch after reshape. Shape: {data_obj.u.shape}, Expected: (0, {SCALAR_GLOBAL_IRREPS_DIM}). Skipping batch.")
                        continue

                v0 = data_obj.u[:,V0_SCALAR_GLOBAL_INDEX] if data_obj.u.numel()>0 and data_obj.u.shape[0] > 0 else torch.empty(0,device=device, dtype=DEFAULT_TORCH_DTYPE)
                pred_delta_v=model(data_obj)
                actual_model_out_graphs = pred_delta_v.shape[0]

                if actual_model_out_graphs == 0 and num_graphs_in_this_batch == 0: continue
                if actual_model_out_graphs != num_graphs_in_this_batch:
                     print(f"Plotting: Model out {actual_model_out_graphs} != batch graphs {num_graphs_in_this_batch}. Skip."); continue
                
                target_vref = data_obj.y_true_vref.squeeze(-1)
                pred_vref = v0 + pred_delta_v
                if pred_vref.numel() > 0: all_preds_vref.append(pred_vref.cpu()); all_targets_vref.append(target_vref.cpu())
            except Exception as e: print(f"Error plotting batch: {e}"); continue
    if not all_preds_vref: print(f"No data for plot: {title}."); return
    preds_np = torch.cat(all_preds_vref).numpy() if all_preds_vref else np.array([])
    targets_np = torch.cat(all_targets_vref).numpy() if all_targets_vref else np.array([])

    if preds_np.size==0 or targets_np.size==0: print(f"No valid data for plot {title}"); return
    plt.figure(figsize=(8,8)); plt.scatter(targets_np,preds_np,alpha=0.5,label="Pred vs Actual Vref")
    min_v_list, max_v_list = [], []
    if targets_np.size > 0: min_v_list.append(np.nanmin(targets_np)); max_v_list.append(np.nanmax(targets_np))
    if preds_np.size > 0: min_v_list.append(np.nanmin(preds_np)); max_v_list.append(np.nanmax(preds_np))
    min_v = np.nanmin(min_v_list) if min_v_list else np.nan
    max_v = np.nanmax(max_v_list) if max_v_list else np.nan
    if not (np.isnan(min_v) or np.isnan(max_v)): plt.plot([min_v,max_v],[min_v,max_v],'r--',lw=2,label="Ideal")
    plt.xlabel("Actual Vref"); plt.ylabel("Predicted Vref"); plt.title(title); plt.legend(); plt.grid(True)
    if save_path:
        try: plt.savefig(save_path); print(f"Plot saved: {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

def main():
    default_params={'root_dir':str(DEFAULT_SPYDER_ROOT_DIR),'epochs':DEFAULT_EPOCHS,'batch_size':DEFAULT_BATCH,'physical_cutoff':DEFAULT_PHYSICAL_CUTOFF,
                    'lr':DEFAULT_LR,'num_rbf':DEFAULT_NUM_RBF,'learnable_rbf_freqs':False,'num_conv_layers':DEFAULT_NUM_CONV_LAYERS,
                    'apply_augmentations':True,'seed':42,
                    'cache_file_name': "cached_graphs.pt", 'num_workers': 2, 'persistent_workers': True, 'pin_memory': True,
                    'augment_cached_train': False, 'physical_jitter_strength': PHYSICAL_JITTER_STRENGTH, 'use_amp': True }
    ap=argparse.ArgumentParser(description="e3nn Delta-Learning Vref")
    for k,v in default_params.items():
        action = argparse.BooleanOptionalAction if isinstance(v,bool) else None
        type_ = type(v) if not isinstance(v,bool) and v is not None else str
        if type_ == Path: type_ = str
        ap.add_argument(f"--{k.replace('_','-')}",type=type_,default=v,action=action)
    
    if RUN_FROM_IDE:
        args = argparse.Namespace(**default_params)
        args.root_dir = str(default_params['root_dir'])
    else:
        args=ap.parse_args()

    internal_cutoff_val = args.physical_cutoff * INT_ANGSTROM_SCALE

    current_root_dir = Path(args.root_dir)
    if not current_root_dir.exists(): print(f"Error: Root {current_root_dir} missing."); sys.exit(1)
    plots_dir, models_dir = current_root_dir/"plots", current_root_dir/"models"
    plots_dir.mkdir(parents=True,exist_ok=True); models_dir.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Params: {vars(args)}")

    cache_fp = current_root_dir / args.cache_file_name
    if not cache_fp.exists():
        print(f"Cache file {cache_fp} not found. Building it now...")
        try:
            from preprocess_and_cache import build_cache
            build_cache(current_root_dir, args.physical_cutoff, outfile_name=args.cache_file_name, 
                        internal_angstrom_scale=INT_ANGSTROM_SCALE, default_torch_dtype=DEFAULT_TORCH_DTYPE)
            print(f"Cache built: {cache_fp}")
        except ImportError:
            print("Error: Could not import 'build_cache'. Ensure preprocess_and_cache.py is accessible.")
            sys.exit(1)
        except Exception as e:
            print(f"Error building cache: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    
    loaded_cache_metadata = {}
    if cache_fp.exists():
        try:
            cache_content = torch.load(cache_fp, weights_only=False)
            if isinstance(cache_content, dict) and 'metadata' in cache_content:
                loaded_cache_metadata = cache_content['metadata']
                cached_internal_cutoff = loaded_cache_metadata.get('internal_cutoff_val', -1.0)
                cached_scale = loaded_cache_metadata.get('INT_ANGSTROM_SCALE', -1.0)
                if not math.isclose(cached_internal_cutoff, internal_cutoff_val) or \
                   not math.isclose(cached_scale, INT_ANGSTROM_SCALE):
                    print(f"Critical Warning: Mismatch between current settings and cached metadata for {cache_fp}!")
                    print(f"  Cached: internal_cutoff={cached_internal_cutoff}, scale={cached_scale}")
                    print(f"  Current: internal_cutoff={internal_cutoff_val}, scale={INT_ANGSTROM_SCALE}")
                    print("Exiting. Please re-generate the cache with current settings or adjust settings to match the cache.")
                    sys.exit(1)
        except Exception as e:
            print(f"Could not load or verify metadata from cache file {cache_fp}: {e}")

    if args.apply_augmentations and not args.augment_cached_train:
        ds_train_full = DimerDataset(current_root_dir, internal_cutoff_val, augment=True, physical_jitter_strength=args.physical_jitter_strength)
        print("Using DimerDataset with on-the-fly augmentations for training.")
    else:
        ds_train_full = CachedDimerDataset(cache_fp, 
                                           augment_with_rotations=args.augment_cached_train and args.apply_augmentations, 
                                           physical_jitter_strength=args.physical_jitter_strength if args.augment_cached_train and args.apply_augmentations else 0.0)
        ds_train_full.metadata = loaded_cache_metadata
        print(f"Using CachedDimerDataset for training. Augment cached: {args.augment_cached_train and args.apply_augmentations}")

    ds_eval_full = CachedDimerDataset(cache_fp, augment_with_rotations=False, physical_jitter_strength=0.0)
    ds_eval_full.metadata = loaded_cache_metadata
    print("Using CachedDimerDataset for validation/test.")

    n = len(ds_eval_full)
    print(f"Total dataset size (from cache for splitting): {n}")
    current_batch_size = args.batch_size
    if current_batch_size <= 0: current_batch_size=1
    if current_batch_size > n and n > 0: current_batch_size=n
    
    all_idx = list(range(n)); np.random.shuffle(all_idx)
    n_tr_count,n_val_count,n_te_count = 0,0,0
    if n==0: train_indices,val_indices,test_indices = [],[],[]
    elif n<10:
        if n==1: n_tr_count=1
        elif n<4: n_tr_count=1; n_val_count=n-1
        else: n_val_count=1; n_te_count=1; n_tr_count=n-2
        train_indices=all_idx[:n_tr_count]; val_indices=all_idx[n_tr_count:n_tr_count+n_val_count]; test_indices=all_idx[n_tr_count+n_val_count:]
        if n>=4 and not test_indices and (n_tr_count+n_val_count < n): test_indices=all_idx[n_tr_count+n_val_count:]
    else:
        n_val_count=max(1,int(round(n*0.1))); n_te_count=max(1,int(round(n*0.1))); n_tr_count=n-n_val_count-n_te_count
        if n_tr_count<1:
            n_tr_count=1
            if n_val_count+n_te_count>=n:
                if n_val_count>=n_te_count and n_val_count>0: n_val_count=max(0,n-1-n_te_count)
                elif n_te_count>0: n_te_count=max(0,n-1-n_val_count)
            n_val_count,n_te_count = max(0,n_val_count), max(0,n_te_count)
            n_tr_count=n-n_val_count-n_te_count
        train_indices=all_idx[:n_tr_count]; val_indices=all_idx[n_tr_count:n_tr_count+n_val_count]; test_indices=all_idx[n_tr_count+n_val_count:]
    n_tr,n_val,n_te = len(train_indices),len(val_indices),len(test_indices)
    if n_tr+n_val+n_te != n:
        if n>=10:
            n_te_calc,n_val_calc=max(1,int(round(n*0.1))),max(1,int(round(n*0.1))); n_tr_calc=n-n_te_calc-n_val_calc
            if n_tr_calc<1: n_tr_calc=1; n_val_calc=max(1,(n-1)//2); n_te_calc=n-1-n_val_calc
            n_tr_count,n_val_count,n_te_count = n_tr_calc,n_val_calc,n_te_calc
        elif n>0: n_tr_count=n; n_val_count=0; n_te_count=0
        else: n_tr_count,n_val_count,n_te_count = 0,0,0
        train_indices=all_idx[:n_tr_count]; val_indices=all_idx[n_tr_count:n_tr_count+n_val_count]; test_indices=all_idx[n_tr_count+n_val_count:]
        n_tr,n_val,n_te = len(train_indices),len(val_indices),len(test_indices)
    print(f"Splitting: Train: {n_tr}, Val: {n_val}, Test: {n_te}")

    datasets = {}
    if train_indices: datasets['train'] = torch.utils.data.Subset(ds_train_full, train_indices)
    if val_indices: datasets['val'] = torch.utils.data.Subset(ds_eval_full, val_indices)
    if test_indices: datasets['test'] = torch.utils.data.Subset(ds_eval_full, test_indices)
    
    dl_args = {'num_workers': args.num_workers, 'persistent_workers': args.persistent_workers if args.num_workers > 0 else False, 'pin_memory': args.pin_memory if device.type=='cuda' else False}
    tr_loader = DataLoader(datasets['train'],batch_size=current_batch_size,shuffle=True,drop_last=(n_tr>current_batch_size),**dl_args) if 'train' in datasets and n_tr>0 else None
    va_loader = DataLoader(datasets['val'],batch_size=current_batch_size,shuffle=False,drop_last=(n_val>current_batch_size),**dl_args) if 'val' in datasets and n_val>0 else None
    te_loader = DataLoader(datasets['test'],batch_size=current_batch_size,shuffle=False,drop_last=(n_te>current_batch_size),**dl_args) if 'test' in datasets and n_te>0 else None

    model=DeltaCoupling(num_rbf=args.num_rbf,internal_rbf_cutoff=internal_cutoff_val,learnable_rbf_freqs=args.learnable_rbf_freqs,
                        num_conv_layers=args.num_conv_layers).to(device)
    
    if device.type == 'cuda' and hasattr(model, 'to') and hasattr(torch, 'channels_last'):
        model = model.to(memory_format=torch.channels_last)
    
    if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
        print("Attempting to compile model with torch.compile (PyTorch 2.0+)")
        backend_to_try = "aot_eager"
        if sys.platform != "win32" and device.type == 'cuda':
            backend_to_try = "inductor"
        try:
            model = torch.compile(model, backend=backend_to_try, mode="default") 
            print(f"Model compiled successfully with {backend_to_try} backend.")
        except Exception as e:
            print(f"torch.compile with {backend_to_try} failed: {e}. Proceeding without compilation.")
    elif hasattr(torch, 'compile') and torch.__version__ >= "2.0.0" and sys.platform == "win32":
         print("torch.compile with 'inductor' backend skipped on Windows due to potential Triton issues. Using eager mode or try 'aot_eager'.")

    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    sched = None
    if tr_loader and len(tr_loader)>0: sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=args.lr,total_steps=max(20,args.epochs*len(tr_loader)),pct_start=0.3,anneal_strategy='cos')
    best_val_mape,patience_count,EARLY_STOPPING_PATIENCE=float('inf'),0,250

    if tr_loader:
        print(f"Training: RBFs:{args.num_rbf}, Lyrs:{args.num_conv_layers}, MaxLR:{args.lr}, Aug:{args.apply_augmentations}. Loss: DistWeightL1(Vref), Eval: MAPE(Vref)")
        for epoch in range(1,args.epochs+1):
            train_loss = run_epoch(tr_loader,model,opt,sched,device,loss_for_backward_fn=distance_weighted_l1_on_vref_loss,metric_fn_for_eval=mape_loss_fn,is_training=True,dist_weight_cutoff=internal_cutoff_val, use_amp=args.use_amp)
            val_mape = run_epoch(va_loader,model,None,None,device,metric_fn_for_eval=mape_loss_fn,is_training=False,dist_weight_cutoff=internal_cutoff_val, use_amp=args.use_amp)
            if epoch==1 or epoch%10==0 or epoch==args.epochs:
                lr_val = sched.get_last_lr()[0] if sched and hasattr(sched,'get_last_lr') and sched.get_last_lr() else (sched._last_lr[0] if sched and hasattr(sched,'_last_lr') and sched._last_lr else args.lr)
                print(f"E {epoch:3d}|Train DistL1:{train_loss:8.3e}|Val MAPE:{val_mape if not math.isnan(val_mape) else 'N/A':>8.2f}%|LR:{lr_val:.2e}")
            if va_loader and not math.isnan(val_mape):
                if val_mape<best_val_mape:
                    best_val_mape=val_mape; torch.save(model.state_dict(),models_dir/"best_model_checkpoint.pt"); patience_count=0
                    print(f" ->Best val MAPE:{best_val_mape:.2f}%. Saved.")
                else: patience_count+=1
                if patience_count>=EARLY_STOPPING_PATIENCE: print(f"Early stop @ ep {epoch}."); break
    else: print("Skipping training: no training data.")
    
    best_model_path = models_dir/"best_model_checkpoint.pt"
    if best_model_path.exists(): print(f"Loading best: {best_model_path}"); model.load_state_dict(torch.load(best_model_path,map_location=device))
    else: print("No best model checkpoint. Using last state (if any).")

    if te_loader:
        test_mape=run_epoch(te_loader,model,None,None,device,metric_fn_for_eval=mape_loss_fn,is_training=False,dist_weight_cutoff=internal_cutoff_val, use_amp=args.use_amp)
        print(f"\nTest MAPE (Vref):{test_mape if not math.isnan(test_mape) else 'N/A':8.2f}%")
        if not math.isnan(test_mape): plot_regression(model,te_loader,device,title=f"Test Set Vref Regression (MAPE:{test_mape:.2f}%)",save_path=plots_dir/"test_final_delta_learning.png")
    else: print("\nNo test data.")
    if va_loader:
        val_mape_final=run_epoch(va_loader,model,None,None,device,metric_fn_for_eval=mape_loss_fn,is_training=False,dist_weight_cutoff=internal_cutoff_val, use_amp=args.use_amp)
        print(f"Final Val MAPE (Vref) with best model:{val_mape_final if not math.isnan(val_mape_final) else 'N/A':.2f}%")
        if not math.isnan(val_mape_final): plot_regression(model,va_loader,device,title=f"Val Set Vref Regression (MAPE:{val_mape_final:.2f}%)",save_path=plots_dir/"val_final_delta_learning.png")
    else: print("\nNo validation data for final model.")
    
    torch.save(model.state_dict(),models_dir/"final_model_delta_learning.pt")
    print(f"Final model saved → {models_dir/'final_model_delta_learning.pt'}")

if __name__ == "__main__":
    main()
```


# Introduction

Modern machine-learning pipelines often juggle vast datasets, complex transformations, and intricate neural-network architectures. To understand and manage such complexity, we rely on **computational thinking**—a mindset that emphasizes abstraction, decomposition, pattern recognition, and algorithmic design. In this blog, we’ll dissect a complete script that builds, trains, and evaluates an equivariant graph-neural-network (GNN) model for predicting reference energies ($V_0 \rightarrow V_{\text{ref}}$ deltas) of paired molecules (dimers). Rather than drowning in mathematical minutiae, we’ll focus on how the code is structured so that each piece makes logical sense in a larger workflow. Along the way, we’ll see:

- **Abstraction**: How raw PDB files, NPZ metadata, and atomic physics get distilled into tidy tensors and irreducible representations (irreps).
- **Decomposition**: How the author splits tasks into small, testable functions and classes—everything from “read a PDB file” to “apply a single convolutional layer.”
- **Pattern Recognition**: How repeated patterns (distance-based edge construction, spherical harmonics, radial basis expansion) unify disparate tasks under the same interface.
- **Algorithmic Design**: How message-passing, data augmentation, and training loops come together to turn graphs of atoms into numerical predictions of $V_{\text{ref}}$.

By the end, you’ll see this script as not a hairball but a layered, logical sequence of computational steps—each solving a small subproblem that contributes to the final goal.

---

## 1. High-Level Overview: From Files to Predictions

Before we dive into details, let’s outline the big picture:

- Parse command-line arguments (or use defaults if running from an IDE). These include paths, hyperparameters (like learning rate, number of radial basis functions, etc.), and flags for data augmentation.
- Build or load a cached dataset of “dimer graphs.” Each graph comes from:
  - A pair of PDB files (molecule A and molecule B → atomic numbers and 3D coordinates).
  - One NPZ file containing “Mulliken populations,” dipole/quadrupole information, and a reference energy $V_{\text{ref}}$.
  - Scalar global features (distance $R$, orientation $\theta$, initial energy $V_0$, participation ratio PR).

From these pieces, the code constructs a PyTorch-Geometric `Data` object representing one dimer (atoms as nodes, edges based on interatomic distances, node attributes, edge attributes, etc.).

- **Augment data on the fly (optional)**: apply small random jitters to atom coordinates, random rotations of the entire dimer, and the corresponding updates of vector/tensor features (dipoles, quadrupoles) to preserve physical equivariance.

- Define an **equivariant convolutional architecture** using the E3NN library (which enforces rotational equivariance in all operations). The core building block is a “points convolution” (`PointsConvolutionIntegrated`) that:
  - Expands interatomic distances into a radial basis (Bessel functions).
  - Computes spherical harmonics on the relative position vectors.
  - Performs a tensor-product operation to mix node features, edge harmonics, and learned radial weights—yielding new node features that respect rotational symmetry.
  - Repeats this for a specified number of message-passing layers (with residual connections).

- Finally, scatters node features back to a graph-level embedding, appends global scalar features ($V_0$, $R$, $\theta$, PR), and applies a small MLP to predict the delta $V_{\text{ref}}$ (error relative to $V_0$).

---

## Train/Evaluate Loop:

- Split the dataset into train/validation/test.
- For each epoch:
  - Iterate over batches,
  - Run a forward pass,
  - Compute a distance-weighted L1 loss on $\Delta V$,
  - Track MAPE (mean absolute percent error) of the final predicted $V_{\text{ref}}$.
- Use automatic mixed precision (AMP) on CUDA if available.
- Save the best-performing checkpoint based on validation MAPE,
- Finally, plot predicted vs. actual $V_{\text{ref}}$ on held-out data.

Throughout these steps, the author uses PyTorch, PyTorch-Geometric, and E3NN primitives to encapsulate low-level tensor manipulations (linear algebra, spherical harmonics, etc.) so that high-level logic remains clear.

---

## 2. Abstraction: Turning Files and Physics into Tensors

### 2.1 Atomic-Number Lookup (Periodic Table, PT)

At the very top, we see:

```python
PT: dict[str,int] = {}
for Z, sym in enumerate(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca".split(), 1):
    PT[sym] = Z
```

**Abstraction:** Instead of scattering atomic numbers and symbol lookups all over the code, we build a single dictionary `PT` that maps `"C"` → 6, `"O"` → 8, etc. Any function needing atomic numbers can just do `PT[elem_symbol]`. This simple data structure hides away lookup details so that higher-level routines don’t care how we convert an `elem_symbol` to an integer.

### 2.2 Quaternion-Based Random Rotation Matrix (`random_rotation_matrix`)

```python
@torch.jit.script
def random_rotation_matrix(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    q = torch.randn(4, device=device, dtype=dtype)
    # Normalize quaternion and unpack into w,x,y,z
    ...
    # Build 3×3 rotation matrix from quaternion
    ...
    # If determinant < 0, flip column 2 to ensure a proper rotation
    ...
    return R_candidate
```

**Abstraction:** Generating a random SO(3) rotation is a common need in data augmentation. Rather than writing those steps inline each time, we encapsulate them in a single JIT-compiled function that returns a 3×3 orthogonal matrix. Now “rotate my coordinates” just means `pos @ R.T`. Internally, it uses quaternions to guarantee uniform sampling on the rotation group.

### 2.3 Tensor-Equivariant Feature Transformation (`transform_by_matrix`)

```python
@torch.no_grad()
def transform_by_matrix(
    irreps: o3.Irreps,
    feats: torch.Tensor,
    rotation: torch.Tensor,
    *, check: bool = False
) -> torch.Tensor:
    # 1) Flatten feats to shape (-1, irreps.dim) if necessary
    # 2) Compute the D⁽ᵢʳʳᵉᵖˢ⁾(rotation) matrix (size irreps.dim × irreps.dim)
    # 3) Perform feats ← feats @ Dᵀ
    # 4) Reshape back to original
```

**Abstraction:** Vectors, dipoles, quadrupoles, and other higher-order features transform under rotations in known ways (irreducible representations of SO(3)). This helper function says: “Given `irreps = 1×1o` (vector) or `1×2e` (quadrupole), rotate all of `feats` by `rotation` in a physically correct manner.” We hide away the details of building the D-matrix from the irreps, so that any time we do “random rotation” of a node’s dipole and quadrupole, we just call this function instead of manually re-implementing spherical harmonics or Wigner D-matrices.

### 2.4 Quadrupole Decomposition (`decompose_quadrupole_to_real_spherical`)
**Abstraction:** A 3×3 quadrupole tensor has 6 independent components, but to feed it into an equivariant model, we need it in the 5-dimensional space of traceless second-order spherical harmonics. This function does exactly that decomposition. Once again, higher-level code (in the dataset constructor) never worries about “how do I subtract the trace and form the real spherical harmonics?”—it just calls `decompose_quadrupole_to_real_spherical`.

```python
def decompose_quadrupole_to_real_spherical(q_vec: torch.Tensor) -> torch.Tensor:
    # Input: Qxx, Qxy, Qxz, Qyy, Qyz, Qzz (shape (..., 6))
    # Output: 5 real spherical components c0,...,c4 (shape (..., 5))
```


## 3. Decomposition: Breaking the Workflow into Small Pieces

### 3. Decomposition: Breaking the Workflow into Small Pieces

A “giant monolithic script” is almost impossible to read or maintain. Here, the author has decomposed the workflow into self-contained functions and classes. Let’s see how:

#### 3.1 File Parsing & Data Extraction

**`read_pdb(path: Path) → (torch.Tensor(elems), torch.Tensor(xyz))`**  
Reads a PDB file line by line, extracts element symbols and atomic coordinates. Returns:

- `elems`: a 1D tensor of integer atomic numbers.  
- `xyz`: an Nx3 tensor of physical coordinates (in ångströms).

**Why separate?**  
This function solves “How do I get atomic numbers & positions from a PDB?” once and for all. Upstream code can then trust that it has clean, verified tensors.

---

**`RadialBesselBasisLayer(torch.nn.Module)`**  
Encodes “how to turn a scalar distance into a radial-basis vector of length `num_rbf`.” This class:

- Builds the fixed (or learnable) frequency array for Bessel functions.  
- Provides a `_cosine_cutoff` so that distances beyond the cutoff smoothly go to zero.  
- In `forward(distances)`, computes `sin(ω r) / (ω r)` for each frequency `ω`, multiplies by the cutoff.  
- Returns a `(num_distances, num_rbf)` tensor.

**Why separate?**  
Any time we need to embed distances into radial features—whether edges between atoms or edges to a virtual “field node”—we call this module.

---

#### 3.2 Dataset & Caching

**`class DimerDataset(Dataset)`**  
**Purpose:** “Given a root directory, how do I build PyG Data objects for each dimer?”

**Key Steps (in `__getitem__`):**

- Read the meta table (`meta.xlsx`) with pandas.  
- For each row, get IDs of molecule A and B, construct file names, call `read_pdb` on both.

**Load NPZ:**

- Extract Mulliken populations (`s, p, d`) → node attributes (3 channels).  
- Extract dipole (3-vector) and quadrupole (6-vector) → transform to internal units and call `decompose_quadrupole_to_real_spherical`.  
- Form `field_node_features_unrotated` by concatenating dipole + 5 real spherical quadrupole values.  
- Rotate node coordinates + dipole/quadrupole (if `augment=True`).  
- Build atomic graph edges: `radius_graph` (PyG) on internal coordinates.  
    - `edge_index_atoms`: pairs of atom indices whose internal distance ≤ cutoff.  
    - `edge_attr_distances_atoms`: the corresponding distance values (shape `(num_edges, 1)`).

- Gather global scalars (`R, θ, V₀, PR`) into `scalar_global_features` and target `ΔV = V_ref – V₀`.

- Package everything into a `Data(...)` dictionary.

  ```python
  Data(
  z_atoms=Z_atoms, 
  pos_atoms=pos_to_augment,
  atomic_node_attr=node_attr_tensor,
  edge_index_atoms=edge_index_atoms, 
  edge_attr_atoms=edge_attr_distances_atoms,
  field_node_features=final_field_node_features,
  u=scalar_global_features,
  y=delta_v_target, 
  y_true_vref=y_true_vref_for_loss_and_plot,
  R_internal_val=torch.tensor([R_internal_shift_val]),
  ...
)
```
Return that Data object.

**Why decompose?** If we ever want to build a slightly different dataset—say, omit quadrupoles or take a different cutoff—we only modify `DimerDataset`. Everything else (model, training loop) remains unchanged.

---

### `class CachedDimerDataset(Dataset)`

**Purpose:** Load pre-computed graphs from a cache file (`cached_graphs.pt`) so that we don’t have to re-parse PDB/NPZ every run.

- **Constructor:** simply `torch.load(cache_file) → self.graphs`.
- In `__getitem__`, it clones an existing `Data` object, and optionally applies jitter/rotation (if augmentation is enabled). 
- Crucially, it **re-computes edges** or rotates existing **spherical harmonics** so that node positions and features stay consistent.

**Why decompose?** Building the dataset can be extremely expensive. By splitting it into:

> “build me a list of raw `Data` objects once, save them,”  
> and  
> “later, load cached ones (optionally augment on the fly),”  

we decouple “I/O & raw feature extraction” from “model training.”

---

### 3.3 Equivariant Convolutional Block: `PointsConvolutionIntegrated`

```python
@compile_mode("script")
class PointsConvolutionIntegrated(torch.nn.Module):
    def __init__(self, 
                 irreps_node_input, irreps_node_attr, irreps_edge_sh, irreps_node_output,
                 fc_hidden_dims, num_rbf, rbf_cutoff, learnable_rbf_freqs):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr  = o3.Irreps(irreps_node_attr)
        self.irreps_edge_sh    = o3.Irreps(irreps_edge_sh)
        self.irreps_node_output= o3.Irreps(irreps_node_output)

        # 1) Build radial basis layer
        self.rbf_basis = RadialBesselBasisLayer(num_rbf, rbf_cutoff, learnable_rbf_freqs)

        # 2) FullyConnectedTensorProducts for “self” and “linear” paths
        self.sc   = FullyConnectedTensorProduct(self.irreps_node_input, 
                                                self.irreps_node_attr, 
                                                self.irreps_node_output)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, 
                                                self.irreps_node_attr, 
                                                self.irreps_node_input)

        # 3) Construct the tensor-product unit (`self.tp`) that mixes node features and edge harmonics
        instructions_for_tp = []
        tp_output_channels_list_for_tp = []
        for i_in1, (mul_1, ir_1) in enumerate(self.irreps_node_input):
            for i_in2, (mul_2, ir_2) in enumerate(self.irreps_edge_sh):
                for ir_out_candidate in ir_1 * ir_2:
                    if ir_out_candidate in self.irreps_node_output or ir_out_candidate.l == 0:
                        i_out = len(tp_output_channels_list_for_tp)
                        tp_output_channels_list_for_tp.append((mul_1, ir_out_candidate))
                        instructions_for_tp.append((i_in1, i_in2, i_out, "uvu", True))
        irreps_tp_direct_output = o3.Irreps(tp_output_channels_list_for_tp)
        self.tp = o3.TensorProduct(
            self.irreps_node_input, 
            self.irreps_edge_sh, 
            irreps_tp_direct_output, 
            instructions_for_tp, 
            internal_weights=False, 
            shared_weights=False
        )

        # 4) A small MLP (FullyConnectedNet) to convert RBF → inner weights for TP
        fc_neurons_full_list = [num_rbf] + fc_hidden_dims + [self.tp.weight_numel]
        self.fc = FullyConnectedNet(fc_neurons_full_list, F.silu)

        # 5) A final tensor product (lin2) to go from TP output → node_output irreps
        self.lin2 = FullyConnectedTensorProduct(self.tp.irreps_out.simplify(), 
                                                self.irreps_node_attr, 
                                                self.irreps_node_output)
        # 6) A gating scalar, α, produced by another FCTP to modulate post-convolution features
        self.alpha = FullyConnectedTensorProduct(self.tp.irreps_out.simplify(), 
                                                 self.irreps_node_attr, 
                                                 "0e")
        with torch.no_grad(): self.alpha.weight.zero_()  # Start gate closed

    def forward(self, node_input, node_attr, edge_sh_attr,
                      edge_scalar_distances, batch_info_for_scatter) -> torch.Tensor:
        """
        node_input: (N, dim_node_input)   [irreps_node_input basis]
        node_attr:  (N, dim_node_attr)     [irreps_node_attr basis]
        edge_sh_attr: (E, dim_edge_sh)     [spherical harmonics on each edge]
        edge_scalar_distances: (E,1)       [physical distances]
        batch_info_for_scatter: dict with ‘edge_src’ and ‘edge_dst’ indices for message passing
        """
        edge_src = batch_info_for_scatter['edge_src']
        edge_dst = batch_info_for_scatter['edge_dst']

        # 1) Expand scalar distances → radial basis (shape: E × num_rbf)
        expanded_edge_scalars = self.rbf_basis(edge_scalar_distances)

        # 2) Turn radial basis → “weight” vector of length (#tensor-product weights)
        weight = self.fc(expanded_edge_scalars)  # shape: (E, tp.weight_numel)

        # 3) Node “self-connection” (like bias): node_input ⊗ node_attr → node_output
        node_self_connection = self.sc(node_input, node_attr)

        # 4) A “linear” transformation: node_input ⊗ node_attr → node_input (for messages)
        node_features_after_lin1 = self.lin1(node_input, node_attr)

        # 5) Gather sender node features on each edge (shape: E × dim_node_input)
        if edge_src.numel() > 0:
            gathered_node_features = node_features_after_lin1[edge_src]
            # 6) Perform the tensor product: (node ⊗ edge_sh) → “message” (E × dim_tp_out)
            edge_message_features = self.tp(gathered_node_features, edge_sh_attr, weight)
            # 7) Aggregate messages at each destination node: scatter_mean over edge_dst
            aggregated_node_features = scatter_mean(edge_message_features, edge_dst, dim=0, dim_size=node_input.shape[0])
        else:
            # No edges → zero out aggregated part
            aggregated_node_features = torch.zeros((node_input.shape[0], self.tp.irreps_out.dim),
                                                   device=node_input.device, dtype=node_input.dtype)

        # 8) A second tensor product to combine aggregated messages + node_attr → candidate node_output
        node_conv_out_before_alpha = self.lin2(aggregated_node_features, node_attr)
        # 9) Gate each irreducible channel by α (learned scalar between -1 and 1)
        alpha_scalars = torch.tanh(self.alpha(aggregated_node_features, node_attr))
        m = self.sc.output_mask
        alpha_gate = (1 - m) + alpha_scalars * m  # only gate non-scalar outputs
        return node_self_connection + alpha_gate * node_conv_out_before_alpha
```
Decomposition: Equivariant Convolution and Model Definition

### PointsConvolutionIntegrated

This single class encapsulates all the computations needed for a single message-passing step. Internally, it breaks the job into these sub-tasks:

- **Expand distances (1D) → RBF features (kD)**  
- **MLP:** RBF → “weights” for each basis in the tensor-product.

- **Compute two tensor-products on node features:**
  - A “self” path (`sc`) that directly maps `node_input → node_output` (skipped if no edges).
  - A “linear” path (`lin1`) that extracts each node’s feature contributions to messages.

- **For each edge:**
  - Gather the sender’s “linear” features
  - Apply `tp(gathered_features, spherical_harmonics, weight)` → a message vector.

- **Scatter/aggregate edge messages (by `edge_dst`).**

- **Another tensor product (`lin2`) to combine aggregated messages + `node_attr` → a candidate output.**

- **A learned gate:**
  - Apply `self.alpha(...)`, pass through `tanh`
  - Mask only irreducible-tensor channels (i.e., we don’t gate the scalar part).

- **Sum the “self” connection + gated convolution output → new node features** (in `irreps_node_output` format).

**Why decompose?**  
By decomposing it this way, each transformation remains logically separate. If you wanted to swap out RBF for Gaussian radial expansion, or replace gating with simple addition, you would only modify the relevant small block.

---

### 3.4 Full Model: DeltaCoupling

```python
class DeltaCoupling(nn.Module):
    SCALAR_GLOBAL_IRREPS_STR_CLS = SCALAR_GLOBAL_IRREPS_STR
    FIELD_NODE_IRREPS_STR_CLS   = "1x1o + 1x2e"

    def __init__(self, max_Z: int, num_rbf: int, internal_rbf_cutoff: float, 
                 learnable_rbf_freqs: bool, num_conv_layers: int):
        super().__init__()
        self.num_conv_layers = num_conv_layers

        # 1) Node feature irreps: 8 scalars, plus vector (1 o), plus quadrupole (2 e)
        self.node_features_irreps = o3.Irreps("8x0e + 1x1o + 1x2e")
        # 1a) We embed atomic number → 8×0e (eight scalars) via an Embedding
        self.atomic_scalar_embed_dim = o3.Irreps("8x0e").dim
        self.embed_atomic_scalar = nn.Embedding(max_Z + 1, self.atomic_scalar_embed_dim)

        # 2) Field node irreps: 1 vector (1×1o) + 1 quadrupole (1×2e)
        self.field_node_irreps = o3.Irreps(DeltaCoupling.FIELD_NODE_IRREPS_STR_CLS)
        self.field_node_dim    = self.field_node_irreps.dim

        # 3) Node attribute embedding: from (3×0e) → (16×0e) via an MLP
        self.input_node_attr_irreps    = o3.Irreps("3x0e")
        self.node_attr_embedding_dim   = 16
        self.embedded_node_attr_irreps = o3.Irreps(f"{self.node_attr_embedding_dim}x0e")
        self.node_attr_mlp = FullyConnectedNet(
            [self.input_node_attr_irreps.dim, 32, self.embedded_node_attr_irreps.dim],
            F.silu
        )

        # 4) Scalar global features irreps (R, θ, V₀, PR) → 4×0e
        self.scalar_global_irreps = o3.Irreps(DeltaCoupling.SCALAR_GLOBAL_IRREPS_STR_CLS)

        # 5) Build a list of `PointsConvolutionIntegrated` layers
        self.irreps_edge_sh_for_conv = IRREPS_EDGE_SH_PRECOMPUTE
        fc_hidden_dims = [32, 32]
        self.convs = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            conv = PointsConvolutionIntegrated(
                irreps_node_input = self.node_features_irreps,
                irreps_node_attr  = self.embedded_node_attr_irreps,
                irreps_edge_sh    = self.irreps_edge_sh_for_conv,
                irreps_node_output= self.node_features_irreps,
                fc_hidden_dims    = fc_hidden_dims,
                num_rbf           = num_rbf,
                rbf_cutoff        = internal_rbf_cutoff,
                learnable_rbf_freqs = learnable_rbf_freqs
            )
            self.convs.append(conv)

        # 6) Final MLP: (node_feature_dim + scalar_global_dim) → 1
        mlp_input_dim = self.node_features_irreps.dim + self.scalar_global_irreps.dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def _build_atom_field_edges(self, ptr, pos_atoms, pos_field_nodes, device, dtype):
        """
        Build edges *between* atoms and the “single field node” per graph:
          - Each atom in a graph connects to that graph’s field node (both directions).
          - Compute spherical harmonics & distances on those “atom↔field” edges.
        Returns: (edge_index_af, edge_attr_sh_af, edge_attr_dist_af)
        """
        num_graphs = len(ptr) - 1
        if num_graphs == 0 or pos_atoms.numel() == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, self.irreps_edge_sh_for_conv.dim), dtype=dtype, device=device),
                torch.empty((0, 1), dtype=dtype, device=device)
            )

        # 1) Figure out how many atoms per graph (ptr = cumulative sum of nodes per graph).
        atoms_per_graph = ptr[1:] - ptr[:-1]
        # 2) Expand graph index for each atom → “which graph this atom belongs to”
        graph_indices_for_atoms = torch.repeat_interleave(torch.arange(num_graphs, device=device), atoms_per_graph)

        # 3) Global atom indices: 0 .. (total_atoms - 1)
        atom_indices_global = torch.arange(ptr[-1], device=device)

        # 4) Field node indices: each graph has 1 field node, whose global index is total_atoms + graph_index
        field_node_indices_global_repeated = ptr[-1] + graph_indices_for_atoms

        # 5) Build source/dest arrays for both directions (atom→field, field→atom)
        src_af = atom_indices_global
        dst_af = field_node_indices_global_repeated
        src_fa = field_node_indices_global_repeated
        dst_fa = atom_indices_global
        src_combined = torch.cat([src_af, src_fa])
        dst_combined = torch.cat([dst_af, dst_fa])

        # 6) Compute relative positions for each “atom↔field” pair
        rel_pos_af = pos_field_nodes[graph_indices_for_atoms] - pos_atoms[atom_indices_global]
        rel_pos_full = torch.cat([rel_pos_af, -rel_pos_af])  # both directions

        # 7) Spherical harmonics on those relative positions
        sh = o3.spherical_harmonics(
            self.irreps_edge_sh_for_conv, rel_pos_full, normalize=True, normalization='component'
        )
        dist = rel_pos_full.norm(dim=1, keepdim=True)

        return torch.stack([src_combined, dst_combined]), sh, dist

    def forward(self, data: Batch) -> torch.Tensor:
        Z_atoms                = data.z_atoms          # (total_atoms_in_batch,)
        pos_atoms              = data.pos_atoms        # (total_atoms_in_batch, 3)
        atomic_node_attr_input = data.atomic_node_attr # (total_atoms_in_batch, 3)  – raw Mulliken s/p/d 
        scalar_globals_u_in    = data.u               # (num_graphs_in_batch, 4)
        field_node_features_in = data.field_node_features  # (num_graphs_in_batch, 1+5)
        num_graphs             = data.num_graphs
        device                 = pos_atoms.device

        # ——— 1) Process node features ———
        # Embed atomic number → “8×0e” scalars
        x_atomic_scalar_embedded = self.embed_atomic_scalar(Z_atoms)   # (atoms, 8)
        # Initialize a full node feature vector (atoms + field nodes) of shape (#nodes, dim_total)
        x_atoms = torch.zeros(
            Z_atoms.shape[0], self.node_features_irreps.dim, 
            device=device, dtype=DEFAULT_TORCH_DTYPE
        )
        # Fill in “atomic scalar” slots within x_atoms
        x_atoms[:, self.slice_for_atomic_scalar] = x_atomic_scalar_embedded

        # 2) Process “field nodes” (one per graph)
        x_field_nodes = torch.zeros(
            num_graphs, self.node_features_irreps.dim, 
            device=device, dtype=DEFAULT_TORCH_DTYPE
        )
        if num_graphs > 0:
            dip_part_dim = o3.Irreps("1x1o").dim  # size = 3
            x_field_nodes[:, self.slice_for_field_1o] = field_node_features_in[:, :dip_part_dim]
            x_field_nodes[:, self.slice_for_field_2e] = field_node_features_in[:, dip_part_dim:]

        # 3) Concatenate “all nodes” (atoms first, then field nodes)
        x_combined = torch.cat([x_atoms, x_field_nodes], dim=0)

        # 4) Embed atomic node attributes (3 scalars) → (16 scalars) for each atom
        embedded_attr_atoms = self.node_attr_mlp(atomic_node_attr_input)  # (atoms, 16)
        embedded_attr_field = torch.zeros(
            num_graphs, self.embedded_node_attr_irreps.dim, 
            device=device, dtype=DEFAULT_TORCH_DTYPE
        )
        node_attr_combined = torch.cat([embedded_attr_atoms, embedded_attr_field], dim=0)

        # 5) Determine “ptr” array for scatter_mean (how many atoms belong to each graph)
        #    If data.ptr doesn’t exist or has num_atoms=0, build a trivial one.
        ptr = data.ptr if hasattr(data, 'ptr') and data.ptr is not None else torch.tensor([0, Z_atoms.shape[0]], device=device, dtype=torch.long)
        if Z_atoms.numel() == 0 and num_graphs > 0:
            ptr = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)

        # 6) Build “atom ↔ field” edges on the fly
        if Z_atoms.numel() > 0:
            pos_field_nodes = scatter_mean(pos_atoms, data.batch, dim=0, dim_size=num_graphs)
        else:
            pos_field_nodes = torch.zeros(num_graphs, 3, device=device, dtype=DEFAULT_TORCH_DTYPE)

        pos_combined = torch.cat([pos_atoms, pos_field_nodes], dim=0)

        # 7) Determine batch indices for “all nodes”
        batch_atoms_val   = data.batch if hasattr(data, 'batch') else torch.zeros(Z_atoms.shape[0], dtype=torch.long, device=device)
        batch_field_nodes = torch.arange(num_graphs, device=device, dtype=torch.long)
        batch_combined    = torch.cat([batch_atoms_val, batch_field_nodes], dim=0)

        # 8) Edge construction for atoms
        edge_index_atoms = data.edge_index_atoms
        if hasattr(data, 'edge_attr_sh') and data.edge_attr_sh is not None:
            edge_attr_sh_atoms = data.edge_attr_sh
        elif edge_index_atoms.numel() > 0:
            rel_pos_aa = pos_atoms[edge_index_atoms[1]] - pos_atoms[edge_index_atoms[0]]
            edge_attr_sh_atoms = o3.spherical_harmonics(
                self.irreps_edge_sh_for_conv, rel_pos_aa, normalize=True, normalization='component'
            )
        else:
            edge_attr_sh_atoms = torch.empty(
                (0, self.irreps_edge_sh_for_conv.dim), device=device, dtype=DEFAULT_TORCH_DTYPE
            )

        if hasattr(data, 'edge_attr_atoms') and data.edge_attr_atoms is not None:
            edge_attr_distances_atoms_val = data.edge_attr_atoms
        elif edge_index_atoms.numel() > 0:
            edge_attr_distances_atoms_val = (pos_atoms[edge_index_atoms[1]] - pos_atoms[edge_index_atoms[0]]).norm(dim=1, keepdim=True)
        else:
            edge_attr_distances_atoms_val = torch.empty((0, 1), device=device, dtype=DEFAULT_TORCH_DTYPE)

        # 9) Build the “atom↔field” edges
        edge_index_af, sh_af, dist_af = self._build_atom_field_edges(
            ptr, pos_atoms, pos_field_nodes, device=device, dtype=DEFAULT_TORCH_DTYPE
        )

        # 10) Concatenate “atomic edges” + “atom↔field” edges
        if edge_index_atoms.numel() > 0:
            edge_index_combined     = torch.cat([edge_index_atoms, edge_index_af], dim=1)
            edge_attr_sh_combined   = torch.cat([edge_attr_sh_atoms, sh_af], dim=0)
            edge_attr_dist_combined = torch.cat([edge_attr_distances_atoms_val, dist_af], dim=0)
        else:
            edge_index_combined     = edge_index_af
            edge_attr_sh_combined   = sh_af
            edge_attr_dist_combined = dist_af

        # ——— 11) Message-Passing Layers ———
        x_conv_out = x_combined
        for i_conv in range(self.num_conv_layers):
            residual_input = x_conv_out
            batch_info_conv = {'edge_src': edge_index_combined[0], 'edge_dst': edge_index_combined[1]}
            x_conv_out = self.convs[i_conv](
                x_conv_out, node_attr_combined, edge_attr_sh_combined,
                edge_attr_dist_combined, batch_info_conv
            )
            # Every two layers, add a residual connection
            if (i_conv + 1) % 2 == 0 and x_conv_out.shape == residual_input.shape:
                x_conv_out = x_conv_out + residual_input

        # 12) Scatter node features → one vector per graph
        x_scattered = scatter_mean(x_conv_out, batch_combined, dim=0, dim_size=num_graphs)

        # 13) Concatenate with scalar global features → final MLP input
        if num_graphs > 0:
            h = torch.cat([x_scattered, scalar_globals_u_in], dim=1)
        else:
            h = torch.empty((0, self.node_features_irreps.dim + self.scalar_global_irreps.dim),
                            device=device, dtype=DEFAULT_TORCH_DTYPE)

        # 14) Final MLP → ΔV predictions
        delta_v_pred = self.mlp(h)  # shape: (num_graphs, 1)
        return delta_v_pred.squeeze(-1)
```

The `DeltaCoupling` model assembles all the building blocks:

#### **Input Embeddings:**
- Atomic number → scalar embedding in `8×0e`.
- Atomic Mulliken populations (3 scalars) → `16×0e` via an MLP.
- Dipole/quadrupole (vector+tensor) → “field node features” in `1×1o + 1×2e`.
- Global scalars (`R`, `θ`, `V₀`, `PR`) → `4×0e`.

#### **Graph Construction & Edges:**
- Input data already provides `edge_index_atoms` + `$edge_attr_atoms$`. If missing, they are recomputed via `radius_graph`.
- Spherical harmonics on edges (for direction information) are either cached or recomputed.
- “Atom ↔ field” edges unify each atom with its graph’s field node.

#### **Message-Passing Layers:**
- Multiple stacked instances of `PointsConvolutionIntegrated`, with interleaved residual connections.

#### **Pooling & MLP:**
- After passing messages, scatter all node features back to a graph vector, append global scalars, and feed into a small MLP to predict ΔV.

Because each piece is its own small function/class, the `forward` method remains a relatively concise sequence of high-level steps: `embed → build edges → run conv layers → scatter → MLP`.

---

### 4. Pattern Recognition: Identifying Repeated Structures

#### **4.1 Spherical Harmonics & Distance Embeddings**
Throughout the model, we repeatedly need:
- Relative positions between two points → spherical harmonics (`o3.spherical_harmonics`) to encode angular orientation.
- Scalar distances (norms) → radial basis expansions (`RadialBesselBasisLayer`).

Recognizing this pattern, the code:
- Wraps “build spherical harmonics on edge vectors” in a single call to `o3.spherical_harmonics`.
- Wraps “convert distances → RBF features” in `RadialBesselBasisLayer`.

Because both “atomic edges” and “atom↔field edges” need these same transformations, the code simply calls these abstractions in both places. This reuse reduces duplication and ensures consistency.

#### **4.2 Irreducible Representations & Tensor Products**
Whenever node features or edge features carry directional/rotational information, the script uses E3NN’s **irreps** (e.g., `"1x1o + 1x2e"`) and `FullyConnectedTensorProduct` or `TensorProduct` to combine them.

By consistently representing all directional features as irreps, the code ensures:
- **Equivariance**: rotating inputs produces exactly rotated outputs, not an arbitrary mixture.
- **Uniformity**: whether we handle dipoles, quadrupoles, or spherical harmonics, they’re always D-matrices acting on irreps.

This pattern appears in:
- `transform_by_matrix` (to rotate dipole/quadrupole).
- `PointsConvolutionIntegrated` (to mix node features with edge spherical harmonics).
- `DeltaCoupling._build_atom_field_edges` (to produce `edge_attr_sh_af`).

Recognizing “any time we have a vector/tensor feature, treat it as an irreducible representation and use `o3.spherical_harmonics`/`TensorProduct`” dramatically simplifies the flow.

---

### 5. Algorithmic Design: From Data to Loss

#### **5.1 Custom Loss Functions**
Two specialized losses appear:
- **Mean Absolute Percent Error** (`mape_loss_fn`)

```python
def mape_loss_fn(pred_vref, target_vref, epsilon=1e-7):
    abs_error   = torch.abs(target_vref - pred_vref)
    abs_target  = torch.abs(target_vref)
    return torch.mean( abs_error / (abs_target + epsilon) ) * 100.0
```

Algorithmic Detail: Avoid division-by-zero by adding a tiny ε. Returns percentage error, averaged across graphs.

Distance-Weighted L1 Loss on V_ref

```python
def distance_weighted_l1_on_vref_loss(pred_delta_v, target_delta_v, R_internal_batch, V0_batch, graph_cutoff_for_weighting):
    # Expand shapes to (batch, 1) if necessary
    pred_vref   = V0_batch + pred_delta_v
    target_vref = V0_batch + target_delta_v
    abs_error_vref = torch.abs(target_vref - pred_vref)
    weights = BOOST_BASE + (R_internal_batch / graph_cutoff_for_weighting) ** BOOST_EXP
    return torch.mean( weights * abs_error_vref )
```

Algorithmic Insight: Larger R (intermolecular distance) → apply a higher weight to the L1 error on V_ref. This implements the domain knowledge that small changes in V_ref at large distances should be penalized more heavily.

By factoring these as separate functions, the training loop can accept any loss_for_backward_fn or metric_fn_for_eval, making the code flexible to try out different error measures without rewriting the loop.

#### **5.2 The Training/Evaluation Loop (run_epoch)**

```python
def run_epoch(loader, model, optimizer=None, scheduler=None, device="cpu",
               loss_for_backward_fn=None, metric_fn_for_eval=mape_loss_fn,
               is_training: bool = False, dist_weight_cutoff: float = ..., use_amp: bool = True):

    model.train(is_training)
    total_loss_display, n_graphs_processed = 0.0, 0
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))

    for batch_idx, data_obj in enumerate(loader):
        data_obj = data_obj.to(device)
        num_graphs_in_batch = data_obj.num_graphs

        # 1) Ensure `data_obj.u` (scalar global features) is reshaped to (batch_size, 4)
        ...  # several if/else cases to reshape 1D → 2D if needed

        v0_from_u = data_obj.u[:, V0_SCALAR_GLOBAL_INDEX]  # shape: (batch_size,)

        # 2) Choose AMP dtype
        amp_dtype = torch.bfloat16 if device.type=="cuda" else torch.float32

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type=="cuda")):
            pred_delta_v = model(data_obj)  # shape: (batch_size,)
            # Build target and compute loss if training
            if is_training:
                target_delta_v = data_obj.y.squeeze(-1)
                r_internal_loss = data_obj.R_internal_val.squeeze(-1)
                loss_opt = loss_for_backward_fn(pred_delta_v, target_delta_v, r_internal_loss, v0_from_u, dist_weight_cutoff)
            else:
                loss_opt = torch.tensor(0.0, device=device, dtype=torch.float32)

        # 3) Compute MAPE metric for logging
        if not is_training and pred_delta_v.numel() > 0:
            pred_vref   = v0_from_u + pred_delta_v
            target_vref = data_obj.y_true_vref.squeeze(-1)
            metric_val  = metric_fn_for_eval(pred_vref, target_vref)
        else:
            metric_val = float("nan")

        # 4) Backpropagation if training
        if is_training:
            if not torch.isnan(loss_opt) and not torch.isinf(loss_opt):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_opt).backward()
                scaler.step(optimizer)
                scaler.update()
            if scheduler:
                scheduler.step()

        # 5) Accumulate for logging
        if is_training:
            val_to_add = loss_opt.item()
        else:
            val_to_add = metric_val if not math.isnan(metric_val) else 0.0

        n_valid = num_graphs_in_batch if (is_training or not math.isnan(metric_val)) else 0
        if n_valid > 0:
            total_loss_display += val_to_add * n_valid
            n_graphs_processed += n_valid

    return total_loss_display / n_graphs_processed if n_graphs_processed > 0 else float("nan")
```

Algorithmic Flow:

Set mode: model.train(is_training) toggles dropout/batchnorm if present.

AMP scaling: If on CUDA and use_amp=True, cast to bfloat16 (or float16) for faster matrix multiplies.

Shape handling: The code systematically ensures data_obj.u is a (batch_size×4) tensor, handling edge cases where u might be 1D or empty. This shape normalization reduces accidental dimension mismatches.

Forward pass → predictions.

Loss computation (if is_training=True) with a custom distance-weighted L1.

Metric computation (if is_training=False) using MAPE.

Backprop + optimizer + scheduler if training.

Aggregation: accumulate “loss or metric” × “batch size” so that we can report a weighted average at the end of the epoch.

By parameterizing the loop over loss_for_backward_fn and metric_fn_for_eval, you can re-use run_epoch for both training (use L1 loss) and validation/test (use MAPE) without rewriting the loop logic.

###  5.3 Plotting Predictions vs. Actuals (plot_regression)

```python
def plot_regression(model, loader, device, title="Regression Plot", save_path=None):
    model.eval()
    all_preds_vref, all_targets_vref = [], []

    for data_obj in loader:
        data_obj = data_obj.to(device)
        # 1) Reshape/normalize data_obj.u as in run_epoch
        ...
        if num_graphs_in_batch == 0: continue

        v0 = data_obj.u[:, V0_SCALAR_GLOBAL_INDEX]
        pred_delta_v = model(data_obj)
        if pred_delta_v.numel() == 0: continue

        target_vref = data_obj.y_true_vref.squeeze(-1)
        pred_vref   = v0 + pred_delta_v

        all_preds_vref.append(pred_vref.cpu())
        all_targets_vref.append(target_vref.cpu())

    # 2) Concatenate all batches
    preds_np   = torch.cat(all_preds_vref).numpy()
    targets_np = torch.cat(all_targets_vref).numpy()

    # 3) Scatter plot + identity line
    plt.figure(figsize=(8, 8))
    plt.scatter(targets_np, preds_np, alpha=0.5, label="Predicted vs. Actual")
    min_v = min(np.nanmin(targets_np), np.nanmin(preds_np))
    max_v = max(np.nanmax(targets_np), np.nanmax(preds_np))
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label="Ideal")
    plt.xlabel("Actual V_ref")
    plt.ylabel("Predicted V_ref")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
```

Algorithmic Steps

**Batchwise inference:** identical to `run_epoch`, we ensure `data_obj.u` is correctly shaped, compute `pred_delta_v`, form $\\hat{V}_\\text{ref} = V_0 + \\Delta$, then collect them.

**Concatenate all predictions/targets** into two 1D NumPy arrays.

**Scatter plot** (actual vs. predicted) with an identity line for reference.

**Optionally save to disk.**

Because “reshape `u`” logic is repeated verbatim from `run_epoch`, you see the power of pattern recognition: the author recognized that these shape fixes are necessary wherever we consume `u`.

---

## 6. Main Script: Putting It All Together

Finally, the `main()` function orchestrates the entire experiment:

- **Parse arguments** (or use defaults if running from IDE).

- **Check / build cache:** If `cached_graphs.pt` doesn’t exist, call an external script `preprocess_and_cache.build_cache(...)` to generate it. Otherwise, load and verify metadata so that “cutoff” and “internal scale” settings match.

- **Instantiate datasets:**
  - `ds_train_full = DimerDataset(...)` (on-the-fly augmentation) or
  - `ds_train_full = CachedDimerDataset(...)` (use precomputed graphs with optional cached augmentation).
  - `ds_eval_full = CachedDimerDataset(...)` (no augmentation).

- **Split indices:** Shuffle indices and carve out train/val/test sets (with 10% each for val/test by default, unless dataset < 10 samples).

- **Build PyG DataLoaders** for train/val/test with `batch_size`, `num_workers`, etc.

- **Instantiate model:**

```python
model = DeltaCoupling(num_rbf, internal_rbf_cutoff, learnable_rbf_freqs, num_conv_layers).to(device)
if torch.cuda.is_available() and torch.__version__ >= "2.0.0":
    model = torch.compile(model, backend="inductor" or "aot_eager")
```
his leverages PyTorch 2.0’s `torch.compile` to (optionally) JIT-compile the model for faster training.

## Optimizer & Scheduler:

```python
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

sched = OneCycleLR(...) if training data exists.
```

## Training loop:

Loop epoch = 1..args.epochs:

```python
train_loss = run_epoch(
    train_loader, model, opt, sched, device,
    loss_fn=distance_weighted_l1_on_vref_loss,
    metric_fn=mape_loss_fn,
    is_training=True
)

val_mape = run_epoch(
    val_loader, model, None, None, device,
    loss_fn=None,
    metric_fn=mape_loss_fn,
    is_training=False
)
```

Log progress every 10 epochs, track best validation MAPE, save checkpoint if improved, implement early stopping.

## Evaluation & Plotting:

After training, load best checkpoint, run `run_epoch(test_loader, ...)` to get test MAPE, call `plot_regression` to visualize test set predictions. Repeat for validation set.

Save final model to disk.

Note how the main function never re-implements “how to build a graph” or “how to convolve.” It just ties together:

- Dataset creation (`DimerDataset` or `CachedDimerDataset`)
- Model instantiation (`DeltaCoupling`)
- Training loop (`run_epoch`)
- Plotting (`plot_regression`)

In other words, each block is a black box that plays a well-defined role in the pipeline. This is a hallmark of good decomposition in computational thinking.

## 7. Putting It All Together: Key Takeaways

### Abstraction

Abstraction helps us hide low-level details (PDB parsing, quaternion rotations, RBF calculation). When building complex pipelines, isolate every “how do you parse this file?” or “how do you embed this scalar?” into a reusable function or module.

### Decomposition

Decomposition ensures that each function, class, or method has a single responsibility. Our script breaks down into:

- File parsers (`read_pdb`)
- Feature transform helpers (`decompose_quadrupole_to_real_spherical`, `transform_by_matrix`)
- Basis expansions (`RadialBesselBasisLayer`)
- Dataset constructors (`DimerDataset`, `CachedDimerDataset`)
- Convolutional layers (`PointsConvolutionIntegrated`)
- Full model assembly (`DeltaCoupling`)
- Loss + metric functions (`mape_loss_fn`, `distance_weighted_l1_on_vref_loss`)
- Training/Eval loops (`run_epoch`, `plot_regression`)
- Orchestration (`main`)

Each of these pieces can be tested and understood in isolation, which makes maintenance and debugging far simpler.

### Pattern Recognition

Pattern Recognition reveals that any relative-position edge requires spherical harmonics + distance embedding. By centralizing this in a few abstractions, the code remains DRY (Don't Repeat Yourself). Similarly, “reshaping global scalars” logic is repeated in both training and plotting, because both need a consistent 2D $(\text{batch\_size} \times 4)$ shape for $u$.

### Algorithmic Design

Algorithmic Design emerges through:

- How we build a multi-layer, equivariant message-passing scheme.
- How we implement data augmentation (jitters + random rotations) to improve robustness.
- How we schedule training (learning-rate scheduler, AMP, early stopping).

Each algorithmic decision is clearly implemented step by step, rather than buried in a tangle of nested loops.

### Equivariance & Physics

By relying on E3NN’s irreps and tensor-product layers, we ensure that the network’s predictions transform correctly under rotations. If we rotate each input dimer, the predicted $\Delta V$ remains unchanged (a scalar), and any intermediate vector/tensor features rotate consistently. This is crucial for making physically meaningful predictions.

## Conclusion

The power of computational thinking lies in transforming a sprawling script into a series of manageable, logically connected modules. By combining:

- Abstraction (irreps, radial basis, file parsing)
- Decomposition (small classes/functions for each subtask)
- Pattern Recognition (“any edge → spherical harmonics + RBF”)
- Algorithmic Design (stacked equivariant convolutions, gated residuals, weighted L1 loss)

we turn the formidable task of “predict $V_\text{ref}$ for dimers using E3NN” into a clear, maintainable pipeline. Whether your goal is to extend this model to trimers, add higher-order multipoles, or switch to a different physical property, you already see where each change should occur: in the dataset parser, the model definition, or the loss function—without rewriting everything from scratch.

By adopting these computational-thinking principles, you’ll be well-equipped to tackle your own high-dimensional, physics-informed learning tasks.
