from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3‑D proj.

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GRO_FILE      = "npt-HK4.gro"
GPICKLE_FILE  = "molecule_graph.gpickle"
Å_PER_NM      = 10.0
XYZ_DIR       = "xyz_pairs"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def guess_element(atomname: str) -> str:
    atomname = atomname.strip()
    if len(atomname) >= 2 and atomname[0].isalpha() and atomname[1].islower():
        return atomname[:2]
    return atomname[0] if atomname else "X"


def minimum_image(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    return vec - box * np.round(vec / box)


def unwrap_molecule(coords: List[np.ndarray], box: np.ndarray) -> List[np.ndarray]:
    ref = coords[0]
    unwrapped = [ref.copy()]
    for c in coords[1:]:
        unwrapped.append(ref + minimum_image(c - ref, box))
    return unwrapped

# -----------------------------------------------------------------------------
# PBC‑aware GRO parser
# -----------------------------------------------------------------------------

def parse_gro(path: str):
    """Return (centers, atoms_unwrap, box_vec)."""
    with open(path, "r") as fh:
        lines = fh.readlines()
    atom_lines = lines[2:-1]
    box_vec = np.array(list(map(float, lines[-1].split()[:3])))

    mol_coords: Dict[str, List[np.ndarray]] = {}
    mol_names:  Dict[str, List[str]] = {}
    for ln in atom_lines:
        resid, resn = ln[0:5].strip(), ln[5:10].strip()
        atomname = ln[10:15].strip()
        xyz = np.array(list(map(float, (ln[20:28], ln[28:36], ln[36:44]))))
        mid = f"{resid}{resn}"
        mol_coords.setdefault(mid, []).append(xyz)
        mol_names.setdefault(mid, []).append(atomname)

    centers: Dict[str, np.ndarray] = {}
    atoms_unwrap: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for mid, crds in mol_coords.items():
        coords_unwrapped = unwrap_molecule(crds, box_vec)
        atoms_unwrap[mid] = [
            (guess_element(name), coord) for name, coord in zip(mol_names[mid], coords_unwrapped)
        ]
        try:
            idx1, idx2 = mol_names[mid].index("O1"), mol_names[mid].index("O2")
        except ValueError:
            continue  # missing O atoms → skip
        centers[mid] = 0.5 * (coords_unwrapped[idx1] + coords_unwrapped[idx2])
    return centers, atoms_unwrap, box_vec

# -----------------------------------------------------------------------------
# Build Gabriel graph (minimum‑image)
# -----------------------------------------------------------------------------

def build_gabriel(centers: Dict[str, np.ndarray], box: np.ndarray) -> nx.Graph:
    names = list(centers)
    pts   = np.vstack([centers[n] for n in names])
    disp  = minimum_image(pts[:, None, :] - pts[None, :, :], box)
    dmat  = np.linalg.norm(disp, axis=2)

    G = nx.Graph()
    for n, c in centers.items():
        G.add_node(n, center=tuple(c))

    n = len(names)
    for i in range(n):
        for j in range(i+1, n):
            dij = dmat[i, j]
            mid = pts[i] + 0.5 * disp[i, j]
            rad2 = (dij / 2.0) ** 2
            if not any(
                (minimum_image(pts[k] - mid, box).dot(minimum_image(pts[k] - mid, box)) < rad2 - 1e-12)
                for k in range(n) if k not in (i, j)
            ):
                G.add_edge(names[i], names[j], weight=float(dij))
    return G

# -----------------------------------------------------------------------------
# Graph persistence & visualisation helpers
# -----------------------------------------------------------------------------

def get_graph(centers: Dict[str, np.ndarray], box: np.ndarray) -> nx.Graph:
    """Load cached graph or build anew using provided centers/box."""
    if os.path.isfile(GPICKLE_FILE):
        return nx.read_gpickle(GPICKLE_FILE)
    G = build_gabriel(centers, box)
    nx.write_gpickle(G, GPICKLE_FILE)
    return G


def draw_graph(H: nx.Graph, title: str):
    xs, ys, zs = zip(*(H.nodes[n]["center"] for n in H))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    for u, v in H.edges():
        xu, yu, zu = H.nodes[u]["center"]
        xv, yv, zv = H.nodes[v]["center"]
        ax.plot([xu, xv], [yu, yv], [zu, zv], lw=0.6, alpha=0.3)
    ax.scatter(xs, ys, zs, s=12)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------

def save_edge_list(G: nx.Graph, fname: str):
    with open(fname, "w") as fh:
        fh.write("nodeA\tnodeB\tdistance_nm\n")
        for u, v, d in G.edges(data="weight"):
            fh.write(f"{u}\t{v}\t{d:.6f}\n")


def write_xyz_pair(u: str, v: str, atoms: Dict[str, List[Tuple[str, np.ndarray]]],
                   centers: Dict[str, np.ndarray], box: np.ndarray):
    if not os.path.isdir(XYZ_DIR):
        os.makedirs(XYZ_DIR)
    duv_min = minimum_image(centers[v] - centers[u], box)
    shift_v = duv_min - (centers[v] - centers[u])
    pair_atoms = atoms[u] + [(e, c + shift_v) for e, c in atoms[v]]
    with open(os.path.join(XYZ_DIR, f"{u}_{v}.xyz"), "w") as fh:
        fh.write(f"{len(pair_atoms)}\nPair {u}-{v} (Å)\n")
        for elem, xyz_nm in pair_atoms:
            x, y, z = xyz_nm * Å_PER_NM
            fh.write(f"{elem:<2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main():
    if not os.path.isfile(GRO_FILE):
        print(f"[error] '{GRO_FILE}' not found")
        sys.exit(1)

    centers, atoms_unwrap, box = parse_gro(GRO_FILE)
    G = get_graph(centers, box)  # loads cache or rebuilds

    # ensure graph stored if not cached already (above covers it)
    save_edge_list(G, "all_network_edges.txt")
    if G.number_of_edges():
        cc = max(nx.connected_components(G), key=len)
        save_edge_list(G.subgraph(cc), "largest_component_edges.txt")

    # xyz export
    for u, v in G.edges():
        write_xyz_pair(u, v, atoms_unwrap, centers, box)

    # visualise
    draw_graph(G, "Full Molecular Network")

    print("[done] Outputs → gpickle, txt, xyz_pairs/ & interactive plot")

if __name__ == "__main__":
    main()
