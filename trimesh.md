# Visualizing molecule using Trimesh

```python

import numpy as np
import trimesh

# --- CONFIG ---
gro_path = "C:/Users/User/Downloads/molecule_100.gro"             # <- change if needed
sphere_radius_scale = 1.5                 # balls (atoms): 2.0 × van der Waals radius
bond_radius = 0.15                        # sticks (bonds): cylinder radius in Å
sphere_subdiv = 2                         # atom sphere detail (2 is moderate)


# --- Minimal GRO parser (coords in nm -> convert to Å) ---
def parse_gro(path):
    with open(path, "r") as f:
        _title = f.readline()
        n = int(f.readline().strip())
        atoms = []
        for _ in range(n):
            line = f.readline()
            x = float(line[20:28]) * 10.0
            y = float(line[28:36]) * 10.0
            z = float(line[36:44]) * 10.0
            name = line[10:15].strip()
            atoms.append((name, np.array([x, y, z], dtype=float)))
        _ = f.readline()  # box
    return atoms

atoms = parse_gro(gro_path)

# --- Element inference ---
def infer_element(atomname):
    # common water aliases
    if atomname in ("OW", "HW", "HW1", "HW2"): return "O" if atomname=="OW" else "H"
    # simple: first letter, capitalize second if lowercase
    a = ''.join([c for c in atomname if c.isalpha()])
    if not a: return "C"
    if len(a) >= 2 and a[1].islower(): return (a[0]+a[1]).capitalize()
    return a[0].upper()

# --- Radii (Å) ---
vdw = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"P":1.80,"S":1.80,"Cl":1.75,"Na":2.27,"K":2.75,"Ca":2.31}
cov = {"H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"P":1.07,"S":1.05,"Cl":1.02,"Na":1.66,"K":2.03,"Ca":1.74}

coords = np.array([p for _, p in atoms])
elements = [infer_element(n) for n, _ in atoms]
vdw_r = np.array([vdw.get(e, 1.70) for e in elements])
cov_r = np.array([cov.get(e, 0.77) for e in elements])

# --- Build ball-and-stick meshes ---
meshes = []

# balls
for pos, r in zip(coords, vdw_r * sphere_radius_scale):
    sph = trimesh.creation.icosphere(subdivisions=sphere_subdiv, radius=float(r))
    sph.apply_translation(pos)
    meshes.append(sph)

# bonds (distance criterion with covalent radii)
d = np.linalg.norm(coords[None,:,:] - coords[:,None,:], axis=-1)
thr = 1.2 * (cov_r[:,None] + cov_r[None,:])
np.fill_diagonal(d, np.inf)
pairs = np.transpose(np.where(d < thr))
pairs = pairs[pairs[:,0] < pairs[:,1]]

for i, j in pairs:
    seg = np.vstack((coords[i], coords[j]))
    cyl = trimesh.creation.cylinder(radius=bond_radius, segment=seg, sections=24)
    meshes.append(cyl)

molecule = trimesh.util.concatenate(meshes)

# --- Ray test: choose a ray that shoots toward the molecule's center ---
bb_min, bb_max = molecule.bounds
center = (bb_min + bb_max) / 2.0
origin = bb_min - np.array([10.0, 0.0, 0.0])     # start left of the bbox
direction = (center - origin); direction /= np.linalg.norm(direction)

# Intersect (True if any hit)
hits_any = molecule.ray.intersects_any(
    ray_origins=origin.reshape(1,3),
    ray_directions=direction.reshape(1,3)
)

print(f"Atoms: {len(atoms)} | Bonds: {len(pairs)}")
print(f"Ray origin: {origin}, direction: {direction}")
print("Ray passes through molecule?" , bool(hits_any))

```
<img width="407" height="292" alt="image" src="https://github.com/user-attachments/assets/7c1bbe14-de5f-4513-831e-a1d08da19a63" />
