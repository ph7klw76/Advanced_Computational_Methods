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

raw gro file

```text
mol only system in water
84
    1HK4    H28 8317  10.309  10.534   6.605 -0.8042  2.9044 -1.5704
    1HK4    C48 8318  10.213  10.487   6.586  0.3206 -0.1392 -0.1323
    1HK4    C47 8319  10.205  10.351   6.559 -0.0039 -0.1203 -0.1355
    1HK4    H27 8320  10.299  10.297   6.545 -0.6760 -1.5097  0.5792
    1HK4    C46 8321  10.085  10.285   6.534 -0.0265  0.0183 -0.3944
    1HK4    H26 8322  10.079  10.176   6.528 -0.6111 -0.0681  1.2339
    1HK4    C40 8323   9.968  10.364   6.530  0.1680  0.3184 -0.2114
    1HK4    C50 8324   9.972  10.503   6.553  0.2328  0.1886  0.5626
    1HK4    H30 8325   9.881  10.564   6.543 -0.1518 -0.6472 -1.1172
    1HK4    C49 8326  10.095  10.561   6.578  0.3083 -0.0555  0.7701
    1HK4    H29 8327  10.105  10.668   6.601 -2.5883  0.6282 -0.9382
    1HK4     N2 8328   9.853  10.312   6.469  0.0601  0.4006 -0.0742
    1HK4    C30 8329   9.856  10.221   6.361  0.0807 -0.4074  0.6008
    1HK4    C29 8330   9.962  10.190   6.278 -0.4157 -0.1950 -0.1135
    1HK4    H15 8331  10.045  10.259   6.259 -0.1193 -0.6932 -0.6468
    1HK4    C28 8332   9.959  10.080   6.194 -0.3121  0.2805 -0.7468
    1HK4    C27 8333   9.846   9.996   6.205 -0.2100  0.2533  0.0992
    1HK4    H14 8334   9.844   9.893   6.167  0.0233  0.2953 -0.0279
    1HK4    C32 8335   9.739  10.022   6.290 -0.8379 -0.1974 -0.5446
    1HK4    H16 8336   9.654   9.955   6.302 -1.4579  0.3657 -1.7757
    1HK4    C31 8337   9.737  10.138   6.367 -0.0467 -0.2963 -0.3645
    1HK4    C34 8338   9.650  10.196   6.468 -0.0200 -0.3060 -0.3369
    1HK4    C33 8339   9.726  10.303   6.530  0.2747 -0.8330  0.2113
    1HK4    C35 8340   9.660  10.378   6.627  0.1768 -0.5330 -0.0862
    1HK4    H17 8341   9.711  10.466   6.668  0.6190 -1.1739  0.7678
    1HK4    C36 8342   9.528  10.356   6.665  0.1003 -0.1084 -0.1074
    1HK4    H18 8343   9.475  10.413   6.742 -0.6709 -0.6499 -0.2322
    1HK4    C37 8344   9.461  10.246   6.613  0.1581 -0.3271  0.2810
    1HK4    H19 8345   9.366  10.221   6.659  0.0983  0.2564  0.4724
    1HK4    C38 8346   9.523  10.168   6.515 -0.1488  0.2249 -0.3579
    1HK4    H20 8347   9.474  10.078   6.478  0.6060 -0.5585  0.5419
    1HK4    C14 8348  10.072  10.070   6.099  0.6002 -0.0941  0.3636
    1HK4     C9 8349  10.078   9.997   5.980 -0.0969 -0.5688  0.6117
    1HK4     C8 8350  10.203   9.981   5.917 -0.1753 -0.1044  0.3375
    1HK4    C11 8351  10.320  10.037   5.968 -0.2595  0.3100  0.0762
    1HK4     H4 8352  10.390  10.073   5.891 -2.2258  0.3554 -1.7239
    1HK4    C12 8353  10.320  10.093   6.096  0.5462  0.1966  0.1288
    1HK4     H5 8354  10.414  10.143   6.125  0.5307 -0.6184  1.6008
    1HK4    C13 8355  10.196  10.105   6.157  0.6259 -0.2040  0.3739
    1HK4     H6 8356  10.204  10.116   6.266  0.1097  1.9037  0.2239
    1HK4     C7 8357  10.224   9.872   5.820  0.2433  0.0930  0.2053
    1HK4     O2 8358  10.298   9.892   5.724  0.2958 -0.2383  0.1754
    1HK4     C4 8359  10.126   9.761   5.821  0.2160  0.1106 -0.1377
    1HK4     C3 8360  10.164   9.645   5.753 -0.2632 -0.2564  0.2176
    1HK4     H3 8361  10.271   9.622   5.747 -0.0968  0.9077 -1.4847
    1HK4     C2 8362  10.073   9.549   5.712  0.3395 -0.9435  0.4717
    1HK4     H2 8363  10.105   9.460   5.656  1.6751  0.8509 -1.7099
    1HK4     C1 8364   9.942   9.570   5.752  0.1871  0.3861 -0.7132
    1HK4     H1 8365   9.873   9.486   5.736  1.2280 -0.5221 -0.4734
    1HK4     C6 8366   9.899   9.694   5.806  0.1957 -0.1311  0.4861
    1HK4     C5 8367   9.991   9.793   5.844  0.1936  0.2069 -0.3928
    1HK4    C10 8368   9.967   9.915   5.926 -0.1270 -0.0981 -0.0331
    1HK4     O1 8369   9.858   9.971   5.917  0.0283  0.1850 -0.1688
    1HK4    C16 8370   9.762   9.731   5.767  0.2754  0.0076  0.3308
    1HK4    C15 8371   9.720   9.675   5.645 -0.2585  0.8717  0.1162
    1HK4     H7 8372   9.768   9.699   5.549  0.6620  0.1039  0.3712
    1HK4    C20 8373   9.601   9.604   5.641  0.1172  0.2323 -0.0788
    1HK4     H9 8374   9.572   9.557   5.547  0.9181 -0.1533 -0.1312
    1HK4    C19 8375   9.521   9.584   5.754  0.0418 -0.4288 -0.2502
    1HK4    C18 8376   9.555   9.658   5.874 -0.1236  0.1000 -0.5225
    1HK4    C17 8377   9.674   9.729   5.874 -0.0539 -0.0144  0.0571
    1HK4     H8 8378   9.694   9.800   5.955 -2.1417 -0.7387  1.2627
    1HK4     N1 8379   9.452   9.628   5.966  0.0757  0.4104 -0.2007
    1HK4    C21 8380   9.368   9.526   5.917 -0.1252  0.0963  0.7766
    1HK4    C22 8381   9.405   9.501   5.779 -0.4078  0.4566  0.6342
    1HK4    C26 8382   9.355   9.386   5.719 -0.0428  0.2344  0.7529
    1HK4    H13 8383   9.416   9.352   5.635 -1.0045  2.2459 -0.7766
    1HK4    C25 8384   9.256   9.312   5.783 -0.2164 -0.1463  0.0476
    1HK4    H12 8385   9.194   9.248   5.720 -0.7657  1.2069 -0.8073
    1HK4    C24 8386   9.220   9.339   5.914  0.0486 -0.4291  0.1813
    1HK4    H11 8387   9.144   9.276   5.964 -1.6263  0.6419 -0.9829
    1HK4    C23 8388   9.280   9.442   5.985  0.5473 -0.8943  0.4341
    1HK4    H10 8389   9.252   9.461   6.089  1.6524  0.5912  0.4877
    1HK4    C39 8390   9.429   9.690   6.090  0.1279  0.0989 -0.0348
    1HK4    C41 8391   9.297   9.720   6.127  0.1830  0.7310 -0.3433
    1HK4    H21 8392   9.207   9.692   6.069  0.2815 -2.9437  1.1609
    1HK4    C42 8393   9.269   9.771   6.253  0.3833 -0.3147  0.1229
    1HK4    H22 8394   9.163   9.788   6.276  0.8445  1.4028  1.0441
    1HK4    C43 8395   9.369   9.796   6.347  0.1259  0.0149  0.3108
    1HK4    H23 8396   9.348   9.848   6.440  1.4131 -1.7727  1.6362
    1HK4    C44 8397   9.497   9.753   6.315  0.1391  0.0338  0.3393
    1HK4    H24 8398   9.580   9.776   6.383 -0.0298  0.1419  0.5064
    1HK4    C45 8399   9.528   9.700   6.190 -0.1135  0.2312  0.1909
    1HK4    H25 8400   9.625   9.652   6.172 -1.4345 -1.9781 -1.2928
  11.24798  11.24798  11.24798
```
