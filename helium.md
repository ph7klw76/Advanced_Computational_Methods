
### 1) Physical model and the target observable

We work with the nonrelativistic, clamped-nucleus Hamiltonian for a two-electron atom,

$$
H = -\frac{1}{2}(\nabla_1^2 + \nabla_2^2) - \frac{Z}{r_1} - \frac{Z}{r_2} + \frac{1}{r_{12}}, \quad Z = 2 \text{ for He.}
$$

The first ionization energy is

$$
I_1 = E(\text{He}^+) - E(\text{He}).
$$

Experimentally for $^4$He: $I_1 = 24.587387$ eV (NIST), which serves as our benchmark.

Because the two-electron Coulomb problem has no closed-form solution, we use a variational discretization: choose a basis $\{\phi_k\}$, form matrices

$$
H_{mn} = \langle \phi_m \vert H \vert \phi_n \rangle, \quad S_{mn} = \langle \phi_m \vert \phi_n \rangle,
$$

and solve the generalized eigenproblem

$$
Hc = E Sc
$$

to obtain a rigorous upper bound $E$ for the exact ground-state energy (Rayleigh–Ritz).

---

### 2) Hylleraas $s,t,u$ coordinates and basis (why they converge fast)

Following Hylleraas, define collective variables

$$
s = r_1 + r_2, \quad t = r_1 - r_2, \quad u = r_{12}.
$$

These variables expose the electron–electron cusp (the $u$-dependence) and the permutation symmetry ($t$ even powers for an $S$-state). Hylleraas and later Pekeris, Drake, Sims–Hagstrom, and many others showed that explicit $u$-dependence produces spectacularly fast variational convergence, reaching many significant digits in nonrelativistic helium.

Our trial space is

$$
\Psi(r_1, r_2) = e^{-\alpha s} \sum_{i=0}^I \sum_{j=0}^J \sum_{k=0}^K C_{ijk} s^i u^j t^{2k},
$$

with a single nonlinear scale $\alpha > 0$ optimized by 1-D search. The even power $t^{2k}$ enforces the correct spatial symmetry for the $1\,^1S$ ground state.

**Why the exponential envelope?**  
It captures the asymptotic hydrogenic decay while the polynomial in $(s,u,t^2)$ captures short- and intermediate-range correlation, including the Kato electron–electron cusp via the explicit $u$. This is the minimal, classic Hylleraas form used in many high-precision computations.

---

### 3) Quadrature and change of variables (how the integrals are stabilized)

All matrix elements reduce to 3-D integrals over $(r_1, r_2, \mu = \cos \gamma)$ after integrating out trivial angles. We perform a scale-free change of variables tailored to the envelope $e^{-\alpha(r_1 + r_2)}$:

$$
x_1 = 2 \alpha r_1, \quad x_2 = 2 \alpha r_2,
$$

so the residual weight becomes $e^{-x_1 - x_2}$. We evaluate radial integrals with Gauss–Laguerre (exact for the Laguerre weight) and the angular $\mu$-integral with Gauss–Legendre. In code:

```python
np.polynomial.laguerre.laggauss(n_r)  # for x1, x2
np.polynomial.legendre.leggauss(n_mu) # for mu
```

The code constructs a tensor product grid $(x_1, x_2, \mu)$, evaluates $s, t, u$, and assembles the weight

$$
W = \frac{\pi^2}{8} \alpha^6 x_1^2 x_2^2 w_{x1} w_{x2} w_\mu,
$$

which already includes Jacobians and angular factors consistent with the reduced integral measure. (See `setup_quadrature(...)`.)

---

### 4) Matrix elements without second derivatives (stable kinetic energy)

Rather than applying $-\frac{1}{2}(\nabla_1^2 + \nabla_2^2)$ directly to basis functions, we use the well-known identity (valid for square-integrable functions vanishing at infinity):

$$
\langle \Psi \vert T \vert \Psi \rangle = \frac{1}{2} \int (|\nabla_1 \Psi|^2 + |\nabla_2 \Psi|^2) \, d^3 r_1 \, d^3 r_2.
$$

With $\Psi = e^{-\alpha s} f(s,t,u)$, the gradients can be written in terms of first derivatives of $f$ only:

$$
\partial_{r_1} \Psi = e^{-\alpha s} \left[(-\alpha) f + f_s + f_t + f_u (\hat{u} \cdot \hat{r}_1) \right],
$$

$$
\partial_{r_2} \Psi = e^{-\alpha s} \left[(-\alpha) f + f_s - f_t - f_u (\hat{u} \cdot \hat{r}_2) \right],
$$

where $f_s = \partial f / \partial s$, etc., and $\hat{u} = (r_1 - r_2)/u$.

The code evaluates the needed scalar products through the precomputed:

$$
u_1 = \hat{u} \cdot \hat{r}_1 = \frac{r_1 - r_2 \mu}{u}, \quad
u_2 = \hat{u} \cdot \hat{r}_2 = \frac{r_1 \mu - r_2}{u}.
$$

From these, the kinetic bilinear form becomes

$$
\langle \phi_m \vert T \vert \phi_n \rangle = \frac{1}{2} \int \left[ (A_m A_n + B_m B_n + (A_m B_n + A_n B_m) u_1) + (C_m C_n + D_m D_n - (C_m D_n + C_n D_m) u_2) \right] W,
$$

with $A = (-\alpha)f + f_s + f_t$, $B = f_u$, $C = (-\alpha)f + f_s - f_t$, $D = f_u$, and $W$ the combined quadrature weight. This avoids second derivatives and improves numerical stability. (See `build_matrices(...)`.)

The potential is algebraic:

$$
V = -Z\left(\frac{1}{r_1} + \frac{1}{r_2}\right) + \frac{1}{u} \quad \Rightarrow \quad V(x_1, x_2, \mu) = -2\alpha Z\left(\frac{1}{x_1} + \frac{1}{x_2} \right) + \frac{2\alpha}{\sqrt{x_1^2 + x_2^2 - 2 x_1 x_2 \mu}},
$$

so the $V$-matrix is simply $\int \phi_m \phi_n V W$.

---

### 5) Generalized eigenproblem and the $\alpha$ search

For a fixed $\alpha$ and basis index set $\{(i,j,k)\}$, the code builds $S$ and $H = T + V$, and solves the symmetric generalized eigenproblem via the $S^{-1/2}$-similarity transform:

$$
S = U \Lambda U^\top, \quad S^{-1/2} = U \Lambda^{-1/2} U^\top, \quad H_t = S^{-1/2} H S^{-1/2}, \quad \min E = \lambda_{\min}(H_t).
$$

(See `lowest_generalized_eigenvalue`.)  
We then optimize $\alpha$ by bracketing + golden-section search, evaluating a few dozen energies; this is cheap compared to building $H, S$. (See `optimize_alpha`.)

**Variational guarantee:** Energies decrease monotonically as you (i) enrich the basis $(I,J,K)$ or (ii) optimize $\alpha$; the result is always an upper bound to the clamped-nucleus, nonrelativistic exact energy.

---

### 6) From energy to ionization energy: recoil, then (tiny) relativistic/QED

The variational calculation gives $E(\text{He})$ for an infinite-mass nucleus. To compare with experiment we must place the He$^+$ threshold at its finite-mass value. Using the reduced mass $\mu = M / (M + 1)$ (in electron masses),

$$
E(\text{He}^+) = -\frac{Z^2}{2} \mu \quad \text{(Hartree)},
$$

so the threshold shifts upward by $\sim 7.5$ meV for $^4$He, slightly reducing the ionization energy compared to the infinite-mass value. The program applies this by default via `--M_over_me` ($\approx 7294.3$). The remaining Breit–Pauli (mass–velocity, Darwin, etc., order $m \alpha^4$) and QED ($m \alpha^5$ and beyond, e.g. Bethe logarithm) corrections are meV to $\mu$eV-scale and only matter once your nonrelativistic baseline is already very tight; hooks are provided as additive meV knobs.

For rigor and magnitudes, see:
- Pachucki/Yerokhin (QED in helium)
- Korobov (Bethe logarithm)
- Drake (Hylleraas high-precision)

**Why this order of operations?**  
Because correlation dominates the error budget by orders of magnitude; once the Hylleraas solver is converged, finite-mass + Breit–Pauli + QED bring you the last few meV required to match the NIST $I_1 = 24.587387$ eV within <1%.

---

### 7) What each major function does (code ↔ math)

- `setup_quadrature(alpha, n_r, n_mu)`: builds the tensor grid in $(x_1, x_2, \mu)$, computes $s, t, u, u_1, u_2$, and the combined weight $W$.
- `hylleraas_indices(I, J, K, total_deg)`: lists monomials $s^i u^j t^{2k}$ (optionally trimmed by total degree).
- `precompute_powers(...)`: caches powers of $s, u, t^2$ for fast reuse in large bases.
- `basis_eval(i, j, k, ...)`: returns $f, f_s, f_t, f_u$ for the monomial; this feeds the kinetic bilinear form.
- `build_matrices(...)`: assembles $S$, $V$, and the kinetic $T$ from the quadratic form in Section 4.
- `lowest_generalized_eigenvalue(H, S)`: computes $\min \text{eig}(S^{-1/2} H S^{-1/2})$.
- `optimize_alpha(...)`: bracketing + golden search for the best $\alpha$.
- `helium_ionization_energy_eV(...)`: converts $E(\text{He})$ to $I_1$ with the finite-mass He$^+$ threshold and (optional) meV-scale add-ons.

---

### 8) Numerical behavior and how to reach <1%

**Convergence knobs:** increase $(I, J, K)$ and quadrature orders $(n_r, n_\mu)$. Typical sweet spots:

- **Fast:** $I = 3$, $J = 3$, $K = 1$, $n_r = 36$, $n_\mu = 80$
- **Tighter:** $I = 6$, $J = 6$, $K = 3$, $n_r = 64$, $n_\mu = 128$

**Optimization:** keep `--opt_alpha 1`. The optimal $\alpha$ changes slightly as you enrich the basis.

**Validation:** monitor the monotone decrease of $E(\text{He})$ and the approach of $I_1$ toward NIST. With the tighter settings above and the finite-mass threshold, you should be safely under 1% absolute error.

**For reference values and the scale of small corrections:**

- NIST ionization energy for $^4$He: **24.587387 eV**
- State-of-the-art nonrelativistic energy (infinite mass) is known to many digits using Hylleraas/ECG/exponential bases.
- Bethe-logarithm and QED corrections are tabulated in Korobov/Pachucki papers; their magnitude confirms they are not the limiting factor at the 1% level.

---

### 9) Limitations and extensions

The program targets the $1\,^1S$ state with a single nonlinear $\alpha$. Adding more nonlinear scales (e.g., a sum of distinct $e^{-\alpha_\ell s}$ envelopes) often accelerates convergence further (Drake’s “triple-basis” strategy).

You can include the **mass-polarization operator** $-\frac{1}{M} \nabla_1 \cdot \nabla_2$ in the neutral-He Hamiltonian as an extra matrix (first-order in $1/M$), rather than treating it as a knob. (Small effect at the meV scale.)

**For Breit–Pauli and QED at first order**, add operator matrices (mass–velocity, Darwin, orbit–orbit; then $m \alpha^5$ self-energy with the Bethe logarithm). The literature provides exact forms and reference values for benchmarking.

**Alternative bases**: explicitly correlated exponentials (Korobov/Drake), Hy-CI (Sims–Hagstrom). Both reach extreme precision; the present Hylleraas implementation aims for clarity and compactness.

---

### 10) Reproducibility checklist

- Fix $Z = 2$, choose $(I, J, K, n_r, n_\mu)$
- Optimize $\alpha$ (on the order of $\sim 1.8 - 2.2$)
- Record $E(\text{He})$ (Hartree)
- Convert with finite-mass threshold to $I_1$ (eV) and compare to NIST: **24.587387 eV**

If desired, add meV-scale relativistic/QED shifts from literature values for even closer agreement.




```python
import numpy as np
import argparse
import sys

HARTREE_TO_EV = 27.211386245988

def gauss_laguerre(n):
    x, w = np.polynomial.laguerre.laggauss(n)
    return x, w

def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def setup_quadrature(alpha, n_r=48, n_mu=96, dtype=np.float64):
    X, WX = gauss_laguerre(n_r)
    MU, WMU = gauss_legendre(n_mu)

    X1, X2, MUg = np.meshgrid(X, X, MU, indexing="ij")
    W1, W2, WMUg = np.meshgrid(WX, WX, WMU, indexing="ij")

    sqrtQ = np.sqrt(np.maximum(X1**2 + X2**2 - 2.0*X1*X2*MUg, np.finfo(dtype).tiny), dtype=dtype)
    Sx = X1 + X2
    Tx = X1 - X2

    s = Sx / (2.0 * alpha)
    t = Tx / (2.0 * alpha)
    u = sqrtQ / (2.0 * alpha)

    invX1 = 1.0 / np.maximum(X1, np.finfo(dtype).tiny)
    invX2 = 1.0 / np.maximum(X2, np.finfo(dtype).tiny)

    u1 = (X1 - X2*MUg) / sqrtQ
    u2 = (X1*MUg - X2) / sqrtQ

    Wgrid = (W1 * W2 * WMUg) * (X1**2) * (X2**2)
    const = (np.pi**2) / (8.0 * (alpha**6))

    return {
        "X1": X1.astype(dtype), "X2": X2.astype(dtype), "MU": MUg.astype(dtype),
        "sqrtQ": sqrtQ.astype(dtype), "Sx": Sx.astype(dtype), "Tx": Tx.astype(dtype),
        "s": s.astype(dtype), "t": t.astype(dtype), "u": u.astype(dtype),
        "u1": u1.astype(dtype), "u2": u2.astype(dtype),
        "invX1": invX1.astype(dtype), "invX2": invX2.astype(dtype),
        "Wgrid": Wgrid.astype(dtype), "const": dtype(const),
        "alpha": float(alpha)
    }

def hylleraas_indices(Imax, Jmax, Kmax, total_deg=None):
    idx = []
    for i in range(Imax+1):
        for j in range(Jmax+1):
            for k in range(Kmax+1):
                if total_deg is None or (i + j + 2*k) <= total_deg:
                    idx.append((i,j,k))
    return idx

def precompute_powers(grid, Imax, Jmax, Kmax):
    s = grid["s"]; u = grid["u"]; t = grid["t"]
    S_pow = [np.ones_like(s)]
    for i in range(1, Imax+1):
        S_pow.append(S_pow[-1] * s)
    U_pow = [np.ones_like(u)]
    for j in range(1, Jmax+1):
        U_pow.append(U_pow[-1] * u)
    t2 = t * t
    T2_pow = [np.ones_like(t2)]
    for k in range(1, Kmax+1):
        T2_pow.append(T2_pow[-1] * t2)
    T_odd = [None] * (Kmax+1)
    for k in range(1, Kmax+1):
        T_odd[k] = T2_pow[k-1] * t
    return S_pow, U_pow, T2_pow, T_odd

def basis_eval(i,j,k, S_pow, U_pow, T2_pow, T_odd):
    f = S_pow[i] * U_pow[j] * T2_pow[k]
    fs = (i * S_pow[i-1] * U_pow[j] * T2_pow[k]) if i>=1 else 0.0
    if k>=1:
        ft = (2*k) * (S_pow[i] * U_pow[j] * T_odd[k])
    else:
        ft = 0.0
    fu = (j * S_pow[i] * U_pow[j-1] * T2_pow[k]) if j>=1 else 0.0
    return f, fs, ft, fu

def build_matrices(alpha, Z, idx_list, quad, powers):
    S_pow, U_pow, T2_pow, T_odd = powers
    sqrtQ = quad["sqrtQ"]
    u1, u2 = quad["u1"], quad["u2"]
    invX1, invX2 = quad["invX1"], quad["invX2"]
    Wgrid, const = quad["Wgrid"], quad["const"]
    a = quad["alpha"]

    termV = -2.0 * a * Z * (invX1 + invX2) + (2.0 * a) / sqrtQ

    dim = len(idx_list)
    S = np.zeros((dim, dim), dtype=np.float64)
    V = np.zeros((dim, dim), dtype=np.float64)
    T = np.zeros((dim, dim), dtype=np.float64)

    f_list = []; fs_list = []; ft_list = []; fu_list = []
    for (i,j,k) in idx_list:
        f, fs, ft, fu = basis_eval(i,j,k, S_pow, U_pow, T2_pow, T_odd)
        f_list.append(f); fs_list.append(fs); ft_list.append(ft); fu_list.append(fu)

    for m,(im,jm,km) in enumerate(idx_list):
        fm = f_list[m]; fsm = fs_list[m]; ftm = ft_list[m]; fum = fu_list[m]
        Am = (-a)*fm + fsm + ftm
        Cm = (-a)*fm + fsm - ftm
        Bm = fum; Dm = fum
        for n in range(m, dim):
            fn = f_list[n]; fsn = fs_list[n]; ftn = ft_list[n]; fun = fu_list[n]
            An = (-a)*fn + fsn + ftn
            Cn = (-a)*fn + fsn - ftn
            Bn = fun; Dn = fun

            Smn = const * np.sum(Wgrid * (fm * fn))
            Vmn = const * np.sum(Wgrid * (fm * fn * termV))

            g11 = (Am*An + Bm*Bn + (Am*Bn + An*Bm) * u1)
            g22 = (Cm*Cn + Dm*Dn - (Cm*Dn + Cn*Dm) * u2)
            Tmn = 0.5 * const * np.sum(Wgrid * (g11 + g22))

            S[m,n] = S[n,m] = Smn
            V[m,n] = V[n,m] = Vmn
            T[m,n] = T[n,m] = Tmn

    H = T + V
    return S, H

def lowest_generalized_eigenvalue(H, S):
    evals, U = np.linalg.eigh(S)
    eps = 1e-14
    evals = np.maximum(evals, eps)
    S_minushalf = (U / np.sqrt(evals)) @ U.T
    Ht = S_minushalf @ H @ S_minushalf
    w, _ = np.linalg.eigh(Ht)
    return float(np.min(w))

def energy_for_alpha(alpha, Z, idx_list, n_r, n_mu):
    quad = setup_quadrature(alpha, n_r=n_r, n_mu=n_mu)
    Imax = max(i for i,_,_ in idx_list)
    Jmax = max(j for _,j,_ in idx_list)
    Kmax = max(k for _,_,k in idx_list)
    powers = precompute_powers(quad, Imax, Jmax, Kmax)
    S, H = build_matrices(alpha, Z, idx_list, quad, powers)
    return lowest_generalized_eigenvalue(H, S)

def bracket_minimum(f, a, b, n=13):
    xs = np.linspace(a, b, n)
    ys = [f(x) for x in xs]
    k = int(np.argmin(ys))
    lo = xs[max(0,k-1)]; hi = xs[min(n-1,k+1)]
    return lo, hi

def golden_search(f, a, b, tol=3e-4, maxit=40):
    gr = 0.5*(np.sqrt(5.0)-1.0)
    c = b - gr*(b-a); d = a + gr*(b-a)
    fc = f(c); fd = f(d); it = 0
    while abs(b-a) > tol and it < maxit:
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - gr*(b-a)
            fc = f(c)
        else:
            a, fc = c, fd
            c = d
            d = a + gr*(b-a)
            fd = f(d)
        it += 1
    xopt = 0.5*(a+b)
    return xopt, f(xopt)

def optimize_alpha(Z, idx_list, n_r, n_mu, a_lo=1.5, a_hi=2.2):
    f = lambda a: energy_for_alpha(a, Z, idx_list, n_r, n_mu)
    lo, hi = bracket_minimum(f, a_lo, a_hi, n=13)
    aopt, Eopt = golden_search(f, lo, hi, tol=2e-4, maxit=32)
    return aopt, Eopt

def helium_ionization_energy_eV(E_He, M_over_me=7294.299541425, Z=2.0,
                                delta_rel_qed_eV=0.0, delta_mp_He_eV=0.0):
    mu = M_over_me / (M_over_me + 1.0)
    E_He_plus = - (Z**2) * 0.5 * mu  # Hartree
    IE_Ha = E_He_plus - E_He
    IE_eV = IE_Ha * HARTREE_TO_EV + delta_rel_qed_eV + delta_mp_He_eV
    return IE_eV, E_He_plus

def run_solver(Imax, Jmax, Kmax, total_deg, n_r, n_mu, Z, optimize, alpha_init, a_lo, a_hi,
               M_over_me, delta_rel_qed_eV, delta_mp_He_eV):
    idx = hylleraas_indices(Imax, Jmax, Kmax, total_deg=total_deg)
    dim = len(idx)
    if optimize:
        aopt, E_He = optimize_alpha(Z, idx, n_r, n_mu, a_lo=a_lo, a_hi=a_hi)
        alpha_used = aopt
    else:
        alpha_used = alpha_init if alpha_init is not None else 1.85
        E_He = energy_for_alpha(alpha_used, Z, idx, n_r, n_mu)

    IE_eV, E_He_plus = helium_ionization_energy_eV(E_He, M_over_me=M_over_me, Z=Z,
                                                   delta_rel_qed_eV=delta_rel_qed_eV,
                                                   delta_mp_He_eV=delta_mp_He_eV)

    return {
        "dim": dim,
        "alpha": alpha_used,
        "E_He_Ha": E_He,
        "E_He_plus_Ha": E_He_plus,
        "IE_eV": IE_eV
    }

def cli(argv=None):
    p = argparse.ArgumentParser(description="Helium ground state via Hylleraas s,t,u basis + optional small corrections", add_help=True)
    p.add_argument("--I", type=int, default=3, help="max power of s")
    p.add_argument("--J", type=int, default=3, help="max power of u")
    p.add_argument("--K", type=int, default=1, help="max half-power of even t (i.e., t^{2k})")
    p.add_argument("--total_deg", type=int, default=None, help="optional total degree cutoff: i + j + 2k <= total_deg")
    p.add_argument("--nr", type=int, default=28, help="Gauss-Laguerre points per radial dimension")
    p.add_argument("--nmu", type=int, default=64, help="Gauss-Legendre points for μ")
    p.add_argument("--Z", type=float, default=2.0, help="nuclear charge (2 for He)")
    p.add_argument("--opt_alpha", type=int, default=1, help="1 to optimize α, 0 to use --alpha")
    p.add_argument("--alpha", type=float, default=1.85, help="fixed α if --opt_alpha=0")
    p.add_argument("--a_lo", type=float, default=1.5, help="lower α bound for optimization")
    p.add_argument("--a_hi", type=float, default=2.2, help="upper α bound for optimization")
    p.add_argument("--M_over_me", type=float, default=7294.299541425, help="nuclear mass in electron masses (≈7294.2995 for 4He nucleus)")
    p.add_argument("--delta_rel_qed_meV", type=float, default=0.0, help="optional additive relativistic+QED correction to IE in meV")
    p.add_argument("--delta_mp_He_meV", type=float, default=0.0, help="optional mass-polarization correction for neutral He in meV")
    args, _ = p.parse_known_args(argv)

    delta_rel_qed_eV = args.delta_rel_qed_meV * 1e-3
    delta_mp_He_eV   = args.delta_mp_He_meV   * 1e-3

    out = run_solver(
        Imax=args.I, Jmax=args.J, Kmax=args.K, total_deg=args.total_deg,
        n_r=args.nr, n_mu=args.nmu, Z=args.Z,
        optimize=bool(args.opt_alpha), alpha_init=args.alpha, a_lo=args.a_lo, a_hi=args.a_hi,
        M_over_me=args.M_over_me,
        delta_rel_qed_eV=delta_rel_qed_eV, delta_mp_He_eV=delta_mp_He_eV
    )

    NIST_IE = 24.587387  # eV (first ionization energy of 4He)
    err_pct = abs(out["IE_eV"] - NIST_IE) / NIST_IE * 100.0

    print("----- Helium 1^1S Ground State (Hylleraas s,t,u) -----")
    print(f"Basis size (dim)     : {out['dim']}  (I={args.I}, J={args.J}, K={args.K}, total_deg={args.total_deg})")
    print(f"Quadrature (nr,nmu)  : ({args.nr}, {args.nmu})")
    print(f"Optimized alpha      : {out['alpha']:.6f}")
    print(f"E(He)                : {out['E_He_Ha']:.12f} Hartree")
    print(f"E(He+) threshold     : {out['E_He_plus_Ha']:.12f} Hartree (finite mass)")
    print(f"First ionization IE  : {out['IE_eV']:.6f} eV  (vs NIST {NIST_IE:.6f} eV)")
    print(f"Absolute % error     : {err_pct:.3f}%")
    print()
    print("Tips: Increase (I,J,K) and (nr,nmu) for higher accuracy; re-optimize α each time.")

if __name__ == "__main__":
    cli()
```
