"""How loose is the Pauli-1-norm Trotter bound on 2D Heisenberg patches?

The bounds in 11_trotter_commutator_bound.py relax every commutator spectral
norm to a Pauli 1-norm, which is extensive and therefore safe but loose.  The
variational side of this project will be measured empirically, so comparing a
measured variational error against a bounded Trotter cost would stack the deck.
This script measures the real thing on small 2D patches.

For a p-th order formula the leading error is C_p t^{p+1}, so

    looseness_p = bound_p(t) / ||S_p(t) - e^{-itH}||_inf

evaluated at small t, where the leading order dominates.  The Trotter number
then rescales as  r_empirical = r_bound / looseness_p^{1/p}.

Everything is dense but built without any d^3 operation except one eigh:
each Hamiltonian group is a sum of commuting Paulis, so its exponential is a
product of factors (cos w - i sin w P), and applying a Pauli to a dense matrix
is a signed row permutation.
"""

import importlib.util
import itertools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "tcb", os.path.join(_HERE, "11_trotter_commutator_bound.py"))
tcb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcb)


def _rect(sx, sy, seed=0):
    """X-Y-Z grouped Hamiltonian of an sx-by-sy open rectangle."""
    rng = np.random.default_rng(seed)
    n = sx * sy
    h = rng.uniform(-1.0, 1.0, size=n)
    bonds = []
    for r in range(sy):
        for c in range(sx):
            i = r * sx + c
            if c + 1 < sx:
                bonds.append((i, i + 1))
            if r + 1 < sy:
                bonds.append((i, i + sx))
    H1, H2, H3 = {}, {}, {}
    for i, j in bonds:
        m = (1 << i) | (1 << j)
        H1[(m, 0)] = H1.get((m, 0), 0j) + 1.0
        H2[(m, m)] = H2.get((m, m), 0j) - 1.0
        H3[(0, m)] = H3.get((0, m), 0j) + 1.0
    for j in range(n):
        k = (0, 1 << j)
        H3[k] = H3.get(k, 0j) + h[j]
    return [H1, H2, H3], bonds, n


def _pauli_dense(x, z, n):
    """Dense matrix of X^x Z^z on n qubits, as (row_index, phase) arrays."""
    d = 1 << n
    cols = np.arange(d, dtype=np.int64)
    rows = cols ^ x                       # X^x permutes basis states
    # Z^z contributes (-1)^popcount(z & col)
    sign = np.ones(d)
    bits = z
    while bits:
        b = bits & -bits
        sign *= np.where(cols & b, -1.0, 1.0)
        bits ^= b
    return rows, sign


def _apply_group_exp(M, group, theta, n):
    """M <- exp(-i theta * group) @ M, for a group of mutually commuting Paulis."""
    for (x, z), c in group.items():
        w = theta * c.real if abs(c.imag) < 1e-12 else theta * c
        w = complex(w).real if abs(complex(w).imag) < 1e-12 else w
        rows, sign = _pauli_dense(x, z, n)
        # (cos w) M - i (sin w) P M ; (P M)[k] = sign[k] * M[rows[k]]
        PM = M[rows] * sign[:, None]
        M = np.cos(w) * M - 1j * np.sin(w) * PM
    return M


def _dense_H(groups, n):
    d = 1 << n
    H = np.zeros((d, d), dtype=np.complex128)
    eye = np.eye(d, dtype=np.complex128)
    for g in groups:
        for (x, z), c in g.items():
            rows, sign = _pauli_dense(x, z, n)
            H[rows, np.arange(d)] += c * sign
    return H


_S4_U = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


def _s2(M, groups, t, n):
    for g in groups:
        M = _apply_group_exp(M, g, t / 2.0, n)
    for g in reversed(groups):
        M = _apply_group_exp(M, g, t / 2.0, n)
    return M


def _s1(M, groups, t, n):
    for g in groups:
        M = _apply_group_exp(M, g, t, n)
    return M


def _s4(M, groups, t, n):
    u = _S4_U
    for a in (u, u, 1.0 - 4.0 * u, u, u):
        M = _s2(M, groups, a * t, n)
    return M


def spectral_norm(A, iters=200, tol=1e-13):
    """Largest singular value by power iteration on A^dag A."""
    d = A.shape[0]
    rng = np.random.default_rng(0)
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    v /= np.linalg.norm(v)
    s_old = 0.0
    for _ in range(iters):
        w = A.conj().T @ (A @ v)
        s = np.linalg.norm(w)
        if s == 0:
            return 0.0
        v = w / s
        if abs(s - s_old) <= tol * max(s, 1e-30):
            break
        s_old = s
    return float(np.sqrt(s))


def run(sx, sy, times=(0.02, 0.05, 0.1)):
    groups, bonds, n = _rect(sx, sy)
    d = 1 << n
    print(f"\n### {sx}x{sy} patch, n={n}, {len(bonds)} bonds, d={d}",
          flush=True)

    H1, H2, H3 = groups
    t0 = time.time()
    A1 = tcb.first_order_constant(H1, H2, H3)
    C2 = tcb.second_order_constant(H1, H2, H3)
    C4 = tcb.fourth_order_constant(H1, H2, H3)
    print(f"  bound constants: A1/2={A1/2:.6g}  C2={C2:.6g}  C4={C4:.6g}"
          f"   ({time.time()-t0:.0f}s)", flush=True)

    H = _dense_H(groups, n)
    t0 = time.time()
    w, V = np.linalg.eigh(H)
    print(f"  eigh done ({time.time()-t0:.0f}s)", flush=True)
    del H

    eye = np.eye(d, dtype=np.complex128)
    print(f"  {'t':>7} {'order':>6} {'measured':>12} {'bound':>12} "
          f"{'loose':>8}", flush=True)
    out = {}
    for t in times:
        U = (V * np.exp(-1j * w * t)) @ V.conj().T
        for order, fn, bnd in ((1, _s1, A1 / 2.0 * t ** 2),
                               (2, _s2, C2 * t ** 3),
                               (4, _s4, C4 * t ** 5)):
            S = fn(eye.copy(), groups, t, n)
            err = spectral_norm(S - U)
            out.setdefault(order, []).append(bnd / err if err > 0 else np.inf)
            print(f"  {t:7.3g} {order:6d} {err:12.5g} {bnd:12.5g} "
                  f"{bnd/err:8.3g}x", flush=True)
        del U
    print("  looseness (mean over t): "
          + "  ".join(f"S{o}={np.mean(v):.3g}x" for o, v in sorted(out.items())),
          flush=True)
    return {o: float(np.mean(v)) for o, v in out.items()}


if __name__ == "__main__":
    shapes = [(2, 3), (2, 4), (3, 3), (2, 5), (3, 4)]
    if len(sys.argv) > 1:
        shapes = [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]]
    results = {}
    for sx, sy in shapes:
        results[(sx, sy)] = run(sx, sy)
    print("\n=== summary: bound / measured ===")
    print(f"{'patch':>8} {'n':>4} {'S1':>9} {'S2':>9} {'S4':>9}")
    for (sx, sy), r in results.items():
        print(f"{f'{sx}x{sy}':>8} {sx*sy:4d} {r[1]:8.3g}x {r[2]:8.3g}x "
              f"{r[4]:8.3g}x")
