"""T-gate cost of HHKL + QSP/qubitization blocks, following Haah et al. App. E.

For one block of k qubits whose Hamiltonian is a sum of M Pauli terms with
coefficient 1-norm alpha, evolved for time t to error eps_blk:

  * QSP truncation error (Eq. 75):  eps <= 32 (alpha t)^q / (2^q q!),
    with q = N/2 + 1 and N the number of controlled-qubiterate calls.
  * One qubiterate W = reflection about |G> composed with the select oracle O.
    O is a SELECT over M Pauli strings: ~M Toffoli-equivalents.  |G> is a
    uniform superposition (all Heisenberg bond coefficients are equal), so G
    costs O(log M) Hadamards plus a (log M)-controlled Z for the reflection.
  * Each Toffoli is charged 4 T gates.
  * The N QSP phase rotations are synthesised with Ross-Selinger,
    1.15 log2(1/eps_synth) + 9.2 T gates each.

This is the Haah et al. Figure 3 cost model, applied to the 10x10 lattice.
"""

import math

import numpy as np

import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "scoping", os.path.join(os.path.dirname(__file__),
                            "10_hhkl_budget_scoping.py"))
scoping = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoping)


def qsp_degree(alpha, t, eps):
    """Smallest q with 32 (alpha t)^q / (2^q q!) <= eps."""
    logx = math.log(alpha * t / 2.0)
    log_eps = math.log(eps)
    q = 1
    while math.log(32.0) + q * logx - math.lgamma(q + 1.0) > log_eps:
        q += 1
        if q > 5000:
            break
    return q


def block_spec(l):
    """(qubits, Pauli-term count M, coefficient 1-norm alpha) of a 2l x 2l block."""
    L = 2 * l
    bonds = 2 * L * (L - 1)
    M = 3 * bonds + L * L          # XX, YY, ZZ per bond + one field term/site
    alpha = 3.0 * bonds + 0.5 * L * L   # E|h_j| = 1/2 for h_j ~ U[-1,1]
    return L * L, M, alpha


def t_count_per_block(M, alpha, t, eps_blk):
    """T gates for one QSP/qubitization block call."""
    eps_qsp = eps_blk / 2.0
    eps_synth_total = eps_blk / 2.0
    q = qsp_degree(alpha, t, eps_qsp)
    N = max(2, 2 * (q - 1))                 # controlled-qubiterate calls
    logM = max(1, int(np.ceil(np.log2(M))))
    toff_per_W = M + (logM - 1)             # SELECT + reflection about |G>
    t_select = N * 4 * toff_per_W
    eps_rot = eps_synth_total / N
    t_rot = N * (1.15 * np.log2(1.0 / eps_rot) + 9.2)
    return q, N, t_select + t_rot


def main(L=10, T=100.0, eps=1e-2, beta_factor=1.0):
    print(f"HHKL + QSP/qubitization, {L}x{L} Heisenberg, T={T:g}, eps={eps:g}, "
          f"beta x{beta_factor:g}")
    print(f"{'l':>3} {'block':>8} {'M':>6} {'alpha':>8} {'t':>11} "
          f"{'m':>11} {'eps_blk':>10} {'q':>4} {'N':>5} {'T/blk':>10} "
          f"{'total T':>11}")
    for l in (2, 3, 4, 5, 6, 7):
        per_unit = 2.25 * (L / l) ** 2
        t, m = scoping.solve_t(l, T, per_unit, eps / 3.0,
                               beta=scoping.BETA * beta_factor,
                               boundary=float(L))
        eps_blk = eps / (3 * m)
        nq, M, alpha = block_spec(l)
        q, N, tb = t_count_per_block(M, alpha, t, eps_blk)
        print(f"{l:3d} {f'{2*l}x{2*l}':>8} {M:6d} {alpha:8.1f} {t:11.4g} "
              f"{m:11.4g} {eps_blk:10.3g} {q:4d} {N:5d} {tb:10.4g} "
              f"{m * tb:11.4g}")
    print()


if __name__ == "__main__":
    main(beta_factor=1.0)
    main(beta_factor=2.0)
