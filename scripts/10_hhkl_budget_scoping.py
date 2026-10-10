"""First-pass scoping of the HHKL error budget.

Answers, at back-of-envelope accuracy, the question "for overlap size l, how
large can the per-block evolution time t be, and how many block calls m does
the whole simulation need?"

Everything here rests on the empirical Lieb-Robinson error fit of Haah et al.
(arXiv:1801.03922) Figure 2,

    eps_LR(l, t) = alpha * (t * beta / (l + gamma))**(l + gamma),
    alpha = 0.175, beta = 7.9, gamma = 0.95,

which was measured for the *one-dimensional* Heisenberg chain of their Eq. (11)
across a single cut.  Two corrections are applied when we reuse it in 2D:

  * a cut of boundary length b carries b times the norm of boundary terms, and
    Lemma 6 of the same paper is linear in that norm, so eps scales like b;
  * the 2D lattice has coordination 4 instead of 2, so the Lieb-Robinson
    velocity roughly doubles.  We model that as beta -> 2 * beta, which is the
    pessimistic reading.  Both readings are reported.

The point of the script is not precision -- it is to see whether l = 2, 3 land
anywhere near a usable regime.  The real numbers have to come from a direct
numerical measurement of eps_LR on 2D patches (see the plan).
"""

import numpy as np

ALPHA, BETA, GAMMA = 0.175, 7.9, 0.95


def eps_lr(l, t, beta=BETA, boundary=1.0):
    """Lieb-Robinson error of a single m=1 cut with overlap l, time t."""
    e = l + GAMMA
    return boundary * ALPHA * (t * beta / e) ** e


def solve_t(l, T, n_blocks_per_unit_time, eps_budget, beta=BETA, boundary=1.0):
    """Largest per-block time t with m(t) * eps_LR(l, t) <= eps_budget.

    m(t) = n_blocks_per_unit_time * T / t, so the constraint is a pure power
    law in t and can be solved in closed form:
        c * t**(e - 1) <= eps_budget,  c = N * T * alpha * (beta/e)**e * boundary
    """
    e = l + GAMMA
    c = n_blocks_per_unit_time * T * ALPHA * boundary * (beta / e) ** e
    t = (eps_budget / c) ** (1.0 / (e - 1.0))
    m = n_blocks_per_unit_time * T / t
    return t, m


def report_1d(n=100, T=100.0, eps=1e-3):
    """Reproduce Haah et al. Appendix A (1D chain) as a sanity check."""
    print(f"=== 1D chain, n={n}, T={T:g}, eps={eps:g} (Haah Appendix A setting) ===")
    print(f"{'l':>3} {'t':>10} {'m':>12} {'eps_block':>11} {'steps':>9}")
    for l in (3, 5, 7, 8, 9):
        # Haah's merged counting, Eq. (13): m = 3*T*n/(2*t*l)
        per_unit = 1.5 * n / l
        t, m = solve_t(l, T, per_unit, eps / 3.0)
        print(f"{l:3d} {t:10.4g} {m:12.4g} {eps/(3*m):11.3g} {T/t:9.4g}")
    print()


def report_2d(L=10, T=100.0, eps=1e-2, beta_factor=1.0, tag=""):
    """10x10 square lattice, two rounds of hyperplane decomposition.

    Per unit time the merged counting gives (3/2)(L/l) slabs, each of which is
    cut into (3/2)(L/l) blocks, i.e. (9/4)(L/l)**2 blocks per unit time.  The
    largest block is 2l x 2l qubits.

    Round-1 cuts have boundary length L; round-2 cuts have boundary length 2l.
    We charge the dominant (round-1) boundary, boundary = L.
    """
    n = L * L
    beta = BETA * beta_factor
    print(f"=== 2D {L}x{L} (n={n}), T={T:g}, eps={eps:g}{tag} ===")
    print(f"{'l':>3} {'block q':>8} {'t':>11} {'m':>12} {'eps_block':>11} "
          f"{'steps':>10} {'S2 steps/blk':>13}")
    for l in (2, 3, 4, 5, 6, 7):
        per_unit = 2.25 * (L / l) ** 2
        t, m = solve_t(l, T, per_unit, eps / 3.0, beta=beta, boundary=float(L))
        eps_block = eps / (3 * m)
        nq = (2 * l) ** 2
        # crude 2nd-order Trotter cost inside one block: err ~ C * t**3 with
        # C ~ (number of bonds in block) * ||h||**3, ||h|| = 3 for XX+YY+ZZ.
        bonds = 2 * (2 * l) * (2 * l - 1)
        C = bonds * 27.0 / 12.0
        r = max(1.0, (C * t ** 3 / eps_block) ** (1 / 2))  # S2: err ~ C t^3/r^2
        print(f"{l:3d} {nq:8d} {t:11.4g} {m:12.4g} {eps_block:11.3g} "
              f"{T/t:10.4g} {r:13.3g}")
    print()


def report_hhkl_vs_trotter(L=10, T=100.0, eps=1e-2, beta_factor=1.0):
    """Total two-qubit gate count of HHKL-with-Trotter-inside-blocks.

    Every block is charged as if it were the largest one (2l x 2l) and is
    implemented with second-order Trotter, which is the cheapest thing a
    variational circuit could plausibly beat.  The comparison baseline is
    second-order Trotter run directly on the full 10x10 lattice.
    """
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "tcb", os.path.join(os.path.dirname(__file__),
                            "11_trotter_commutator_bound.py"))
    tcb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcb)

    # direct 2nd-order Trotter on the full lattice
    H1, H2, H3, n_bonds = tcb.xyz_groups(L)
    C2_full = tcb.second_order_constant(H1, H2, H3)
    r_full = np.sqrt(T ** 3 * C2_full / (2 * eps / 3))
    cnot_full = r_full * 2 * n_bonds * 3

    print(f"=== HHKL block cost vs. direct S2 Trotter "
          f"(2q gates, beta x{beta_factor:g}) ===")
    print(f"direct S2 on {L}x{L}: r = {r_full:.4g}, "
          f"2q gates = {cnot_full:.4g}")
    print(f"{'l':>3} {'block':>8} {'S2/blk':>9} {'2q/blk':>10} "
          f"{'m':>11} {'HHKL 2q':>11} {'vs direct':>11}")
    for l in (2, 3, 4, 5):
        per_unit = 2.25 * (L / l) ** 2
        t, m = solve_t(l, T, per_unit, eps / 3.0,
                       beta=BETA * beta_factor, boundary=float(L))
        eps_block = eps / (3 * m)
        b1, b2, b3, blk_bonds = tcb.xyz_groups(2 * l)
        C2_blk = tcb.second_order_constant(b1, b2, b3)
        r_blk = max(1.0, np.sqrt(t ** 3 * C2_blk / eps_block))
        cnot_blk = r_blk * 2 * blk_bonds * 3
        total = m * cnot_blk
        print(f"{l:3d} {f'{2*l}x{2*l}':>8} {r_blk:9.3g} {cnot_blk:10.4g} "
              f"{m:11.4g} {total:11.4g} {total / cnot_full:10.3g}x")
    print()


def report_certification():
    """How tight is an L2-type loss as a spectral-norm certificate?"""
    print("=== L_HST / statevector-overlap loss -> spectral norm ===")
    print("||U - V||_inf <= ||U - V||_F = sqrt(d) * ||U - V||_rHS,")
    print("and L_HST = sum_{Q != 1} |c_Q|^2 ~ ||U - V||_rHS^2 (BPPP Eq. E10).")
    print("So a spectral-norm target eps_blk needs L_HST <= eps_blk^2 / d.\n")
    print(f"{'block q':>8} {'d':>12} {'eps_blk':>10} {'required L_HST':>16}")
    for nq, eps_blk in ((16, 6e-11), (16, 1e-6), (36, 3e-9), (36, 1e-6)):
        d = 2.0 ** nq
        print(f"{nq:8d} {d:12.4g} {eps_blk:10.2g} {eps_blk ** 2 / d:16.3g}")
    print()


if __name__ == "__main__":
    report_1d()
    report_2d(beta_factor=1.0, tag="  [optimistic: 1D Lieb-Robinson velocity]")
    report_2d(beta_factor=2.0, tag="  [pessimistic: v_LR doubled for z=4]")
    report_hhkl_vs_trotter()
    report_certification()
