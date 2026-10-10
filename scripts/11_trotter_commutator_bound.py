"""Tight first- and second-order Trotter bounds for the 2D Heisenberg model.

Implements Propositions 15 and 16 of Childs, Su, Tran, Wiebe, Zhu,
"Theory of Trotter Error" (arXiv:1912.08854):

    ||S1(t) - e^{-itH}|| <= (t^2/2)  sum_{g1} || sum_{g2>g1} [H_g2, H_g1] ||
    ||S2(t) - e^{-itH}|| <= (t^3/12) sum_{g1} || [sum_{g3>g1} H_g3,
                                                 [sum_{g2>g1} H_g2, H_g1]] ||
                         + (t^3/24) sum_{g1} || [H_g1,
                                                 [H_g1, sum_{g2>g1} H_g2]] ||

The spectral norms of the (extensive) commutator sums are upper-bounded by the
Pauli 1-norm, which is what the bound is normally evaluated with.  That is the
same relaxation used in the reference implementation of the paper.

Hamiltonian (Haah et al. Eq. (11) lifted to a square lattice):

    H = sum_<ij> (X_i X_j + Y_i Y_j + Z_i Z_j) + sum_j h_j Z_j,  h_j ~ U[-1, 1]

grouped X-Y-Z as in Eq. (155) of the Trotter-error paper:
H1 = sum XX, H2 = sum YY, H3 = sum ZZ + field.  Every summand inside a group
commutes, so each exponential is exact.
"""

import numpy as np

# --- sparse Pauli algebra -------------------------------------------------
# A Pauli string is stored as (x_mask, z_mask) meaning X^x Z^z, with an
# explicit complex coefficient.  Products pick up (-1)^popcount(z1 & x2).


def _popcount(v):
    return bin(v).count("1")


def commutator(A, B):
    """[A, B] for two dicts {(x, z): coeff}."""
    out = {}
    for (x1, z1), c1 in A.items():
        for (x2, z2), c2 in B.items():
            if _popcount(z1 & x2) % 2 == _popcount(z2 & x1) % 2:
                continue  # they commute, contribution cancels
            s1 = -1 if _popcount(z1 & x2) % 2 else 1
            s2 = -1 if _popcount(z2 & x1) % 2 else 1
            key = (x1 ^ x2, z1 ^ z2)
            out[key] = out.get(key, 0j) + (s1 - s2) * c1 * c2
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def add(*terms):
    out = {}
    for A in terms:
        for k, v in A.items():
            out[k] = out.get(k, 0j) + v
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def one_norm(A):
    return sum(abs(v) for v in A.values())


# --- model ---------------------------------------------------------------


def square_lattice_bonds(L, periodic=False):
    bonds = []
    for r in range(L):
        for c in range(L):
            i = r * L + c
            if c + 1 < L:
                bonds.append((i, r * L + c + 1))
            elif periodic and L > 2:
                bonds.append((i, r * L))
            if r + 1 < L:
                bonds.append((i, (r + 1) * L + c))
            elif periodic and L > 2:
                bonds.append((i, c))
    return bonds


def xyz_groups(L, periodic=False, seed=0):
    bonds = square_lattice_bonds(L, periodic)
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, size=L * L)

    H1, H2, H3 = {}, {}, {}
    for i, j in bonds:
        m = (1 << i) | (1 << j)
        H1[(m, 0)] = H1.get((m, 0), 0j) + 1.0          # X_i X_j
        H2[(m, m)] = H2.get((m, m), 0j) - 1.0          # Y_i Y_j = -X^m Z^m
        H3[(0, m)] = H3.get((0, m), 0j) + 1.0          # Z_i Z_j
    for j in range(L * L):
        k = (0, 1 << j)
        H3[k] = H3.get(k, 0j) + h[j]                   # h_j Z_j
    return H1, H2, H3, len(bonds)


# --- bounds --------------------------------------------------------------


def first_order_constant(H1, H2, H3):
    """A1 with ||S1(t) - e^{-itH}|| <= (t^2/2) * A1."""
    c1 = one_norm(commutator(add(H2, H3), H1))
    c2 = one_norm(commutator(H3, H2))
    return c1 + c2


def second_order_constant(H1, H2, H3):
    """C2 with ||S2(t) - e^{-itH}|| <= t^3 * C2."""
    H23 = add(H2, H3)
    a = (one_norm(commutator(H23, commutator(H23, H1)))
         + one_norm(commutator(H3, commutator(H3, H2))))
    b = (one_norm(commutator(H1, commutator(H1, H23)))
         + one_norm(commutator(H2, commutator(H2, H3))))
    return a / 12.0 + b / 24.0


# Table 2 of arXiv:1912.08854, coefficients c_{i,j,k,l,m} of Proposition J.2.
# Rows are (i, j, k) in lexicographic order; the three columns are the
# innermost pairs (l, m) = (2,1), (3,1), (3,2).
_TABLE2 = [
    (0.0047, 0.0047, 0.0043), (0.0057, 0.0057, 0.0057),
    (0.0057, 0.0057, 0.0057), (0.0046, 0.0046, 0.0035),
    (0.0074, 0.0070, 0.0062), (0.0082, 0.0082, 0.0082),
    (0.0046, 0.0046, 0.0035), (0.0070, 0.0058, 0.0046),
    (0.0082, 0.0074, 0.0074), (0.0150, 0.0150, 0.0141),
    (0.0161, 0.0161, 0.0161), (0.0161, 0.0161, 0.0161),
    (0.0239, 0.0239, 0.0212), (0.0315, 0.0306, 0.0290),
    (0.0303, 0.0303, 0.0303), (0.0179, 0.0179, 0.0153),
    (0.0232, 0.0206, 0.0179), (0.0259, 0.0241, 0.0241),
    (0.0204, 0.0204, 0.0186), (0.0225, 0.0225, 0.0217),
    (0.0225, 0.0225, 0.0225), (0.0423, 0.0423, 0.0377),
    (0.0585, 0.0571, 0.0537), (0.0502, 0.0502, 0.0502),
    (0.0423, 0.0423, 0.0377), (0.0681, 0.0641, 0.0601),
    (0.0648, 0.0621, 0.0628),
]
_INNER_PAIRS = ((2, 1), (3, 1), (3, 2))


def fourth_order_constant(H1, H2, H3, verbose=False):
    """C4 with ||S4(t) - e^{-itH}|| <= t^5 * C4, from Proposition J.2."""
    Hs = {1: H1, 2: H2, 3: H3}
    total = 0.0
    for col, (l, m) in enumerate(_INNER_PAIRS):
        inner = commutator(Hs[l], Hs[m])
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                for k in (1, 2, 3):
                    row = (i - 1) * 9 + (j - 1) * 3 + (k - 1)
                    c = _TABLE2[row][col]
                    nested = commutator(
                        Hs[i], commutator(Hs[j], commutator(Hs[k], inner)))
                    total += c * one_norm(nested)
                    if verbose:
                        print(f"    [{i},[{j},[{k},[{l},{m}]]]] "
                              f"c={c:.4f} norm={one_norm(nested):.6g}",
                              flush=True)
    return total


def gate_counts(L, r, order, eps_synth_budget, n_bonds):
    """2-qubit gate count, CNOT depth, and T count for r Trotter steps.

    Each bond exponential exp(-i th (XX+YY+ZZ)) is one canonical two-qubit
    gate = 3 CNOTs + 3 single-qubit rotations.  Single-site Z rotations are
    free of CNOTs.  Bonds are scheduled with the 4-colouring of the square
    lattice, so one sweep over all bonds costs 4 layers of parallel gates.
    """
    stages = {1: 1, 2: 2, 4: 10}[order]   # S4 = 5 x S2 = 10 half-sweeps
    sweeps = r * stages
    cnots = sweeps * n_bonds * 3
    depth = sweeps * 4 * 3                       # 4 colours x 3 CNOT each
    n_rot = sweeps * (n_bonds * 3 + L * L)
    eps_rot = eps_synth_budget / n_rot
    t_per_rot = 1.15 * np.log2(1.0 / eps_rot) + 9.2   # Ross-Selinger
    return cnots, depth, n_rot * t_per_rot


def main(L=10, T=100.0, eps=1e-2, periodic=False, do_fourth=True):
    H1, H2, H3, n_bonds = xyz_groups(L, periodic)
    n = L * L
    A1 = first_order_constant(H1, H2, H3)
    C2 = second_order_constant(H1, H2, H3)
    C4 = fourth_order_constant(H1, H2, H3) if do_fourth else None

    print(f"2D Heisenberg {L}x{L}, n={n}, {n_bonds} bonds, "
          f"{'PBC' if periodic else 'OBC'}")
    print(f"  Prop 15 constant A1 (Pauli 1-norm) = {A1:.6g}")
    print(f"  Prop 16 constant C2 (Pauli 1-norm) = {C2:.6g}")
    if C4 is not None:
        print(f"  Prop J.2 constant C4 (Pauli 1-norm) = {C4:.6g}")
    print(f"  T = {T:g}, target eps = {eps:g}   "
          f"(2/3 of budget to Trotter, 1/3 to rotation synthesis)")
    print()

    eps_trot = 2.0 * eps / 3.0
    eps_synth = eps / 3.0

    # r * bound(t/r) <= eps_trot
    r1 = T ** 2 * A1 / (2.0 * eps_trot)
    r2 = np.sqrt(T ** 3 * C2 / eps_trot)

    orders = [(1, r1), (2, r2)]
    if C4 is not None:
        orders.append((4, (T ** 5 * C4 / eps_trot) ** 0.25))

    print(f"{'order':>6} {'r':>12} {'dt':>11} {'2q gates':>12} "
          f"{'CNOT depth':>12} {'T count':>12}")
    for order, r in orders:
        r = int(np.ceil(r))
        c, d, tc = gate_counts(L, r, order, eps_synth, n_bonds)
        print(f"{order:6d} {r:12.4g} {T / r:11.4g} {c:12.4g} {d:12.4g} "
              f"{tc:12.4g}")


if __name__ == "__main__":
    main()
