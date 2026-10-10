"""Compare HHKL tilings of the 10x10 lattice, and derive what a variational
block compiler would have to deliver for the whole scheme to beat Trotter.

The overlap size l and the block extent s are independent knobs.  Haah et al.
always take s = 2l, but nothing forces that: l alone fixes the Lieb-Robinson
error, while s fixes how many blocks there are and how big each one is.  That
matters here because the largest block has to fit in GPU memory.

Per direction, forward blocks of extent s are laid down with stride (s - l),
with a backward block of extent l at every junction:

    N_cut = L / (s - l)          cuts per direction per unit time
    N_dir = 2 * L / (s - l)      blocks per direction per unit time

Two rounds of decomposition (y then x) give N_dir_x * N_dir_y blocks per unit
time, the largest of which is s_x by s_y qubits.  Round-1 cuts have boundary
length L; round-2 cuts have boundary length s_y.  Lemma 6 is linear in the
number of boundary terms, so

    error per unit time = (N_cut_y * L + N_dir_y * N_cut_x * s_y) * eps_1(l, t)

with eps_1 the per-boundary-bond Lieb-Robinson error from Haah et al. Fig. 2.
"""

import importlib.util
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tcb = _load("tcb", "11_trotter_commutator_bound.py")
qub = _load("qub", "12_hhkl_qubitization_cost.py")

ALPHA, BETA, GAMMA = 0.175, 7.9, 0.95

# Direct Trotter baselines on the full 10x10 lattice, from
# 11_trotter_commutator_bound.py (Prop 15 / 16 / J.2, Pauli 1-norm relaxation).
DIRECT = {1: dict(r=4.926e9, twoq=2.660e12, t_count=2.094e14),
          2: dict(r=1.498e6, twoq=1.618e9, t_count=1.038e11),
          4: dict(r=6.549e4, twoq=3.537e8, t_count=2.163e10)}

# Measured bound/empirical ratio of the leading Trotter error term on small 2D
# patches (14_trotter_bound_tightness.py).  r scales as looseness^(1/p), so the
# honest baseline is the bound divided by these.
LOOSENESS = {1: 4.1, 2: 6.4, 4: 1500.0}


def empirical_direct(order, L=10, T=100.0, eps=1e-2, n_bonds=180):
    """Direct Trotter baseline with the bound's measured slack removed."""
    d = DIRECT[order]
    r = int(np.ceil(d["r"] / LOOSENESS[order] ** (1.0 / order)))
    twoq, depth, tc = tcb.gate_counts(L, r, order, eps / 3.0, n_bonds)
    return dict(r=r, twoq=twoq, depth=depth, t_count=tc)


_CACHE_FILE = os.path.join(_HERE, ".block_constants.json")
_CACHE = {}
if os.path.exists(_CACHE_FILE):
    import json
    _CACHE = {tuple(int(x) for x in k.split("x")): tuple(v)
              for k, v in json.load(open(_CACHE_FILE)).items()}


def _save_cache():
    import json
    json.dump({f"{k[0]}x{k[1]}": list(v) for k, v in _CACHE.items()},
              open(_CACHE_FILE, "w"))


def block_constants(sx, sy):
    """(bonds, C2, C4) of an sx-by-sy open patch of the Heisenberg model."""
    key = tuple(sorted((sx, sy)))
    if key not in _CACHE:
        H1, H2, H3, bonds = _rect_groups(sx, sy)
        _CACHE[key] = (bonds,
                       tcb.second_order_constant(H1, H2, H3),
                       tcb.fourth_order_constant(H1, H2, H3))
        _save_cache()
    return _CACHE[key]


def _rect_groups(sx, sy, seed=0):
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
    return H1, H2, H3, len(bonds)


def tiling(L, T, eps, l, sx, sy, beta_factor=1.0):
    """Solve for the per-block time t and the total number of block calls."""
    ncx, ncy = L / (sx - l), L / (sy - l)
    ndx, ndy = 2 * ncx, 2 * ncy
    per_unit_blocks = ndx * ndy
    boundary = ncy * L + ndy * ncx * sy      # boundary bonds cut per unit time

    e = l + GAMMA
    beta = BETA * beta_factor
    c = T * boundary * ALPHA * (beta / e) ** e
    t = ((eps / 3.0) / c) ** (1.0 / (e - 1.0))
    m = per_unit_blocks * T / t
    return dict(t=t, m=m, eps_blk=eps / (3 * m), per_unit=per_unit_blocks,
                qubits=sx * sy)


def block_costs(sx, sy, t, eps_blk):
    """Cost of implementing one block call three different ways."""
    bonds, C2, C4 = block_constants(sx, sy)
    # same looseness correction as the direct baseline, so the two are
    # compared on equal footing
    r2 = math.sqrt(t ** 3 * C2 / eps_blk) / LOOSENESS[2] ** 0.5
    r4 = (t ** 5 * C4 / eps_blk) ** 0.25 / LOOSENESS[4] ** 0.25
    # a block circuit cannot be cheaper than one sweep over its bonds
    twoq_s2 = max(1.0, r2) * 2 * bonds * 3
    twoq_s4 = max(1.0, r4) * 10 * bonds * 3
    twoq_min = bonds * 3          # one layer of XX+YY+ZZ on every bond
    M = 3 * bonds + sx * sy
    alpha = 3.0 * bonds + 0.5 * sx * sy
    _, _, t_qsp = qub.t_count_per_block(M, alpha, t, eps_blk)
    return dict(bonds=bonds, r2=r2, r4=r4, twoq_s2=twoq_s2, twoq_s4=twoq_s4,
                twoq_min=twoq_min, t_qsp=t_qsp)


def sweep_configs(L=10, max_block=40):
    out = []
    for l in (2, 3, 4, 5):
        for sx in range(l + 1, L + 1):
            for sy in range(sx, L + 1):
                if sx * sy > max_block:
                    continue
                out.append((f"l={l} {sx}x{sy}", l, sx, sy))
    return out


GPU_LIMIT = 34   # exact complex64 statevector limit on 2 x H200 (141 GB each)


def main(L=10, T=100.0, eps=1e-2, beta_factor=1.0, empirical=True):
    base = empirical_direct(4) if empirical else DIRECT[4]
    kind = "empirical-corrected" if empirical else "bound"
    print(f"10x10 Heisenberg, T={T:g}, eps={eps:g}, "
          f"Lieb-Robinson beta x{beta_factor:g}")
    print(f"direct S4 Trotter baseline ({kind}): r={base['r']:.3g}, "
          f"{base['twoq']:.3g} 2Q gates, {base['t_count']:.3g} T gates\n")

    hdr = (f"{'config':>12} {'q':>4} {'sim?':>5} {'t':>10} {'m':>10} "
           f"{'eps_blk':>10} {'2Q/blk S4':>10} {'2Q budget':>10} "
           f"{'layers':>7} {'compress':>9} {'HHKL+QSP T':>11}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for label, l, sx, sy in sweep_configs(L):
        tl = tiling(L, T, eps, l, sx, sy, beta_factor)
        bc = block_costs(sx, sy, tl["t"], tl["eps_blk"])
        budget = base["twoq"] / tl["m"]          # 2Q gates/block to tie S4
        compress = bc["twoq_s4"] / budget        # how much cheaper HVA must be
        sim = "yes" if tl["qubits"] <= GPU_LIMIT else "NO"
        rows.append((label, tl, bc, budget, compress, sim))
    rows.sort(key=lambda r: r[4])
    for label, tl, bc, budget, compress, sim in rows:
        layers = budget / bc["twoq_min"]
        print(f"{label:>12} {tl['qubits']:4d} {sim:>5} {tl['t']:10.3g} "
              f"{tl['m']:10.3g} {tl['eps_blk']:10.3g} {bc['twoq_s4']:10.3g} "
              f"{budget:10.3g} {layers:7.2f} {compress:8.3g}x "
              f"{tl['m'] * bc['t_qsp']:11.3g}")
    print()
    print("q          = qubits in the largest block")
    print("sim?       = fits an exact complex64 statevector on 2 x H200")
    print("eps_blk    = spectral-norm accuracy each block must reach")
    print("2Q/blk S4  = 2-qubit gates 4th-order Trotter needs inside one block")
    print("2Q budget  = 2-qubit gates per block available if HHKL is to tie")
    print("             direct S4 Trotter on the full lattice")
    print("layers     = that budget expressed in full sweeps over the block's")
    print("             bonds; < 1 means not even one gate per bond")
    print("compress   = (2Q/blk S4) / (2Q budget); the factor by which the")
    print("             variational circuit must beat in-block Trotter")
    return rows


if __name__ == "__main__":
    main(beta_factor=1.0)
    print()
    main(beta_factor=2.0)
