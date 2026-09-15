# HHKL + variational circuit compilation

Simulating the 2D Heisenberg model on a 100-qubit (10×10) lattice by
decomposing the time evolution with the Lieb-Robinson-based method of Haah,
Hastings, Kothari and Low (arXiv:1801.03922, "HHKL"), and implementing each
small block with a classically compiled variational circuit instead of a
product formula or qubitization.

The claim we are testing is a **gate-count** claim: that the total cost of
`HHKL decomposition + compiled blocks` is lower than optimized Trotter at the
same total error, where error is measured in **operator (spectral) norm**.

The project plan lives in [`docs/plan_notion.md`](docs/plan_notion.md).

## Layout

| Path | What it is |
|:--|:--|
| `docs/` | Project plan, design notes, decisions |
| `documents/` | Reference papers |
| `src/hamiltonians/` | 2D lattice, bond ordering, 4-colouring, couplings |
| `src/ansatz/` | HVA and Trotter circuit builders |
| `src/bppps/` | Sparse Pauli dynamics engines + BP-PPS gradient trainer |
| `src/classical_bench/` | Exact diagonalisation and exact time evolution |
| `scripts/` | Model construction, validation suite, statevector pilot, depth audit |
| `configs/` | YAML settings for every stage |
| `reference/spin-glass-qml-4x4/` | Measured 4×4 numbers from the predecessor project — **baselines, not new results** |

## Provenance

`src/`, `scripts/`, `configs/` and `reference/` were carried over from
[hwchoi0302/spin-glass-qml](https://github.com/hwchoi0302/spin-glass-qml), a
predecessor project that trained HVA circuits on a 4×4 transverse-field Ising
spin glass with Backpropagating Pauli Propagation. That project's measured
results are kept under `reference/` because they are the starting baseline for
the block-compilation work here, and because several of its findings constrain
what this project may claim. Two worth knowing before reading any number:

- On this lattice **circuit depth is not an independent axis** — HVA, grouped
  S2 and grouped S4 all satisfy `depth ≈ 5 × (2Q gates / 24)`, because all
  three schedule the same 24 bonds with the same 4-colouring. Report 2-qubit
  gate count (or T-count); depth only confirms it.
- The predecessor's variational advantage over Trotter at matched gate count
  was **2.3× at exactly one point** (t=0.5, 72 two-qubit gates), and it
  disappears at higher accuracy. See
  [`reference/spin-glass-qml-4x4/results_4x4.md`](reference/spin-glass-qml-4x4/results_4x4.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
