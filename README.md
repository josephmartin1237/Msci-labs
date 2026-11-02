# Quantum many-body simulations of small spin chains

Exact diagonalisation of spin-1/2 chains for the MSci Quantum Many-Body (QMB) lab.
This repository holds the code and data behind the report
*What small spin rings can and cannot tell us about quantum criticality* (J. Martin, 2026).

The main study is the antiferromagnetic Heisenberg ring

$$H = J\sum_{i=1}^{L} \mathbf{S}_i\cdot\mathbf{S}_{i+1}, \qquad J = 1,\ \mathbf{S}_{L+1}=\mathbf{S}_1,$$

solved with sparse Lanczos (`scipy.sparse.linalg.eigsh`) inside fixed-$S^z_{\rm tot}$ sectors for
rings up to $L = 24$. The transverse-field Ising chain is added as an extension.

## Layout

```
Week 1 - Two spins/              two-spin warm-up (Problems 1-4)
Week 2 - Many spin operators/    dense vs sparse many-spin operators (Problems 5-7)
Heisenberg chain/                everything for the Heisenberg ring (run scripts from here)
    Generators.py                Hamiltonian builders, imported by every script below
    QMB week 3.py                Problem 10: [H, Sz_tot] = 0 check
    QMB week 4.py                Problems 11-12: Sz_tot sector, Neel variational estimate
    Research task 1.py           ground-state energies for L = 3-24 -> energy_density.txt
    Research task 2.py           energy density against 1/L and 1/L^2
    Research task 3.py           extrapolation to L -> infinity, fit windows and error budget
    QMB week 5.py                Problems 13-14, Wrap up 1: gap, correlations, summary figure
    energy_density.txt           saved data: L, E0, E0/L, residual ||Hv - Ev||
    Figures/                     output figures
    operators/                   saved Hamiltonians for L <= 10 (optional, see below)
Ising chain/                     transverse-field Ising extension (to be added)
```

## Running

```bash
pip install -r requirements.txt
cd "Heisenberg chain"
python "Research task 1.py"    # writes energy_density.txt (needed by the scripts below)
python "Research task 2.py"
python "Research task 3.py"
python "QMB week 4.py"
python "QMB week 5.py"
```

All paths are relative, so each script must be run from inside `Heisenberg chain/`.
On a desktop PC the five scripts above take about 3 min in total (`Research task 1.py` ~50 s,
`QMB week 4.py` ~40 s, `QMB week 5.py` ~70 s). Each `plt.show()` pauses until the plot window is closed.
The scripts build the sector Hamiltonians on the fly from `Generators.py`; the saved matrices in
`operators/` are only used by `QMB week 4.py` to report non-zero counts of the full $2^L$ matrix,
and it skips any that are missing.

`python Generators.py` cross-checks every Hamiltonian builder for $L \le 8$ and then regenerates
the full set of saved matrices. Be aware that the full $2^L$ sparse matrices up to $L = 23$ took
about 26 min and ~16 GB of RAM; the sector matrices up to $L = 26$ took about 2.5 min.
Only the $L \le 10$ files are committed because of their size.

## Where each result comes from

| Result | Value | Script |
|---|---|---|
| Ground state lies in the $S^z_{\rm tot}=0$ sector (checked against full $2^L$, $L \le 16$) | agreement $< 10^{-13}$ | `QMB week 4.py` |
| Sector size reduction at $L = 20$ | $5.7\times$ | `QMB week 4.py` |
| Eigenpair residuals $\lVert Hv - Ev\rVert$ | $< 10^{-13}$ for all $L$ | `Research task 1.py`, `QMB week 5.py` |
| Thermodynamic energy density $e_\infty$ (Bethe: $1/4-\ln 2 = -0.443147$) | $-0.44316 \pm 0.00004$ | `Research task 3.py` |
| Gap extrapolation $L\Delta \to \pi^2/2 = 4.9348$ | $4.935$ | `QMB week 5.py` |
| Effective correlation exponent ($\eta = 1$ in theory) | $0.87$ | `QMB week 5.py` |

| Figure | Script |
|---|---|
| `Figures/research_task2_energy_density.png` | `Research task 2.py` |
| `Figures/research_task3_fits.png`, `research_task3_windows.png` | `Research task 3.py` |
| `Figures/problem11_sector.png`, `problem12_neel.png` | `QMB week 4.py` |
| `Figures/problem13_gap.png`, `problem14_correlations.png` | `QMB week 5.py` |
| `Figures/figure_heisenberg_summary.png` | `QMB week 5.py` |
| Report Figs. 1-2 | (final figure script, to be added) |

## Validation

- Every Hamiltonian builder (dense, sparse, and four sector constructions) agrees for $L \le 8$,
  and each sector spectrum lies inside the full spectrum (`Generators.py`).
- $L = 3$ and $4$ reproduce the exact ground-state energies $-0.75$ and $-2$.
