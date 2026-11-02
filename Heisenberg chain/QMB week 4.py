# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:30:00 2026

@author: Joe Martin
"""

# -*- coding: utf-8 -*-
"""
QMB Week 4: Problems 11 and 12
Fixed total magnetisation (Sz) sectors and the Neel variational estimate
"""

# Imports
import os
from math import comb, ceil
from time import perf_counter
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from Generators import (generate_hamiltonian_sparse, hamiltonian_middle_sparse_fast,
                        sector_states_fast)

# Function definitions
def ground_energy(hamiltonian, k=1):
    # Lowest k energies with the sparse Lanczos solver (dense eigh for tiny matrices)
    if hamiltonian.shape[0] <= 64:
        return np.linalg.eigh(hamiltonian.toarray())[0][:k]
    eigvals = sparse.linalg.eigsh(hamiltonian, k=k, which="SA")[0]
    return np.sort(eigvals)

def sparse_nbytes(operator):
    return operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes

def readable_bytes(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if nbytes < 1000:
            return f"{nbytes:.2f} {unit}"
        nbytes = nbytes / 1000
    return f"{nbytes:.2f} EB"

def neel_state(L):
    # Alternating up/down spins as an integer, bit j = 1 means site j is up.
    # For odd L the last site has to match the first one, so one bond is parallel
    return sum(2**j for j in range(0, L, 2))

def neel_energy_analytic(L):
    # Each anti-parallel bond gives <S_i.S_j> = -1/4 (from the two spin problem),
    # each parallel bond gives +1/4. Even rings: L anti-parallel bonds.
    # Odd rings: L-1 anti-parallel bonds and 1 parallel bond
    if L % 2 == 0:
        return -L/4
    return -(L-1)/4 + 1/4

def neel_energy_numeric(L):
    # <Neel|H|Neel> is just the diagonal element of the sector Hamiltonian
    n_up = ceil(L/2)
    states = sector_states_fast(L, n_up)
    hamiltonian = hamiltonian_middle_sparse_fast(L, n_up)
    index = np.searchsorted(states, neel_state(L))
    return hamiltonian[index, index]

# Variables
operator_folder = "operators"
max_L_full_check = 16   # largest L where the full 2^L ground state is still quick
max_L_sector = 26

###############################################################################
# Problem 11
###############################################################################

# Part 1 checking the sector ground state is the true ground state
print("Problem 11: is the Sz = 0 (1/2 for odd L) ground state the full ground state?")
print(f"{'L':>3} {'E0 full':>14} {'E0 sector':>14} {'difference':>12}")
for L in range(3, max_L_full_check+1):
    E_full = ground_energy(generate_hamiltonian_sparse(L))[0].real
    E_sector = ground_energy(hamiltonian_middle_sparse_fast(L))[0]
    print(f"{L:>3} {E_full:>14.10f} {E_sector:>14.10f} {abs(E_full-E_sector):>12.1e}")
    assert np.isclose(E_full, E_sector)
print("Yes! The ground state always lives in the middle sector."+"\n")

# Part 2 the reduction in matrix size
sizes = np.arange(2, max_L_sector+1, 2)
full_dimension = 2.0**sizes
sector_dimension = np.array([comb(L, L//2) for L in sizes], dtype=float)
ratio = sector_dimension/full_dimension
stirling = np.sqrt(2/(np.pi*sizes))   # C(L, L/2)/2^L -> sqrt(2/(pi L)) for large L

print("Matrix size reduction for even L:")
print(f"{'L':>3} {'2^L':>10} {'C(L,L/2)':>10} {'ratio':>8} {'sqrt(2/piL)':>12} {'nnz full':>10} {'nnz sector':>11}")
nnz_full = []
nnz_sector = []
memory_sector = []
build_time = []
for L, D, d, r, s in zip(sizes, full_dimension, sector_dimension, ratio, stirling):
    start = perf_counter()
    h_sector = hamiltonian_middle_sparse_fast(L)
    build_time.append(perf_counter()-start)
    nnz_sector.append(h_sector.nnz)
    memory_sector.append(sparse_nbytes(h_sector))
    # Full nnz from the small saved Hamiltonians (Generators.py makes them), the big ones
    # are skipped since loading them just to count takes GBs
    path = os.path.join(operator_folder, f"h_{L}_sparse.npz")
    small_enough = L <= max_L_full_check and os.path.exists(path)
    nnz_full.append(sparse.load_npz(path).nnz if small_enough else np.nan)
    print(f"{L:>3} {D:>10.0f} {d:>10.0f} {r:>8.4f} {s:>12.4f} {nnz_full[-1]:>10.0f} {nnz_sector[-1]:>11}")
    del h_sector

print("\nThe dimension drops by roughly sqrt(2/(pi L)), the dense matrix by the square of that")
print(f"L = 20: {comb(20,10)} instead of {2**20} states, x{2**20/comb(20,10):.1f} smaller")
print(f"L = 26 sector Hamiltonian takes {readable_bytes(memory_sector[-1])} and "
      f"{build_time[-1]:.1f} s to build"+"\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(sizes, full_dimension, "o-", label=r"Full, $2^L$")
ax1.semilogy(sizes, sector_dimension, "s-", label=r"$S^z_{tot}=0$, $\binom{L}{L/2}$")
ax1.set_xlabel("L")
ax1.set_ylabel("Matrix dimension")
ax1.legend()
ax2.plot(sizes, ratio, "o", label="Measured ratio")
ax2.plot(sizes, stirling, "-", label=r"$\sqrt{2/\pi L}$")
ax2.set_xlabel("L")
ax2.set_ylabel("Sector dimension / full dimension")
ax2.legend()
fig.suptitle("Problem 11: size reduction from the fixed $S^z_{tot}$ sector")
fig.tight_layout()
fig.savefig("Figures/problem11_sector.png", dpi=200)
plt.show()

###############################################################################
# Problem 12
###############################################################################

print("Problem 12: Neel state variational estimate")
print(f"{'L':>3} {'E Neel':>8} {'analytic':>9} {'E0 exact':>10} {'E Neel/L':>9} {'E0/L':>9}")
neel_sizes = np.arange(3, 6+1)
neel_densities = []
exact_densities = []
for L in neel_sizes:
    E_neel = neel_energy_numeric(L)
    E_exact = ground_energy(hamiltonian_middle_sparse_fast(L))[0]
    assert np.isclose(E_neel, neel_energy_analytic(L))
    assert E_neel >= E_exact   # variational principle
    neel_densities.append(E_neel/L)
    exact_densities.append(E_exact/L)
    print(f"{L:>3} {E_neel:>8.3f} {neel_energy_analytic(L):>9.3f} {E_exact:>10.5f} "
          f"{E_neel/L:>9.4f} {E_exact/L:>9.4f}")
print(f"Bethe ansatz: E0/L = 1/4 - ln 2 = {0.25-np.log(2):.4f}")
print("Even rings: every bond anti-parallel so E/L = -1/4 exactly.")
print("Odd rings: going round the ring flips the spin L times, an odd number, so you can't "
      "get back to the start - one bond has to be parallel (frustration), E/L = -(L-2)/(4L).")
print("So odd L sits above even L and they approach the limit from different sides -> fit them separately.")

fig, ax = plt.subplots()
ax.plot(neel_sizes, neel_densities, "o-", label="Neel state")
ax.plot(neel_sizes, exact_densities, "s-", label="Exact ground state")
ax.axhline(0.25-np.log(2), color="k", ls="--", label=r"Bethe ansatz $1/4-\ln 2$")
ax.set_xlabel("L")
ax.set_ylabel(r"$E/L$")
ax.legend()
ax.set_title("Problem 12: Neel variational estimate")
fig.tight_layout()
fig.savefig("Figures/problem12_neel.png", dpi=200)
plt.show()
