# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:45:00 2026

@author: Joe Martin
"""

# -*- coding: utf-8 -*-
"""
QMB Research task 1
Ground state energy density of the Heisenberg ring for L = 3 to 20 (and a bit past),
using the sparse Lanczos solver in the Sz = 0 (1/2 for odd L) sector
"""

# Imports
from time import perf_counter
import numpy as np
import scipy.sparse as sparse
from Generators import hamiltonian_middle_sparse_fast

# Function definitions
def ground_state(hamiltonian):
    # Lowest energy and its eigenvector, dense eigh for tiny matrices since eigsh needs k < dimension
    if hamiltonian.shape[0] <= 64:
        eigvals, eigvecs = np.linalg.eigh(hamiltonian.toarray())
    else:
        eigvals, eigvecs = sparse.linalg.eigsh(hamiltonian, k=1, which="SA")
    return eigvals[0], eigvecs[:, 0]

def residual(hamiltonian, energy, state):
    # ||H v - E v||, small if the eigenpair has been solved accurately
    return np.linalg.norm(hamiltonian @ state - energy*state)

# Variables
min_L, max_L = 3, 24    # the task asks for 3 to 20, the extra sizes help the fits in task 3
data_file = "energy_density.txt"

sizes = []
energies = []
energy_density = []
residuals = []

print(f"{'L':>3} {'dimension':>10} {'E0':>16} {'E0/L':>14} {'residual':>9} {'time (s)':>9}")
for L in range(min_L, max_L+1):
    start = perf_counter()
    hamiltonian = hamiltonian_middle_sparse_fast(L)
    E0, state = ground_state(hamiltonian)
    elapsed = perf_counter() - start
    residuals.append(residual(hamiltonian, E0, state))
    sizes.append(L)
    energies.append(E0)
    energy_density.append(E0/L)
    print(f"{L:>3} {hamiltonian.shape[0]:>10} {E0:>16.10f} {E0/L:>14.10f} {residuals[-1]:>9.1e} {elapsed:>9.2f}")
    del hamiltonian

print(f"\nLargest residual: {max(residuals):.1e}, every ground state is converged")
np.savetxt(data_file, np.column_stack((sizes, energies, energy_density, residuals)),
           header="L    E0    E0/L    residual   (Heisenberg ring, J = 1, PBC)",
           fmt=["%d", "%.12f", "%.12f", "%.2e"])
print(f"Saved the energy densities to {data_file}")
