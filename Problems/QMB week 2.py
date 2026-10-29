# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:00:17 2026

@author: Joe Martin + Maks Selevic 
"""

# -*- coding: utf-8 -*-
"""
QMB Week 2: Problems 5, 6 and 7
Building the spin operators for L spin-1/2 particles (dense and sparse)
"""

#imports
import time
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

# Function definitions
def spin_operators_dense(spin, L):
    # Makes a list of L dense operators, entry j is "spin" acting on site j
    # and the identity on every other site (Eq. 18)
    operators = []
    for j in range(L):
        factors = [identity] * L
        factors[j] = spin
        operator = factors[0]
        for factor in factors[1:]:
            operator = np.kron(operator, factor)
        operators.append(operator)
    return operators

def spin_operators_sparse(spin, L):
    # Same as spin_operators_dense but every matrix is a sparse CSR matrix
    spin_sparse = sparse.csr_matrix(spin)
    identity_sparse = sparse.identity(2, format="csr", dtype=complex)
    operators = []
    for j in range(L):
        factors = [identity_sparse] * L
        factors[j] = spin_sparse
        operator = factors[0]
        for factor in factors[1:]:
            operator = sparse.kron(operator, factor, format="csr")
        operators.append(operator)
    return operators

def commutator(A, B):
    return A @ B - B @ A

def check_commutation(ops, L):
    # Tests [S^a_i, S^b_j] = i delta_ij eps_abc S^c_i for every i, j, a and b
    # ops = [sx_ops, sy_ops, sz_ops]. Returns how many commutators failed
    failures = 0
    for i in range(L):
        for j in range(L):
            for a in range(3):
                for b in range(3):
                    left = commutator(ops[a][i], ops[b][j])
                    right = np.zeros_like(left)
                    if i == j:
                        for c in range(3):
                            right = right + 1.0j*levi_civita[a, b, c]*ops[c][i]
                    if not np.allclose(left, right):
                        failures += 1
    return failures

def dense_hamiltonian_bytes(L):
    # One dense D x D complex matrix, D = 2^L, 16 bytes per complex number
    return 16 * (2**L)**2

def dense_operators_bytes(L):
    # Sx_i, Sy_i, Sz_i for all L sites = 3L dense matrices
    return 3 * L * dense_hamiltonian_bytes(L)

def sparse_operators_bytes(L):
    # Each operator has D non-zeros (one per row). CSR stores the values
    # (16 bytes each), the column indices (4 bytes each) and the row
    # pointers (D+1 entries, 4 bytes each)
    D = 2**L
    one_operator = 16*D + 4*D + 4*(D + 1)
    return 3 * L * one_operator

def sparse_nbytes(operator):
    return operator.data.nbytes + operator.indices.nbytes + operator.indptr.nbytes

def largest_L(memory_gb, memory_function):
    # Largest L for which memory_function(L) (in bytes) fits in memory_gb
    L = 1
    while memory_function(L + 1) <= memory_gb * 1e9:
        L += 1
    return L

def readable_bytes(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if nbytes < 1000:
            return f"{nbytes:.2f} {unit}"
        nbytes = nbytes / 1000
    return f"{nbytes:.2f} EB"

# Variables
# Spin elements in x, y and z directions (complex so every operator is complex)
identity = np.eye(2, dtype=complex)
sx = 0.5*np.array([[0, 1], [1, 0]], dtype=complex)
sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]], dtype=complex)
sz = 0.5*np.array([[1, 0], [0, -1]], dtype=complex)

# Levi-Civita symbol, eps_abc, used in the commutation relations
levi_civita = np.zeros((3, 3, 3))
levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1
levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1

# Limits that stop the memory scans from freezing the computer
safe_limit_gb = 10    # never build a set of operators predicted to be bigger than this
max_L_dense = 12
max_L_sparse = 22
time_limit = 30        # seconds, stop the sparse scan if one build takes longer

###############################################################################
# Problem 5
###############################################################################

L = 4
sx_ops = spin_operators_dense(sx, L)
sy_ops = spin_operators_dense(sy, L)
sz_ops = spin_operators_dense(sz, L)
ops = [sx_ops, sy_ops, sz_ops]

print(f"Each operator is a {sx_ops[0].shape[0]} x {sx_ops[0].shape[1]} matrix (2^{L} = {2**L})\n")

# Hermiticity of every operator
hermitian = True
for a in range(3):
    for j in range(L):
        if not np.allclose(ops[a][j], ops[a][j].conj().T):
            hermitian = False
print(f"Are all the operators Hermitian? {'Yes!' if hermitian else 'No!'}")

# S.S = 3/4 on every site (spin-1/2)
check = True
for j in range(L):
    s_squared = sx_ops[j] @ sx_ops[j] + sy_ops[j] @ sy_ops[j] + sz_ops[j] @ sz_ops[j]
    if not np.allclose(s_squared, 0.75*np.eye(2**L)):
        check = False
print(f"Is S.S = 3/4 on every site? {'Yes!' if check else 'No!'}\n")

# Cyclic commutation relations on the same site
print("Same-site cyclic relations:")
for j in range(L):
    xy = np.allclose(commutator(sx_ops[j], sy_ops[j]), 1.0j*sz_ops[j])
    yz = np.allclose(commutator(sy_ops[j], sz_ops[j]), 1.0j*sx_ops[j])
    zx = np.allclose(commutator(sz_ops[j], sx_ops[j]), 1.0j*sy_ops[j])
    print(f"Site {j}: [Sx,Sy]=iSz {xy}, [Sy,Sz]=iSx {yz}, [Sz,Sx]=iSy {zx}")

# Different sites commute
print(f"\n[Sx_0, Sz_1] = 0? {np.allclose(commutator(sx_ops[0], sz_ops[1]), 0)}")

# Full relation for every pair of sites and every pair of axes
failures = check_commutation(ops, L)
total = L * L * 9
print(f"Full relation checked for {total} commutators, {failures} failed")

###############################################################################
# Problem 6
###############################################################################

# Part 1 predicted memory
table_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
print("Predicted dense memory (one complex number = 16 bytes)")
print(f"{'L':>3} {'D = 2^L':>10} {'dense H':>14} {'all 3L operators':>18}")
for L in table_sizes:
    print(f"{L:>3} {2**L:>10} {readable_bytes(dense_hamiltonian_bytes(L)):>14} "
          f"{readable_bytes(dense_operators_bytes(L)):>18}")

print("\nLargest L that fits in memory (1 GB = 10^9 bytes):")
for memory_gb in [1, 8, 64]:
    L_hamiltonian = largest_L(memory_gb, dense_hamiltonian_bytes)
    L_operators = largest_L(memory_gb, dense_operators_bytes)
    print(f"{memory_gb:>2} GB: dense Hamiltonian L = {L_hamiltonian}, "
          f"all 3L operators L = {L_operators}")

# Part 2 measured memory and time
dense_L = []
dense_predicted = []
dense_measured = []
dense_time = []

print(f"\nMeasured dense scan (safe limit {safe_limit_gb} GB)")
print(f"{'L':>3} {'predicted':>12} {'measured':>12} {'ratio':>6} {'time (s)':>9}")
for L in range(2, max_L_dense + 1):
    predicted = dense_operators_bytes(L)
    if predicted > safe_limit_gb * 1e9:
        print(f"Stopping before L = {L}: predicted {readable_bytes(predicted)} "
              f"is over the safe limit")
        break
    start = time.time()
    dense_ops = (spin_operators_dense(sx, L) + spin_operators_dense(sy, L)
                 + spin_operators_dense(sz, L))
    elapsed = time.time() - start
    measured = sum(op.nbytes for op in dense_ops)
    print(f"{L:>3} {readable_bytes(predicted):>12} {readable_bytes(measured):>12} "
          f"{measured/predicted:>6.3f} {elapsed:>9.4f}")
    dense_L.append(L)
    dense_predicted.append(predicted)
    dense_measured.append(measured)
    dense_time.append(elapsed)
    del dense_ops

# Fitting how fast memory and time grow: log2(y) = slope*L + intercept
memory_slope = np.polyfit(dense_L, np.log2(dense_measured), 1)[0]
time_slope = np.polyfit(dense_L[-4:], np.log2(dense_time[-4:]), 1)[0]
print(f"\nMemory grows like 2^({memory_slope:.2f} L) (theory: 4^L times L)")
print(f"Time grows like 2^({time_slope:.2f} L) over the last four sizes")
print(f"Largest dense L reached on this computer: {dense_L[-1]}")

# Plotting memory and time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(dense_L, np.array(dense_measured)/1e9, "o", label="Measured")
ax1.semilogy(dense_L, np.array(dense_predicted)/1e9, "-", label="Predicted")
ax1.set_xlabel("L")
ax1.set_ylabel("Memory for all 3L operators (GB)")
ax1.legend()
ax2.semilogy(dense_L, dense_time, "o-")
ax2.set_xlabel("L")
ax2.set_ylabel("Construction time (s)")
fig.suptitle("Dense spin operators")
fig.tight_layout()
fig.savefig("problem6_dense.png", dpi=200)
plt.show()

###############################################################################
# Problem 7
###############################################################################

# Part 1 the sparse operators are the same as the dense ones
L = 4
same = True
for spin, name in [(sx, "x"), (sy, "y"), (sz, "z")]:
    dense_check = spin_operators_dense(spin, L)
    sparse_check = spin_operators_sparse(spin, L)
    for j in range(L):
        if not np.allclose(sparse_check[j].toarray(), dense_check[j]):
            same = False
print(f"Do the sparse and dense operators agree for L = {L}? {'Yes!' if same else 'No!'}")

# Part 2 predicted memory
print("\nPredicted sparse memory for all 3L operators (nnz per operator = 2^L)")
for memory_gb in [1, 8, 64]:
    print(f"{memory_gb:>2} GB: L = {largest_L(memory_gb, sparse_operators_bytes)}")

# Part 3 measured memory and time
sparse_L = []
sparse_predicted = []
sparse_measured = []
sparse_time = []

print(f"\nMeasured sparse scan (safe limit {safe_limit_gb} GB, time limit {time_limit} s)")
print(f"{'L':>3} {'nnz/operator':>13} {'predicted':>12} {'measured':>12} "
      f"{'ratio':>6} {'time (s)':>9}")
for L in range(2, max_L_sparse + 1):
    predicted = sparse_operators_bytes(L)
    if predicted > safe_limit_gb * 1e9:
        print(f"Stopping before L = {L}: predicted {readable_bytes(predicted)} "
              f"is over the safe limit")
        break
    start = time.time()
    sparse_ops = (spin_operators_sparse(sx, L) + spin_operators_sparse(sy, L)
                  + spin_operators_sparse(sz, L))
    elapsed = time.time() - start
    measured = sum(sparse_nbytes(op) for op in sparse_ops)
    print(f"{L:>3} {sparse_ops[0].nnz:>13} {readable_bytes(predicted):>12} "
          f"{readable_bytes(measured):>12} {measured/predicted:>6.3f} {elapsed:>9.4f}")
    sparse_L.append(L)
    sparse_predicted.append(predicted)
    sparse_measured.append(measured)
    sparse_time.append(elapsed)
    del sparse_ops
    if elapsed > time_limit:
        print(f"Stopping after L = {L}: the build took longer than {time_limit} s")
        break

memory_slope = np.polyfit(sparse_L, np.log2(sparse_measured), 1)[0]
time_slope = np.polyfit(sparse_L[-4:], np.log2(sparse_time[-4:]), 1)[0]
print(f"\nMemory grows like 2^({memory_slope:.2f} L) (theory: 2^L times L)")
print(f"Time grows like 2^({time_slope:.2f} L) over the last four sizes")
print(f"Largest sparse L reached on this computer: {sparse_L[-1]}")

# Part 4 comparing with dense at the same size, and the cost of one state vector
print("\nDense against sparse for all 3L operators:")
for L in [4, 8, 10]:
    index = sparse_L.index(L)
    dense_index = dense_L.index(L)
    print(f"L = {L:>2}: dense {readable_bytes(dense_measured[dense_index]):>10}, "
          f"sparse {readable_bytes(sparse_measured[index]):>10}, "
          f"saving x{dense_measured[dense_index]/sparse_measured[index]:.0f}")

print("\nOne state vector (2^L complex numbers) still grows exponentially:")
for L in [20, 30, 40, 50]:
    print(f"L = {L}: {readable_bytes(16 * 2**L)}")

# Plotting dense against sparse
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(dense_L, np.array(dense_measured)/1e9, "o-", label="Dense")
ax1.semilogy(sparse_L, np.array(sparse_measured)/1e9, "s-", label="Sparse")
ax1.set_xlabel("L")
ax1.set_ylabel("Memory for all 3L operators (GB)")
ax1.legend()
ax2.semilogy(dense_L, dense_time, "o-", label="Dense")
ax2.semilogy(sparse_L, sparse_time, "s-", label="Sparse")
ax2.set_xlabel("L")
ax2.set_ylabel("Construction time (s)")
ax2.legend()
fig.suptitle("Dense against sparse spin operators")
fig.tight_layout()
fig.savefig("problem7_sparse.png", dpi=200)
plt.show()
    
    














    
   
