# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:35:09 2026

@author: Joe Martin
"""
# Problem 10: numerical check that [H, Sz_tot] = 0 (analytic proof above).
from scipy import sparse

# ---- single-spin operators ----
X = 0.5 * sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
Y = 0.5 * sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
Z = 0.5 * sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
Splus = sparse.csr_matrix([[0, 1], [0, 0]], dtype=complex)
Sminus = sparse.csr_matrix([[0, 0], [1, 0]], dtype=complex)


def site_operator(op, j, L):
    left = sparse.identity(2 ** j, dtype=complex, format="csr")
    right = sparse.identity(2 ** (L - j - 1), dtype=complex, format="csr")
    return sparse.kron(sparse.kron(left, op, format="csr"), right, format="csr")


def biggest(A):
    return abs(A).max() if A.nnz else 0.0


print("Problem 10: checking [H, Sz_tot] = 0 and Eq. (21)")
print("  L  |[H,Sz_tot]|  |[SxSx,Sz_tot]|  |Eq.21 error|")
for L in range(3, 9):
    Sx = [site_operator(X, j, L) for j in range(L)]
    Sy = [site_operator(Y, j, L) for j in range(L)]
    Sz = [site_operator(Z, j, L) for j in range(L)]
    Sp = [site_operator(Splus, j, L) for j in range(L)]
    Sm = [site_operator(Sminus, j, L) for j in range(L)]

    H = sparse.csr_matrix((2 ** L, 2 ** L), dtype=complex)
    eq21_error = 0.0
    for i in range(L):
        j = (i + 1) % L
        dot = Sx[i] @ Sx[j] + Sy[i] @ Sy[j] + Sz[i] @ Sz[j]
        raising_lowering = Sz[i] @ Sz[j] + 0.5 * (Sp[i] @ Sm[j] + Sm[i] @ Sp[j])  # Eq. (21)
        eq21_error = max(eq21_error, biggest(dot - raising_lowering))
        H = H + dot

    Sz_tot = sum(Sz)
    comm = biggest(H @ Sz_tot - Sz_tot @ H)
    single_term = Sx[0] @ Sx[1]  # one term alone does NOT commute with Sz_tot
    comm_single = biggest(single_term @ Sz_tot - Sz_tot @ single_term)
    assert comm < 1e-12 and eq21_error < 1e-12 and comm_single > 1e-3
    print(f"{L:>3} {comm:>14.1e} {comm_single:>16.3f} {eq21_error:>14.1e}")
print("Conserved: H only connects states with the same Sz_tot.\n")



