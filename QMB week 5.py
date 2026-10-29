# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:15:00 2026

@author: Joe Martin
"""

# -*- coding: utf-8 -*-
"""
QMB Week 5: Problems 13, 14 and Wrap up task 1
Gap scaling and ground state spin correlations of the Heisenberg ring (even L)
"""

# Imports
import os
from time import perf_counter
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from Generators import hamiltonian_middle_sparse_fast, sector_states_fast

# Function definitions
def lowest_states(hamiltonian, k):
    # Lowest k eigenpairs sorted, dense eigh for tiny matrices since eigsh needs k < dimension
    if hamiltonian.shape[0] <= 64:
        eigvals, eigvecs = np.linalg.eigh(hamiltonian.toarray())
        return eigvals[:k], eigvecs[:, :k]
    eigvals, eigvecs = sparse.linalg.eigsh(hamiltonian, k=k, which="SA")
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]

def count_degenerate(eigvals, value, tolerance=1e-8):
    return int(np.sum(np.abs(eigvals - value) < tolerance))

def zz_correlation(state, L, n_up):
    # C(r) = <S^z_0 S^z_r>, averaged over every starting site j since the ring is
    # translation invariant (less rounding noise than using site 0 only)
    basis = sector_states_fast(L, n_up)
    probability = np.abs(state)**2
    spins = np.array([((basis >> j) & 1) - 0.5 for j in range(L)])   # S^z_j for every basis state
    correlation = np.zeros(L//2 + 1)
    for r in range(L//2 + 1):
        for j in range(L):
            correlation[r] += np.sum(probability * spins[j] * spins[(j+r) % L])
    return correlation/L

def dot_correlation(state, L, n_up, r):
    # <S_0 . S_r> = <S^z_0 S^z_r> + 1/2 <S^+_0 S^-_r + S^-_0 S^+_r>, the flip part swaps
    # anti-parallel spins 0 and r so it pairs amplitude n with amplitude n ^ mask
    basis = sector_states_fast(L, n_up)
    mask = 1 + 2**r
    zz = np.sum(np.abs(state)**2 * (((basis & 1) - 0.5) * (((basis >> r) & 1) - 0.5)))
    anti_parallel = ((basis & mask) == 1) | ((basis & mask) == 2**r)
    flipped = np.searchsorted(basis, basis[anti_parallel] ^ mask)
    flip = 0.5*np.sum(np.conj(state[anti_parallel]) * state[flipped])
    return zz + flip.real

def residual(hamiltonian, energy, state):
    # ||H v - E v||, small if the eigenpair has been solved accurately
    return np.linalg.norm(hamiltonian @ state - energy*state)

# Variables
min_L, max_L = 4, 24     # the problems ask for 4 <= L <= 20, 22 and 24 are cheap with the sector code
data_file = "energy_density.txt"
bethe_ansatz = 0.25 - np.log(2)
cloizeaux_pearson = np.pi**2/2   # L*gap -> pi v with spin wave velocity v = pi/2
max_L_dot_check = 16     # largest L for the full <S_0 . S_r> = 3 C(r) check

###############################################################################
# Problem 13
###############################################################################

sizes = np.arange(min_L, max_L+1, 2)
E0_values = []
E1_values = []
gaps = []
residuals = []
correlations = {}

print("Problem 13: ground state and first excited state, even L")
print(f"{'L':>3} {'E0':>14} {'E1':>14} {'gap':>10} {'L*gap':>8} {'E0 deg':>7} {'E1 deg (Sz=0)':>14} "
      f"{'E1 = Sz=1 ground?':>18} {'residual':>9} {'time (s)':>9}")
for L in sizes:
    start = perf_counter()
    # Sz = 0 sector holds the singlet ground state and the Sz = 0 member of every triplet
    hamiltonian = hamiltonian_middle_sparse_fast(L)
    eigvals, eigvecs = lowest_states(hamiltonian, k=4)
    E1_index = count_degenerate(eigvals, eigvals[0])
    E0, E1 = eigvals[0], eigvals[E1_index]
    # Largest residual of the two eigenpairs used
    residuals.append(max(residual(hamiltonian, E0, eigvecs[:, 0]),
                         residual(hamiltonian, E1, eigvecs[:, E1_index])))
    del hamiltonian
    # Sz = 1 sector has no singlets, so its lowest state being E1 means E1 is a triplet (S = 1)
    E_sz1 = lowest_states(hamiltonian_middle_sparse_fast(L, L//2 + 1), k=1)[0][0]
    triplet = np.isclose(E_sz1, E1)
    correlations[L] = zz_correlation(eigvecs[:, 0], L, L//2)
    elapsed = perf_counter() - start

    E0_values.append(E0)
    E1_values.append(E1)
    gaps.append(E1 - E0)
    print(f"{L:>3} {E0:>14.9f} {E1:>14.9f} {E1-E0:>10.6f} {L*(E1-E0):>8.4f} "
          f"{count_degenerate(eigvals, E0):>7} {count_degenerate(eigvals, E1):>14} "
          f"{'Yes!' if triplet else 'No!':>18} {residuals[-1]:>9.1e} {elapsed:>9.2f}")
gaps = np.array(gaps)
inverse_L = 1.0/sizes

print("\nE0 is a unique singlet for every even L and E1 is the S = 1 triplet (3-fold, Sz = -1, 0, 1),")
print("so the gap being tracked is the singlet-triplet (spin) gap.")

# Fits: gap = a + b/L and gap = a + b/L + c/L^2, the windows drop the smallest L
print(f"\n{'window':>9} {'form':>14} {'gap(L->inf)':>12} {'fit error':>10} {'slope b':>9}")
gap_intercepts = []
for degree, form in [(1, "a + b/L"), (2, "+ c/L^2")]:
    for start in range(0, len(sizes) - degree - 3):
        p, cov = np.polyfit(inverse_L[start:], gaps[start:], degree, cov=True)
        gap_intercepts.append(p[-1])
        print(f"{sizes[start]:>3}-{sizes[-1]:<5} {form:>14} {p[-1]:>12.5f} {np.sqrt(cov[-1,-1]):>10.1e} "
              f"{p[-2]:>9.4f}")
gap_intercepts = np.array(gap_intercepts)
print(f"Extrapolated gap ranges from {gap_intercepts.min():.4f} to {gap_intercepts.max():.4f}, "
      f"compared with a gap of {gaps[-1]:.4f} at L = {sizes[-1]}")
print(f"L*gap at L = {sizes[-1]} is {sizes[-1]*gaps[-1]:.4f}, still creeping towards pi^2/2 = {cloizeaux_pearson:.4f}"
      f" (log corrections)")

# Log-log slope, a gapless (critical) chain should have gap ~ L^-1
slope, intercept = np.polyfit(np.log(sizes[2:]), np.log(gaps[2:]), 1)
print(f"Log-log fit (L >= {sizes[2]}): gap ~ L^({slope:.3f}), gapless theory gives L^-1")

# Log corrections: L*gap = (pi^2/2)(1 - 1/(2 ln L) + ...), so L*gap is a polynomial in 1/ln L
# whose intercept is pi v. This tests the gapless prediction quantitatively
inverse_log = 1.0/np.log(sizes)
print(f"\n{'window':>9} {'form in 1/ln L':>15} {'L*gap(L->inf)':>14} {'fit error':>10} {'v = intercept/pi':>17}")
log_fits = {}
for L_min in [8, 12]:
    window = sizes >= L_min
    for degree in [1, 2]:
        p, cov = np.polyfit(inverse_log[window], (sizes*gaps)[window], degree, cov=True)
        log_fits[(L_min, degree)] = p
        print(f"{L_min:>3}-{sizes[-1]:<5} {'linear' if degree == 1 else 'quadratic':>15} {p[-1]:>14.4f} "
              f"{np.sqrt(cov[-1,-1]):>10.1e} {p[-1]/np.pi:>17.4f}")
print(f"Theory: L*gap -> pi^2/2 = {cloizeaux_pearson:.4f}, v = pi/2 = {np.pi/2:.4f}")
print(f"Largest residual: {max(residuals):.1e}"+"\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x_line = np.linspace(0, inverse_L.max(), 200)
p_linear = np.polyfit(inverse_L[2:], gaps[2:], 1)
p_quadratic = np.polyfit(inverse_L[2:], gaps[2:], 2)
ax1.scatter(inverse_L, gaps, color="C0", label="Exact diagonalisation")
ax1.plot(x_line, np.polyval(p_linear, x_line), "C1-", label=f"a + b/L, a = {p_linear[-1]:.3f}")
ax1.plot(x_line, np.polyval(p_quadratic, x_line), "C2--", label=f"+ c/L$^2$, a = {p_quadratic[-1]:.3f}")
ax1.plot(x_line, cloizeaux_pearson*x_line, "k:", label=r"$\pi^2/2L$")
ax1.set_xlabel(r"$1/L$")
ax1.set_ylabel(r"$\Delta = E_1 - E_0$")
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=8)
x_log = np.linspace(0, inverse_log.max(), 200)
ax2.scatter(inverse_log, sizes*gaps, color="C0", label="Exact diagonalisation")
ax2.plot(x_log, np.polyval(log_fits[(8, 2)], x_log), "C2--", label="Quadratic in 1/ln L, L >= 8")
ax2.plot(0, cloizeaux_pearson, "k*", ms=12, label=r"$\pi^2/2$ (theory)")
ax2.set_xlabel(r"$1/\ln L$")
ax2.set_ylabel(r"$L\Delta$")
ax2.set_xlim(left=-0.02)
ax2.legend(fontsize=8)
fig.suptitle("Problem 13: singlet-triplet gap of the Heisenberg ring")
fig.tight_layout()
fig.savefig("Figures/problem13_gap.png", dpi=200)
plt.show()

###############################################################################
# Problem 14
###############################################################################

# Neel prediction: S^z_0 S^z_r is +1/4 if r even (same spin) and -1/4 if r odd,
# so C(r) = (-1)^r/4 with no decay at all
print("Problem 14: C(r) = <S^z_0 S^z_r> in the ground state (Neel predicts (-1)^r/4)")
for L in sizes:
    signs_match = np.all(np.sign(correlations[L]) == (-1.0)**np.arange(L//2 + 1))
    values = " ".join(f"{c:+.4f}" for c in correlations[L])
    print(f"L = {L:>2}: Neel signs {'Yes!' if signs_match else 'No!'}  C(r) = {values}")

# Checking the SU(2) symmetry of the singlet, <S_0 . S_r> = 3 C(r), for the smaller rings
print(f"\nChecking <S_0 . S_r> = 3 C(r) (singlet is rotationally invariant), L <= {max_L_dot_check}:")
worst = 0.0
for L in sizes[sizes <= max_L_dot_check]:
    state = lowest_states(hamiltonian_middle_sparse_fast(L), k=1)[1][:, 0]
    for r in range(1, L//2 + 1):
        worst = max(worst, abs(dot_correlation(state, L, L//2, r) - 3*correlations[L][r]))
print(f"Largest difference: {worst:.1e} {'Yes!' if worst < 1e-8 else 'No!'}")
print(f"<S_0 . S_1> = 3 C(1) = {3*correlations[sizes[-1]][1]:.5f} and E0/L = {E0_values[-1]/sizes[-1]:.5f} "
      f"at L = {sizes[-1]} (should be equal, every bond is the same)")

# Longest distance r = L/2 against L, Neel would stay at 1/4
staggered_far = np.array([(-1)**(L//2) * correlations[L][L//2] for L in sizes])
slope_far, intercept_far = np.polyfit(np.log(sizes[2:]), np.log(staggered_far[2:]), 1)
print(f"\n(-1)^(L/2) C(L/2) ~ L^({slope_far:.3f}), the critical chain gives ~ 1/r (eta = 1) up to log corrections")

# Log correction test: C(L/2) ~ sqrt(ln L)/L, so L |C(L/2)| should grow like sqrt(ln L) and not be flat.
# L/2 even and L/2 odd are kept apart since the smooth 1/r^2 part adds to one and subtracts from the other
print("L |C(L/2)| against sqrt(ln L) (flat would mean a pure 1/r):")
for name, group in [("L/2 even", sizes % 4 == 0), ("L/2 odd", sizes % 4 == 2)]:
    group = group & (sizes >= 8)
    p = np.polyfit(np.sqrt(np.log(sizes[group])), (sizes*staggered_far)[group], 1)
    values = " ".join(f"{l}: {v:.4f}" for l, v in zip(sizes[group], (sizes*staggered_far)[group]))
    print(f"{name:>9}: {values}  -> slope {p[0]:.3f}")

# Distance dependence for the biggest ring, using the chord distance to undo the ring finite size effect
L = sizes[-1]
r = np.arange(1, L//2 + 1)
chord = (L/np.pi)*np.sin(np.pi*r/L)
staggered = (-1.0)**r * correlations[L][1:]
slope_r, intercept_r = np.polyfit(np.log(chord[1:]), np.log(staggered[1:]), 1)
print(f"L = {L}: (-1)^r C(r) ~ chord^({slope_r:.3f}), magnitude falls from {staggered[0]:.4f} at r = 1 "
      f"to {staggered[-1]:.4f} at r = {L//2}, Neel would be 0.25 for all r"+"\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for L in sizes[::2]:
    ax1.plot(np.arange(L//2 + 1), correlations[L], "o-", ms=4, label=f"L = {L}")
ax1.plot(np.arange(sizes[-1]//2 + 1), 0.25*(-1.0)**np.arange(sizes[-1]//2 + 1), "k:", label="Neel")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_xlabel("r")
ax1.set_ylabel(r"$C(r) = \langle S^z_0 S^z_r\rangle$")
ax1.legend(fontsize=8, ncol=2)
for L in sizes[1::2]:
    r = np.arange(1, L//2 + 1)
    chord = (L/np.pi)*np.sin(np.pi*r/L)
    ax2.loglog(chord, (-1.0)**r * correlations[L][1:], "o", ms=4, label=f"L = {L}")
x_line = np.linspace(1, sizes[-1]/np.pi, 50)
ax2.loglog(x_line, np.exp(intercept_r)*x_line**slope_r, "k--", label=f"slope {slope_r:.2f}")
ax2.set_xlabel(r"Chord distance $(L/\pi)\sin(\pi r/L)$")
ax2.set_ylabel(r"$(-1)^r C(r)$")
ax2.legend(fontsize=8)
fig.suptitle("Problem 14: ground state spin correlations")
fig.tight_layout()
fig.savefig("Figures/problem14_correlations.png", dpi=200)
plt.show()

###############################################################################
# Wrap up task 1
###############################################################################

print("Wrap up task 1: does the data tell the story of a gapless antiferromagnet?")
if os.path.exists(data_file):
    L_all, E_all, density_all = np.loadtxt(data_file, usecols=(0, 1, 2), unpack=True)
    even_mask = (L_all % 2 == 0) & (L_all >= 8)
    odd_mask = (L_all % 2 == 1) & (L_all >= 9)
    even_limit = np.polyfit(1/L_all[even_mask]**2, density_all[even_mask], 1)[-1]
    odd_limit = np.polyfit(1/L_all[odd_mask]**2, density_all[odd_mask], 1)[-1]
    print(f"Energy density: even L -> {even_limit:.5f}, odd L -> {odd_limit:.5f}, "
          f"Bethe ansatz {bethe_ansatz:.5f}")
print(f"Gap: falls like L^({slope:.2f}) and extrapolates to {gap_intercepts.min():.3f} - "
      f"{gap_intercepts.max():.3f} depending on the fit, so consistent with 0 but not proven")
print(f"With the log correction L*gap -> {log_fits[(8, 2)][-1]:.3f} against pi^2/2 = {cloizeaux_pearson:.3f}, "
      "so the gap closes exactly as the gapless theory predicts")
print(f"Correlations: Neel signs everywhere (antiferromagnetic), but |C| decays like r^({slope_r:.2f}), "
      "so no long range Neel order")
print("Best supported: the energy density (odd and even bracket the Bethe value) and the antiferromagnetic")
print("sign pattern. The gap closing is well supported once the log correction is included, but a plain 1/L")
print("fit only bounds it. Least secure: the correlation exponent, the two L/2 sub-sequences still drift")
print("and the sqrt(ln r) correction can't be pinned down over r <= 12.")

###############################################################################
# Report figure (Heisenberg chain: intrinsic gaplessness)
###############################################################################

# Three panel summary figure for the report: energy density extrapolation, gap with log
# corrections, and the correlation power law
if os.path.exists(data_file):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    x2 = 1/L_all**2
    x_line = np.linspace(0, x2.max(), 200)
    for name, mask, colour in [("odd L", (L_all % 2 == 1), "C0"), ("even L", (L_all % 2 == 0), "C1")]:
        fit_mask = mask & (L_all >= 8)
        p = np.polyfit(x2[fit_mask], density_all[fit_mask], 2)
        ax1.scatter(x2[mask], density_all[mask], color=colour, s=18, label=name)
        ax1.plot(x_line, np.polyval(p, x_line), color=colour, lw=1)
    ax1.axhline(bethe_ansatz, color="k", ls="--", lw=1, label=r"Bethe $1/4-\ln 2$")
    ax1.set_xlabel(r"$1/L^2$")
    ax1.set_ylabel(r"$E_0/L$")
    ax1.set_title("(a) Energy density")
    ax1.legend(fontsize=8)

    ax2.scatter(inverse_log, sizes*gaps, color="C0", s=18, label="even L")
    ax2.plot(x_log, np.polyval(log_fits[(8, 2)], x_log), "C2--", lw=1, label="fit in 1/ln L")
    ax2.plot(0, cloizeaux_pearson, "k*", ms=11, label=r"$\pi^2/2$")
    ax2.set_xlabel(r"$1/\ln L$")
    ax2.set_ylabel(r"$L\Delta$")
    ax2.set_xlim(left=-0.02)
    ax2.set_title("(b) Gap closes as 1/L")
    ax2.legend(fontsize=8)

    for L in sizes[1::2]:
        r = np.arange(1, L//2 + 1)
        chord = (L/np.pi)*np.sin(np.pi*r/L)
        ax3.loglog(chord, (-1.0)**r * correlations[L][1:], "o", ms=3.5, label=f"L = {L}")
    x_line_r = np.linspace(1, sizes[-1]/np.pi, 50)
    ax3.loglog(x_line_r, np.exp(intercept_r)*x_line_r**slope_r, "k--", lw=1, label=f"slope {slope_r:.2f}")
    ax3.set_xlabel(r"$(L/\pi)\sin(\pi r/L)$")
    ax3.set_ylabel(r"$(-1)^r C(r)$")
    ax3.set_title("(c) Power law correlations")
    ax3.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("Figures/figure_heisenberg_summary.png", dpi=200)
    plt.show()
