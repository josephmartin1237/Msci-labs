# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:00:00 2026

@author: Joe Martin
"""

# -*- coding: utf-8 -*-
"""
QMB Research task 3
Extrapolating the energy density to L -> infinity with straight line fits in 1/L^2,
for several size windows and fit forms, and comparing with the Bethe ansatz
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def fit_window(x, y, degree):
    # Polyfit with the covariance matrix, returns coefficients and the intercept error
    p, cov = np.polyfit(x, y, degree, cov=True)
    return p, np.sqrt(cov[-1, -1])

def significant_figures(value, error):
    # Number of significant figures of value that are bigger than the error
    return int(np.floor(np.log10(abs(value))) - np.floor(np.log10(error)) + 1)

# Variables
data_file = "energy_density.txt"
bethe_ansatz = 0.25 - np.log(2)
min_points = 4    # fewest sizes allowed in a window (polyfit needs > degree + 2 for its error)

sizes, energies, energy_density = np.loadtxt(data_file, usecols=(0, 1, 2), unpack=True)
sizes = sizes.astype(int)
inverse_L2 = 1.0/sizes**2
odd = (sizes % 2 == 1)
even = ~odd

###############################################################################
# Basic fit, all sizes, like the handbook
###############################################################################

p_odd = np.polyfit(inverse_L2[odd], energy_density[odd], 1)
p_even = np.polyfit(inverse_L2[even], energy_density[even], 1)
print("Straight line fits of E0/L against 1/L^2 using every size:")
print(f"Odd L:  gradient {p_odd[0]:.5f}, intercept {p_odd[1]:.7f}")
print(f"Even L: gradient {p_even[0]:.5f}, intercept {p_even[1]:.7f}")
print(f"Bethe ansatz: {bethe_ansatz:.7f}"+"\n")

###############################################################################
# Size windows and fit forms
###############################################################################

# Windows are always "L_min up to the biggest L", dropping the smallest sizes one at a
# time since the small rings have the biggest corrections beyond 1/L^2.
# Fit forms: linear in 1/L^2, and quadratic in 1/L^2 (adds a 1/L^4 correction)
results = {"odd": [], "even": []}
for name, mask in [("odd", odd), ("even", even)]:
    L_values = sizes[mask]
    x_values = inverse_L2[mask]
    y_values = energy_density[mask]
    print(f"{name.capitalize()} L windows:")
    print(f"{'window':>9} {'form':>10} {'intercept':>12} {'fit error':>10} "
          f"{'change from last':>17} {'- Bethe':>10}")
    for degree, form in [(1, "a + b/L^2"), (2, "+ c/L^4")]:
        last_intercept = None
        for start in range(len(L_values) - min_points - degree + 2):
            p, error = fit_window(x_values[start:], y_values[start:], degree)
            change = np.nan if last_intercept is None else p[-1] - last_intercept
            last_intercept = p[-1]
            results[name].append((L_values[start], degree, p, error))
            print(f"{L_values[start]:>3}-{L_values[-1]:<5} {form:>10} {p[-1]:>12.7f} {error:>10.1e} "
                  f"{change:>17.1e} {p[-1]-bethe_ansatz:>10.1e}")
    print()

###############################################################################
# Systematic estimate
###############################################################################

# Using only the windows that start at L >= 8, the smallest rings are clearly not
# in the 1/L^2 regime (see the residuals plot)
print("Final estimates, windows starting at L >= 8:")
estimates = {}
for name in ["odd", "even"]:
    intercepts = np.array([p[-1] for L_min, degree, p, error in results[name] if L_min >= 8])
    errors = np.array([error for L_min, degree, p, error in results[name] if L_min >= 8])
    central = np.median(intercepts)
    systematic = (intercepts.max() - intercepts.min())/2
    statistical = np.max(errors)
    total = np.hypot(systematic, statistical)
    estimates[name] = (central, total)
    print(f"{name.capitalize()} L: {central:.7f} +- {systematic:.1e} (window/form) +- {statistical:.1e} (fit), "
          f"{significant_figures(central, total)} significant figures, "
          f"{abs(central-bethe_ansatz)/total:.1f} sigma from Bethe")

# Odd and even come from opposite sides, so the true value has to be between them
odd_central, odd_error = estimates["odd"]
even_central, even_error = estimates["even"]
combined = 0.5*(odd_central + even_central)
bracket = 0.5*abs(odd_central - even_central)
combined_error = max(bracket, odd_error, even_error)
print(f"\nOdd/even bracket: {combined:.6f} +- {combined_error:.1e}")
print(f"Bethe ansatz:     {bethe_ansatz:.6f}")
print(f"Supported significant figures: {significant_figures(combined, combined_error)}")
print("A straight looking line isn't enough: the fit error only measures the scatter about the "
      "line, not whether the line is the right form. Log corrections (marginal operator in the "
      "Heisenberg chain) and 1/L^4 terms bend the curve at small L, and that only shows up as "
      "the intercept drifting when the window or form is changed."+"\n")

###############################################################################
# Plots
###############################################################################

x_line = np.linspace(0, inverse_L2.max(), 200)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for name, mask, colour in [("odd", odd, "C0"), ("even", even, "C1")]:
    p = np.polyfit(inverse_L2[mask], energy_density[mask], 1)
    ax1.scatter(inverse_L2[mask], energy_density[mask], color=colour, label=f"{name} L")
    ax1.plot(x_line, np.polyval(p, x_line), color=colour, ls="-", lw=1)
    ax2.plot(sizes[mask], energy_density[mask] - np.polyval(p, inverse_L2[mask]), "o-",
             color=colour, label=f"{name} L")
ax1.axhline(bethe_ansatz, color="k", ls="--", label="Bethe ansatz")
ax1.set_xlabel(r"$1/L^2$")
ax1.set_ylabel(r"$E_0/L$")
ax1.legend()
ax1.set_title("Linear fits in 1/L$^2$ (all sizes)")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_xlabel("L")
ax2.set_ylabel("Residual from linear fit")
ax2.legend()
ax2.set_title("Residuals show the curvature the line misses")
fig.tight_layout()
fig.savefig("Figures/research_task3_fits.png", dpi=200)
plt.show()

fig, ax = plt.subplots()
for name, colour, marker in [("odd", "C0", "o"), ("even", "C1", "s")]:
    for degree, fill in [(1, colour), (2, "none")]:
        rows = [(L_min, p[-1], error) for L_min, d, p, error in results[name] if d == degree]
        L_min, intercept, error = np.array(rows).T
        ax.errorbar(L_min, intercept, yerr=error, fmt=marker, mfc=fill, color=colour,
                    label=f"{name} L, {'linear' if degree == 1 else 'quadratic'} in 1/L$^2$")
ax.axhline(bethe_ansatz, color="k", ls="--", label="Bethe ansatz")
ax.set_xlabel(r"Smallest L in the fit window")
ax.set_ylabel(r"Extrapolated $E_0/L$")
ax.legend(fontsize=8)
ax.set_title("Research task 3: intercept against fit window")
fig.tight_layout()
fig.savefig("Figures/research_task3_windows.png", dpi=200)
plt.show()
