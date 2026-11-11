# -*- coding: utf-8 -*-
"""
Report figure (Fig. 1): finite-size signatures of criticality in the Heisenberg ring
(a) energy density against 1/L^2 with fits, (b) scaled gap against 1/ln L,
(c) staggered correlations against chord distance.
Reads energy_density.txt (Research task 1) and heisenberg_even_L.npz (QMB week 5),
so run those first. Drawn at its printed width (7 inches, full page width).
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

# Variables
bethe_ansatz = 0.25 - np.log(2)
cloizeaux_pearson = np.pi**2/2
odd_colour, even_colour = "#0072b2", "#d55e00"

grey = "0.45"

# Template rules: lettering at least the 10pt main text, symbols at least 2 mm (about 6pt)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.linewidth": 1.2,
                     "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.markersize": 6})

L_all, density = np.loadtxt("energy_density.txt", usecols=(0, 2), unpack=True)
L_all = L_all.astype(int)
data = np.load("heisenberg_even_L.npz")
sizes, gaps = data["sizes"], data["gaps"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)

###############################################################################
# (a) Energy density against 1/L^2, with fits for L >= 8
###############################################################################

x_line = np.linspace(0, 1.0/L_all.min()**2, 200)
for name, parity, colour, marker, offset in [("odd $L$", 1, odd_colour, "o", 0.03),
                                             ("even $L$", 0, even_colour, "s", -0.045)]:
    mask = (L_all % 2 == parity)
    fit = mask & (L_all >= 8)
    # a + b/L^2 + c/L^4, the form used for the final estimate in Research task 3
    p = np.polyfit(1.0/L_all[fit]**2, density[fit], 2)
    ax1.plot(x_line, np.polyval(p, x_line), color=colour, alpha=0.8)
    ax1.plot(1.0/L_all[mask]**2, density[mask], marker, color=colour, mec="white", mew=0.4)
    # Direct label beside each line instead of a legend
    ax1.text(0.06, np.polyval(p, 0.06) + offset, name, color=colour, ha="center")
ax1.axhline(bethe_ansatz, color=grey, ls="--", lw=0.8)
ax1.text(0.112, bethe_ansatz + 0.008, r"$\frac{1}{4}-\ln 2$", color=grey, ha="right", va="bottom")
ax1.set_xlabel(r"$1/L^2$")
ax1.set_ylabel(r"$E_0/L$")
ax1.xaxis.set_major_locator(FixedLocator([0, 0.05, 0.1]))
ax1.yaxis.set_major_locator(FixedLocator([-0.55, -0.45, -0.35, -0.25]))
ax1.set_title("(a)", loc="left")

###############################################################################
# (b) Scaled gap against 1/ln L
###############################################################################

inverse_log = 1.0/np.log(sizes)
window = sizes >= 8
p_log = np.polyfit(inverse_log[window], (sizes*gaps)[window], 2)
p_line = np.polyfit(1.0/sizes[window], gaps[window], 1)    # gap = a + b/L
x_line = np.linspace(0, inverse_log.max(), 200)
x_curve = np.linspace(0.235, inverse_log.max(), 200)
L_curve = np.exp(1.0/x_curve)
ax2.plot(x_line, np.polyval(p_log, x_line), "k-")
ax2.plot(x_curve, L_curve*np.polyval(p_line, 1.0/L_curve), ls="--", color=grey)
ax2.plot(inverse_log, sizes*gaps, "o", color=odd_colour, mec="white", mew=0.4)
ax2.plot(0, cloizeaux_pearson, "*", ms=8, color="k", clip_on=False)
ax2.text(0.03, cloizeaux_pearson, r"$\pi^2/2$", va="center")
ax2.text(0.0, 4.05, r"fit in $1/\ln L$", ha="left", va="center")
ax2.text(0.27, 4.95, r"$\Delta=a+b/L$", color=grey, va="center")
ax2.set_xlabel(r"$1/\ln L$")
ax2.set_ylabel(r"$L\Delta$")
ax2.set_xlim(-0.02, inverse_log.max() + 0.03)
ax2.set_ylim(3.9, 5.05)
ax2.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
ax2.yaxis.set_major_locator(FixedLocator([4.0, 4.4, 4.8]))
ax2.set_title("(b)", loc="left")

###############################################################################
# (c) Staggered correlations against chord distance
###############################################################################

shown = [10, 18, 26]    # three sizes are enough to show the collapse without crowding
colours = ["#0072b2", "#e69f00", "#009e73"]    # Okabe-Ito, colour-blind safe
for L, colour in zip(shown, colours):
    r = np.arange(1, L//2 + 1)
    chord = (L/np.pi)*np.sin(np.pi*r/L)
    ax3.plot(chord, (-1.0)**r * data[f"C_{L}"][1:], "o", color=colour, mec="white", mew=0.4,
             label=f"$L={L}$")

# Power law fit for the largest ring, leaving out r = 1 as in QMB week 5
L = sizes[-1]
r = np.arange(1, L//2 + 1)
chord = (L/np.pi)*np.sin(np.pi*r/L)
staggered = (-1.0)**r * data[f"C_{L}"][1:]
slope, intercept = np.polyfit(np.log(chord[1:]), np.log(staggered[1:]), 1)
d_line = np.linspace(1, chord.max(), 50)
fit_line, = ax3.plot(d_line, np.exp(intercept)*d_line**slope, "k-", zorder=0)
# 1/d prediction, scaled to meet the fit at the largest distance
amplitude = np.exp(intercept)*chord.max()**(slope + 1)
theory_line, = ax3.plot(d_line, amplitude/d_line, ls="--", color=grey, zorder=0)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
ax3.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1]))
for axis in (ax3.xaxis, ax3.yaxis):
    axis.set_major_formatter(FormatStrFormatter("%g"))
    axis.set_minor_formatter(NullFormatter())
ax3.set_xlabel(r"Chord distance $d$")
ax3.set_ylabel(r"$(-1)^r\,C(r)$")
# Two small legends: ring sizes bottom left, the two lines top right
sizes_legend = ax3.legend(loc="lower left", handletextpad=0.1, borderaxespad=0.2)
ax3.add_artist(sizes_legend)
ax3.legend([fit_line, theory_line], [f"slope ${slope:.2f}$", r"$1/d$"], loc="upper right",
           handlelength=1.6, borderaxespad=0.2)
ax3.set_title("(c)", loc="left")

print(f"(b) L*gap(L -> inf) = {p_log[-1]:.4f}, straight line gap intercept a = {p_line[-1]:.4f}")
print(f"(c) slope {slope:.3f}")
fig.savefig("Figures/fig1_heisenberg.pdf")
fig.savefig("Figures/fig1_heisenberg.png", dpi=300)
plt.show()
