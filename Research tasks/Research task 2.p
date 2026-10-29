# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:50:00 2026

@author: Joe Martin
"""

# -*- coding: utf-8 -*-
"""
QMB Research task 2
Plotting the saved energy densities against 1/L and 1/L^2, odd and even L separately
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Variables
data_file = "energy_density.txt"
bethe_ansatz = 0.25 - np.log(2)

sizes, energies, energy_density = np.loadtxt(data_file, usecols=(0, 1, 2), unpack=True)
sizes = sizes.astype(int)
inverse_L = 1.0/sizes
inverse_L2 = 1.0/sizes**2
odd = (sizes % 2 == 1)
even = ~odd

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for ax, x, label in [(ax1, inverse_L, r"$1/L$"), (ax2, inverse_L2, r"$1/L^2$")]:
    ax.scatter(x[odd], energy_density[odd], label="odd L")
    ax.scatter(x[even], energy_density[even], label="even L")
    ax.axhline(bethe_ansatz, color="k", ls="--", label=r"Bethe ansatz $1/4-\ln 2$")
    ax.set_xlabel(label)
    ax.set_ylabel(r"$E_0/L$")
    ax.legend()
ax1.set_title("Against 1/L (curved)")
ax2.set_title(r"Against 1/L$^2$ (close to straight)")
fig.suptitle("Research task 2: Heisenberg ring energy density")
fig.tight_layout()
fig.savefig("Figures/research_task2_energy_density.png", dpi=200)
plt.show()
