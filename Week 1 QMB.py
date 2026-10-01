# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:19:58 2026

@author: zv24041
"""

#imports
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
 
# Function definitions
def check_hermitian(matrix):
    state = lin.ishermitian(matrix)
    if state == True:
        result = "Yes!"
    elif state == False:
        result = "No!"
    print(f"Is the calculated Hamiltonian Hermitian? {result}"+"\n")
 
def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    table = ax.table(cellText=matrix,
                     loc=(0,0), cellLoc="center",
                     cellColours=plt.cm.coolwarm_r(norm(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cell = table[i, j]
            cell.set_height(1/len(matrix))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
 
def check_normalisation(state):
    norm = np.linalg.norm(state)
    return np.isclose(norm, 1)
 
def randomised_state(n):
    coefficients = np.zeros(n, dtype = complex)
    for i in range(n):
        r = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2*np.pi)
        coefficients[i] = r*np.cos(theta)+1.0j*r*np.sin(theta)
    coefficients = coefficients/np.linalg.norm(coefficients)
    return coefficients
 
def rayleigh_quotient(state, hamiltonian):
    bra = state.conj()
    numerator = np.dot(np.dot(bra, hamiltonian), state)
    denominator = np.dot(bra, state)
    return (numerator/denominator).real
 
def perturbed_state(ground_state, perturbation, epsilon):
    # Removing the component of the perturbation along the ground state
    perturbation = perturbation - np.dot(ground_state.conj(), perturbation)*ground_state
    perturbation = perturbation/np.linalg.norm(perturbation)
    # cos/sin mixing keeps the new state exactly normalised
    return np.cos(epsilon)*ground_state + np.sin(epsilon)*perturbation
 
# Variables
# Fixing the random seed so that the results can be reproduced
#np.random.seed(1)
 
# Spin elements in x, y and z directions
sx = 0.5*np.array([[0, 1], [1,0]])
sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]])
sz = 0.5*np.array([[1, 0], [0, -1]])
 
# Trial States
trial_singlet = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
trial_triplet_zero = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
trial_triplet_uu = np.array([1, 0, 0, 0])
trial_neel = np.array([0, 1, 0, 0])
trial_product = np.kron(randomised_state(2), randomised_state(2))
trial_random = randomised_state(4)
 
trial_states = {"Singlet (ground) state": trial_singlet,
                "Triplet (m=0) state": trial_triplet_zero,
                "Pure up/up product state": trial_triplet_uu,
                "Pure up/down (Neel) product state": trial_neel,
                "Randomised product state": trial_product,
                "Randomised (general) state": trial_random}
 
###############################################################################
# Problem 2
###############################################################################
 
# Tensor products of the spin elemets with themselves
sx_sx = np.kron(sx, sx)
sy_sy = np.kron(sy, sy)
sz_sz = np.kron(sz, sz)
 
# Calculating the Hamiltonian using equation 14 (J=1)
hamiltonian = np.sum((sx_sx,sy_sy, sz_sz), axis=0)
 
# Checking for hermiticity
check_hermitian(hamiltonian)
 
# Creating a table showing the Hamiltonian
plot_table(hamiltonian.real, "The Hamiltonian")
 
###############################################################################
# Problem 3
###############################################################################
 
# Extracting the eigenvalues and the eigenvectors:
eigvals, eigvecs = np.linalg.eigh(hamiltonian)
 
print(f"The eigenvalues of the Hamiltonian are {eigvals}")
 
# Plotting the table of eigenvectors for easier visualisation
plot_table(np.round(eigvecs.real, 4), "Eigenvectors of the Hamiltonian")
 
###############
# Problem 4
##############

#Part 1 rayleigh coefficients of the trial states

exact_energy = eigvals[0]
ground_state = eigvecs[:,0]

print("\n Checking if the trial state is normalised")

for name, state in trial_states.items():
    result = "Yes states are normalised" if check_normalisation(state) == True else "No states are not normalised"
    print(f"{name}: {result}")
    
print(f"The exact ground state energy is {exact_energy}")

print("The Rayleigh coefficients of the trial states are:")

for name, state in trial_states.items():
    energy = rayleigh_quotient(state, hamiltonian)
    bound = "Yes!" if energy >= exact_energy - 1e-12 else "No!"
    print(f"{energy} for the {name}. Variational bound satisfied? {bound}")

#Part 2 perturbing the exact ground state
perturbation = randomised_state(4)
epsilons = np.logspace(-5, -1, 25)
# Only fitting the small-epsilon region where sin(eps) ~ eps


energy_differences = np.zeros(len(epsilons))
small = epsilons < 0.1
exponent, intercept = np.polyfit(np.log(epsilons[small]), np.log(energy_differences[small]), 1)
prefactor = np.exp(intercept)

for i in range(len(epsilons)):
 state = perturbed_state(ground_state, perturbation, epsilons[i])
 energy_differences[i] = rayleigh_quotient(state, hamiltonian) - exact_energy

# Fitting a power law, log(dE) = exponent*log(epsilon) + log(prefactor)
exponent, intercept = np.polyfit(np.log(epsilons), np.log(energy_differences), 1)
prefactor = np.exp(intercept)
 
# Predicted prefactor from <r|H|r> - E0 for the (orthogonalised) perturbation
orthogonal_perturbation = perturbed_state(ground_state, perturbation, np.pi/2)
predicted_prefactor = rayleigh_quotient(orthogonal_perturbation, hamiltonian) - exact_energy
 
print(f"""
Fitted power law exponent: {exponent: .4f} (predicted: 2)
Fitted prefactor: {prefactor: .4f} (predicted: {predicted_prefactor: .4f})
""")


#plotting the log log graph
fig, ax = plt.subplots()
ax.loglog(epsilons, energy_differences, "o", label="Numerical")
ax.loglog(epsilons, predicted_prefactor*epsilons**2, "-", label=r"Prediction $\propto\varepsilon^2$")
ax.set_xlabel(r"Perturbation angle $\varepsilon$")
ax.set_ylabel(r"$E[\phi]-E_0$")
ax.set_title("Error in the Rayleigh quotient against perturbation amplitude")
ax.legend()
fig.savefig("problem4_loglog.png", dpi=200)
plt.show()
