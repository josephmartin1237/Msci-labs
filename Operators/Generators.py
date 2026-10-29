# Imports
import os
from time import perf_counter
from math import ceil, floor
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def create_operators_dense(size):
    sx = 0.5*np.array([[0,1],[1,0]])
    sy = 0.5*np.array([[0, -1.0j],[1.0j,0]])
    sz = 0.5*np.array([[1,0],[0,-1]])
    identity = np.identity(2)

    sx_operators = []
    sy_operators = []
    sz_operators = []

    for j in range(size):
        factors_x = [identity]*size
        factors_y = [identity]*size
        factors_z = [identity]*size

        factors_x[j] = sx
        factors_y[j] = sy
        factors_z[j] = sz

        operator_x = factors_x[0]
        operator_y = factors_y[0]
        operator_z = factors_z[0]

        for factor_x in factors_x[1:]:
            operator_x = np.kron(operator_x, factor_x)
        for factor_y in factors_y[1:]:
            operator_y = np.kron(operator_y, factor_y)
        for factor_z in factors_z[1:]:
            operator_z = np.kron(operator_z, factor_z)

        sx_operators.append(operator_x)
        sy_operators.append(operator_y)
        sz_operators.append(operator_z)

    return sx_operators, sy_operators, sz_operators

def create_operators_sparse(size):
    sx = sparse.csr_matrix([[0,0.5],[0.5,0]], dtype=complex)
    sy = sparse.csr_matrix([[0, -0.5j],[0.5j,0]], dtype=complex)
    sz = sparse.csr_matrix([[0.5,0],[0,-0.5]], dtype=complex)
    
    identity = sparse.identity(2, format="csr", dtype=complex)

    sx_operators = []
    sy_operators = []
    sz_operators = []

    for j in range(size):
        factors_x = [identity]*size
        factors_y = [identity]*size
        factors_z = [identity]*size

        factors_x[j] = sx
        factors_y[j] = sy
        factors_z[j] = sz

        operator_x = factors_x[0]
        operator_y = factors_y[0]
        operator_z = factors_z[0]

        for factor_x in factors_x[1:]:
            operator_x = sparse.kron(operator_x, factor_x, format="csr")
        for factor_y in factors_y[1:]:
            operator_y = sparse.kron(operator_y, factor_y, format="csr")
        for factor_z in factors_z[1:]:
            operator_z = sparse.kron(operator_z, factor_z, format="csr")

        sx_operators.append(operator_x)
        sy_operators.append(operator_y)
        sz_operators.append(operator_z)

    return sx_operators, sy_operators, sz_operators

def generate_hamiltonian_sparse(size):
    sx_ops, sy_ops, sz_ops = create_operators_sparse(size)
    
    dimension = 2**size   
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype = complex)

    for i in range(size):
        pair_hamiltonian = (sx_ops[i]@sx_ops[(i+1)%size]+
                            sy_ops[i]@sy_ops[(i+1)%size]+
                            sz_ops[i]@sz_ops[(i+1)%size])
        
        hamiltonian += pair_hamiltonian
    return hamiltonian

def generate_hamiltonian_dense(size):
    sx_ops, sy_ops, sz_ops = create_operators_dense(size)

    dimension = 2**size
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    for i in range(size):
        pair_hamiltonian = (np.dot(sx_ops[i], sx_ops[(i+1)%size])+
                            np.dot(sy_ops[i], sy_ops[(i+1)%size])+
                            np.dot(sz_ops[i], sz_ops[(i+1)%size]))

        hamiltonian+=pair_hamiltonian
    return hamiltonian

def sector_states(n_particles, n_up):
    return [index for index in range(2**n_particles) if index.bit_count()==n_up]

def hamiltonian_middle_dense(n_particles):

    indices = sector_states(n_particles, ceil(n_particles/2))
    states = []

    for index in indices:
        bitstring = format(index, "b")
        bitstring = bitstring.rjust(n_particles, "0")
        states.append(bitstring)

    hamiltonian = np.zeros((len(states), len(states)))
    index_tracker = {state:index for index, state in enumerate(states)}

    for index, state in enumerate(states):
        #print("\r"+f"Calculating row {index+1}/{len(states)}.", end="")
        for spin in range(n_particles):
            spin_1 = state[spin]
            spin_2 = state[(spin+1)%len(state)]
            if spin_1 == spin_2:
                hamiltonian[index, index_tracker[state]]+=0.25
            else:
                hamiltonian[index, index_tracker[state]]-=0.25

                new_state = list(state)
                new_state[spin] = spin_2
                new_state[(spin+1)%n_particles] = spin_1
                new_state = "".join(new_state)

                hamiltonian[index, index_tracker[new_state]]+=0.5

    return hamiltonian

def hamiltonian_middle_sparse(n_particles):

    indices = sector_states(n_particles, ceil(n_particles/2))

    states = []
    for index in indices:
        bitstring = format(index, "b")
        bitstring = bitstring.rjust(n_particles, "0")
        states.append(bitstring)
    
    hamiltonian = sparse.lil_matrix((len(states), len(states)), dtype=float)
    index_tracker = {state:index for index, state in enumerate(states)}

    for index, state in enumerate(states):
        #print("\r"+f"Calculating row {index+1}/{len(states)}.", end="")
        for spin in range(n_particles):
            spin_1 = state[spin]
            spin_2 = state[(spin+1)%n_particles]
            if spin_1 == spin_2:
                hamiltonian[index, index_tracker[state]]+=0.25
            else:
                hamiltonian[index, index_tracker[state]]-=0.25
                
                new_state = list(state)
                new_state[spin] = spin_2
                new_state[(spin+1)%n_particles] = spin_1
                new_state = "".join(new_state)

                hamiltonian[index, index_tracker[new_state]]+=0.5

    return hamiltonian

def hamiltonian_middle_sparse_bit(n_particles):
    
    states = sector_states(n_particles, ceil(n_particles/2))
    hamiltonian = sparse.lil_matrix((len(states), len(states)), dtype=float)
    index_tracker = {state:index for index, state in enumerate(states)}

    for index, state in enumerate(states):
        #print("\r"+f"Calculating row {index+1}/{len(states)}.", end="")
        for spin in range(n_particles):
            bit1 = 2**spin
            bit2 = 2**((spin+1)%n_particles)
            mask = bit1 + bit2
            if (state & mask) == bit1 or (state&mask) == bit2:
                hamiltonian[index, index_tracker[state]]-=0.25
                hamiltonian[index, index_tracker[state ^ mask]]+=0.5
            else:
                hamiltonian[index, index_tracker[state]]+=0.25

    return hamiltonian

def sector_states_fast(n_particles, n_up):
    # Same as sector_states but with numpy, the python loop is too slow past L ~ 24
    all_states = np.arange(2**n_particles, dtype=np.int64)
    return all_states[np.bitwise_count(all_states) == n_up]

def hamiltonian_middle_sparse_fast(n_particles, n_up=None):
    # Same matrix as hamiltonian_middle_sparse_bit but every state is done at once
    # with numpy arrays, so L = 26 takes seconds instead of hours.
    # n_up picks the sector, default is the middle one (Sz = 0 or 1/2)
    if n_up is None:
        n_up = ceil(n_particles/2)
    states = sector_states_fast(n_particles, n_up)
    n_states = len(states)
    diagonal = np.zeros(n_states)
    rows = [np.arange(n_states)]
    columns = [np.arange(n_states)]

    for spin in range(n_particles):
        bit1 = 2**spin
        bit2 = 2**((spin+1)%n_particles)
        mask = bit1 + bit2
        anti_parallel = ((states & mask) == bit1) | ((states & mask) == bit2)
        # Sz Sz term, +1/4 if parallel and -1/4 if anti-parallel
        diagonal += np.where(anti_parallel, -0.25, 0.25)
        # 1/2 (S+S- + S-S+) term, flips the anti-parallel pairs
        flipped = states[anti_parallel] ^ mask
        rows.append(np.nonzero(anti_parallel)[0])
        columns.append(np.searchsorted(states, flipped))

    values = [diagonal] + [np.full(len(row), 0.5) for row in rows[1:]]
    hamiltonian = sparse.coo_matrix((np.concatenate(values),
                                     (np.concatenate(rows), np.concatenate(columns))),
                                    shape=(n_states, n_states))
    # coo -> csr adds up the repeated entries (only happens for L = 2)
    return hamiltonian.tocsr()

def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    if np.all(matrix == np.conj(matrix)):
        matrix = matrix.real
    table = ax.table(cellText=matrix,
                     loc=(0,0), cellLoc="center",
                     cellColours=plt.cm.coolwarm_r(norm(np.abs(matrix))))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cell = table[i, j]
            cell.set_height(1/len(matrix))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

if __name__ == "__main__":
    # Everything below only runs when this file is run directly, so the other
    # scripts can do "from Generators import ..." without regenerating everything

    # Variables
    operator_folder = "operators"
    max_L_sparse = 23    # full 2^L sparse Hamiltonians, L = 23 needs ~15 GB while building
    max_L_dense = 12     # full dense Hamiltonians, L = 12 is a 4096 x 4096 matrix
    max_L_mid = 26       # Sz = 0 (or 1/2) sector Hamiltonians
    check_L = 8          # largest size used to check every builder agrees
    os.makedirs(operator_folder, exist_ok=True)

    ###########################################################################
    # Checking every builder gives the same physics
    ###########################################################################

    print("Checking the generators against each other:")
    for i in range(2, check_L+1):
        h_sparse = generate_hamiltonian_sparse(i)
        h_dense = generate_hamiltonian_dense(i)
        h_mid_dense = hamiltonian_middle_dense(i)
        h_mid_sparse = hamiltonian_middle_sparse(i).toarray()
        h_mid_bit = hamiltonian_middle_sparse_bit(i).toarray()
        h_mid_fast = hamiltonian_middle_sparse_fast(i).toarray()

        full_same = np.allclose(h_sparse.toarray(), h_dense)
        hermitian = np.allclose(h_dense, h_dense.conj().T) and np.allclose(h_mid_fast, h_mid_fast.T)
        mid_same = (np.allclose(h_mid_dense, h_mid_sparse) and np.allclose(h_mid_dense, h_mid_bit)
                    and np.allclose(h_mid_dense, h_mid_fast))

        # The sector spectrum has to be part of the full spectrum and hold the ground state
        full_eigvals = np.linalg.eigh(h_dense)[0]
        mid_eigvals = np.linalg.eigh(h_mid_dense)[0]
        in_spectrum = all(np.any(np.isclose(value, full_eigvals)) for value in mid_eigvals)
        same_ground = np.isclose(full_eigvals[0], mid_eigvals[0])

        all_good = full_same and hermitian and mid_same and in_spectrum and same_ground
        print(f"L = {i}: sparse = dense {full_same}, Hermitian {hermitian}, all 4 sector builders agree {mid_same}, "
              f"sector inside full spectrum {in_spectrum}, same ground state {same_ground}")
        assert all_good, f"The generators disagree for L = {i}"
    print("All the generators agree!"+"\n")

    plot_table(np.round(generate_hamiltonian_dense(2), 2), "Hamiltonian of a two particle ring")
    plt.show()

    ###########################################################################
    # Generating Hamiltonians
    ###########################################################################

    start_sparse = perf_counter()
    for i in range(2, max_L_sparse+1):
        hamiltonian_sparse = generate_hamiltonian_sparse(i)
        sparse.save_npz(os.path.join(operator_folder, f"h_{i}_sparse"), hamiltonian_sparse)
        del hamiltonian_sparse
    end_sparse = perf_counter()
    print("\n"+f"It took {end_sparse-start_sparse:.2f}s to generate all {max_L_sparse-1} sparse matrices."+"\n")

    start_dense = perf_counter()
    for i in range(2, max_L_dense+1):
        hamiltonian_dense = generate_hamiltonian_dense(i)
        np.save(os.path.join(operator_folder, f"h_{i}_dense"), hamiltonian_dense)
    end_dense = perf_counter()
    print(f"It took {end_dense-start_dense:.2f}s to generate all {max_L_dense-1} dense matrices."+"\n")

    # Loading Hamiltonians
    start_sparse_load = perf_counter()
    for i in range(2, max_L_sparse+1):
        sparse.load_npz(os.path.join(operator_folder, f"h_{i}_sparse.npz"))
    end_sparse_load = perf_counter()
    print(f"It took {end_sparse_load-start_sparse_load:.2f}s to load all {max_L_sparse-1} sparse matrices."+"\n")

    start_dense_load = perf_counter()
    for i in range(2, max_L_dense+1):
        np.load(os.path.join(operator_folder, f"h_{i}_dense.npy"))
    end_dense_load = perf_counter()
    print(f"It took {end_dense_load-start_dense_load:.2f}s to load all {max_L_dense-1} dense matrices."+"\n")

    # Sector Hamiltonians, the fast version is used because the lil versions take hours past L ~ 22
    start_mid = perf_counter()
    for i in range(2, max_L_mid+1):
        print(f"Generating the Hamiltonian for a {i}-particle system.")
        hamiltonian_mid = hamiltonian_middle_sparse_fast(i)
        sparse.save_npz(os.path.join(operator_folder, f"h_{i}_mid_sparse"), hamiltonian_mid)
        del hamiltonian_mid
    end_mid = perf_counter()
    print(f"It took {end_mid-start_mid:.2f}s to generate all {max_L_mid-1} sector matrices.")
