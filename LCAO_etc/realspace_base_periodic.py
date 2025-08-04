import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sci_linalg
from LCAO_on_numerov import poeschl_teller, inner_prod
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


def make_potential_unitcell(atom_pot, n_points, a): # a is length of the unit cell, n_points the number of points that will be in each unit cell of the supercell
    n_wells = 30 
    x_space = np.linspace(0, n_wells * a, n_wells * n_points+1) # linspace guarantees even spacing for certain number of points
    V_tot = np.zeros(x_space.shape)
    for i in range(n_wells):
        V_tot += atom_pot(x_space - i * a)
    unit_start = int(n_wells * n_points * 0.5) # first ...
    unit_stop = int(n_points * (n_wells*0.5 + 1)) # ... and last element to cut out
    V_unit = V_tot[unit_start : unit_stop + 1] # second index is inclusive
    V_unit += -np.max(V_unit)
    x_space = x_space[unit_start : unit_stop + 1] - x_space[unit_start]
    return x_space, V_unit # returns arrays where first and last elements correspond to lattice points

def make_supercell(x_space, V_unit, n_super):
    n_points = len(x_space) - 1
    max = x_space[-1]
    long_space = np.linspace(-max*n_super, max*n_super, n_points*n_super*2, endpoint=False)# dont repeat endpoint
    V_unit = V_unit[:-1]
    long_V = np.tile(V_unit, reps=n_super*2)
    return long_space, long_V # returns array where the last point is not symmetry equivalent to first point

# create laplacian matrix (as np)
def make_periodic_laplacian(x_space):
    h = x_space[1] - x_space[0]
    l_diag = -2 * np.ones(len(x_space))
    l_offdiag = np.ones(len(x_space)-1)
    l_mat = np.diag(l_diag, k=0)
    l_mat += np.diag(l_offdiag, k=1) + np.diag(l_offdiag, k=-1)
    # periodic boundary conditions
    l_mat[0, -1] = 1
    l_mat[-1, 0] = 1
    l_mat *= 1/h**2
    return l_mat

# with option sparse
def make_periodic_hamiltonian(x_space, V, is_sparse=True):
    v_mat = np.diag(V, k=0) 
    h_mat = -0.5 * make_periodic_laplacian(x_space=x_space) + v_mat
    if is_sparse:
        h_mat = sparse.csr_matrix(h_mat) 
        return h_mat
    return h_mat

def solve_eigenprob(x_space, V, is_sparse):
    H = make_periodic_hamiltonian(x_space=x_space, V=V, is_sparse=is_sparse)
    if is_sparse:
        E, vec = eigsh(H, k=2, which='SA')
    else:
        E,make_potential_unitcell(poeschl_teller, 500, 3)
        E, vec = eigh(H)
    return E, vec.T

def scan_supercells(N_max, is_sparse, a, lam, n_points):
    x_u, V_u = make_potential_unitcell(lambda x : poeschl_teller(x, lam=lam), n_points=n_points, a=a)
    ground_states = np.zeros((N_max))
    for N in range(1, N_max+1):
        x, V = make_supercell(x_u, V_u, n_super=N)
        H = make_periodic_hamiltonian(x, V)
        E, vec = solve_eigenprob(x, V, is_sparse=is_sparse)
        i = N-1
        ground_states[i] = E[0]
    return ground_states

def cut_unit(x_space, vec, n_super):
    comp = len(vec)
    comp_short = int(comp/n_super)
    unit_vec = vec[:comp_short-1]
    x_unit = x_space[:comp_short-1]
    return x_unit, unit_vec

def norm_ucell(x_space, vec, n_super):
    x_unit, unit_vec = cut_unit(x_space=x_space, vec=vec, n_super=n_super)
    norm_fac = inner_prod(unit_vec, unit_vec, x_unit)
    return norm_fac

    

if __name__ == "__main__":
    start = time.time()
    A = 4
    N_SUPER = 10
    x_u, V_u = make_potential_unitcell(lambda x : poeschl_teller(x, lam=5), n_points=500, a=A)
    x, V = make_supercell(x_u, V_u, n_super=N_SUPER)
    H = make_periodic_hamiltonian(x, V)
    E, vec = solve_eigenprob(x, V, is_sparse=True)
    end = time.time()
    print(end - start)
    # plot for supercell
    plt.plot(x, np.abs(vec[0])**2 / norm_ucell(x_space=x, vec=vec[0], n_super=N_SUPER)**2, label='psi mod square')
    plt.hlines(E[1], x[0], x[-1])
    plt.plot(x, V, label='V')
    plt.legend()
    plt.show()
    
    # plot for unit cell
    plt.plot

    # ground_states = scan_supercells(N_max=10, is_sparse=True, a=2, lam=5, n_points=200)
    # plt.plot(np.arange(1, 11), ground_states)
    # plt.show()
    # print(ground_states)

#write normalization with respect to integral
