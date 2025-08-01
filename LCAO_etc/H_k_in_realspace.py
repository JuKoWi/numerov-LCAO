import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from realspace_base_periodic import make_potential_unitcell, make_supercell, make_periodic_laplacian
from LCAO_on_numerov import poeschl_teller



def apply_nabla(x_space):
    h = x_space[1] - x_space[0]
    nabla_mat = np.diag(-np.ones(len(x_space)-1), k=-1)
    nabla_mat += np.diag(np.ones(len(x_space)-1), k=1)
    nabla_mat[0,-1] = -1
    nabla_mat[-1,0] = 1
    nabla_mat = nabla_mat * (1/(2*h))
    return nabla_mat 


def make_k_hamiltonian(k_vec, V, x_space):
    v_mat = np.diag(V, k=0)
    lap_mat = make_periodic_laplacian(x_space=x_space)
    k_mat = k_vec * np.identity(len(x_space))
    nabla_mat = apply_nabla(x_space=x_space)
    kin_mat = -0.5 * (lap_mat + 2j * k_mat @ nabla_mat - k_vec**2 * np.identity(len(x_space)))
    h_mat = kin_mat + v_mat
    return sparse.csr_matrix(h_mat)



def scan_k(num_k, a, V, x_space, n_bands):
    k_vals = np.linspace(-np.pi/a, np.pi/a, num_k)
    E_vals = np.zeros((len(k_vals), n_bands))
    for i,k_val in enumerate(k_vals):
        H = make_k_hamiltonian(k_val, V, x_space)
        print(np.allclose(H.toarray(), H.toarray().conj().T))
        print(H.nnz)  # Check matrix size and number of non-zeros
        E, vec = eigsh(H, k=n_bands, which='SA', return_eigenvectors=True)
        E_vals[i] = E
        overlap = np.abs(np.vdot(vec[:,0], vec[:,1])) / (np.linalg.norm(vec[:,0]) * np.linalg.norm(vec[:,1]))
        print("Overlap:", overlap)
    E_vals = E_vals.T
    return k_vals, E_vals

def single_k(V, x_space, n_bands, k=0):
    H = make_k_hamiltonian(k, V, x_space)
    E, vec = eigsh(H, k=n_bands, which='SA')
    vec = np.abs(vec.T)**2
    plt.plot(x_space, V, linestyle="--")
    for i in range(n_bands):
        plt.plot(x_space, vec[i] + E[i])
    plt.show()





if __name__ == "__main__":
    A = 4
    x_u, V_u = make_potential_unitcell(lambda x : poeschl_teller(x, lam=5), n_points=500, a=A)
    x, V = make_supercell(x_u, V_u, n_super=1)

    k_vals, E_vals = scan_k(num_k=100, a=A, V=V, x_space=x, n_bands=4)
    fix, ax = plt.subplots(2, 1)
    for i in range(np.shape(E_vals)[0]):
        ax[0].scatter(k_vals, E_vals[i] + 0.1*i)
    ax[1].plot(x, V)
    plt.show()

    single_k(V,x, n_bands=1)

    