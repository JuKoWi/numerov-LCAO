import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from LCAO_on_numerov import inner_prod
from realspace_base_periodic import make_potential_unitcell, make_supercell
from LCAO_on_numerov import poeschl_teller
import os
from scipy.linalg import eigh




def shifted_function(R, m, a, x):
    psi = np.load("numerov-five.npy")[m]
    xg = np.load("numerov-grid.npy")
    xg = xg + (R+4) * a
    spline = UnivariateSpline(xg, psi, s=0)
    psi_shift = spline(x)
    return psi_shift

class LCAOIntegrals:
    R_max = 20 # fixed cutoff radius for orbital interaction
    m_max = 5 # use 5 atomic orbitals max
    def __init__(self, a, n_points, cached_int):
        self.R_max = 20
        self.m_max = 5
        self.a = a
        self.n_points = n_points #grid points per uni cell
        self.cached_int = cached_int

    def create_potential(self):
        x_u, V_u = make_potential_unitcell(lambda x : poeschl_teller(x, lam=5), n_points=self.n_points, a=self.a)
        self.x_space, self.V = make_supercell(x_u, V_u, n_super=self.R_max+8) # choose supercell big enough for all integrals until R_max
    
    def calc_S_mat(self):
        self.S_mat = np.zeros((self.R_max, self.m_max, self.m_max)) 
        if self.cached_int and os.path.exists("S_mat.npy"):
            self.S_mat = np.load("S_mat.npy")
        else:
            for R in range(self.R_max):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.S_mat[R, m, n] = self._two_center_int(m=m, n=n, R=R, hamilton=False)
            np.save("S_mat.npy", self.S_mat)
    
    def calc_H_mat(self):
        if self.cached_int and os.path.exists("H_mat.npy"):
            self.H_mat = np.load("H_mat.npy")
        else:
            self.H_mat = np.zeros((self.R_max, self.m_max, self.m_max))
            for R in range(self.R_max):
                for m in range(self.m_max):
                    for n in range(self.m_max):
                        self.H_mat[R, m, n] = self._two_center_int(m=m, n=n, R=R, hamilton=True)
            np.save("H_mat.npy", self.H_mat)
        
    
    def _two_center_int(self, m, n, R, hamilton=False): # does not converge to machine precision 0 for large distances
        x = self.x_space
        psi1 = shifted_function(R=0, m=m, a=self.a, x=x) # rewrite to avoid loading same funciton from file in every iteration of for loop
        psi2 = shifted_function(R=R, m=n, a=self.a, x=x) # also use caching
        # plt.plot(self.x_space, psi1)
        # plt.plot(self.x_space, self.V)
        # plt.plot(self.x_space, psi2)
        # plt.show()
        if hamilton:
            psi2 = self._hamiltonian(psi2)
        return inner_prod(psi1, psi2, x) # still non-machine precision value at high distances 

    def _hamiltonian(self, psi):
        h = self.x_space[1] - self.x_space[0]
        laplacian = np.zeros(np.shape(psi))
        psi_pad = np.pad(psi, pad_width=1, mode="constant", constant_values=0)
        laplacian = (psi_pad[:-2] + psi_pad[2:] - 2 * psi_pad[1:-1]) / h**2
        H = - 0.5 * laplacian + self.V * psi
        return H

class Crystal:

    R_max = 20

    def __init__(self, a, n_points, delta_k, cached_int=False):
        self.a = a
        self.n_points = n_points
        self.k_list = np.arange(-np.pi/a, np.pi/a, delta_k)
        self.n_blocks = len(self.k_list)
        self.m_max = np.shape(np.load("numerov-five.npy"))[0]
        self.cached_int = cached_int

    def get_interals(self):
        self.integrals = LCAOIntegrals(a=self.a, n_points=self.n_points, cached_int=self.cached_int)
        self.integrals.create_potential()
        self.integrals.calc_S_mat()
        self.integrals.calc_H_mat()

    def solve_k_blocks(self):
        self.S_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        self.H_blocks = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        self.c_vecs = np.zeros((self.n_blocks, self.m_max, self.m_max), dtype='complex')
        self.eigvals = np.zeros((self.n_blocks, self.m_max))
        for i, k in enumerate(self.k_list):
            S = self._make_k_block(k=k, hamilton=False)
            self.S_blocks[i] = S
            H = self._make_k_block(k=k, hamilton=True)
            self.H_blocks[i] = H 
            E, vec = eigh(H, S) 
            """ for small unit cells the The leading minor of order 5 of B is not positive definite. 
            The factorization of B could not be completed and no eigenvalues or eigenvectors were computed."""
            self.eigvals[i] = E
            self.c_vecs[i] = vec.T
    
    def plot_bands(self):
        fig, ax = plt.subplots(1,1)
        for m in range(self.m_max):
            ax.plot(self.k_list, self.eigvals[:,m], marker='o', markersize=1, linestyle='', label=f'band index: {m}')
        ax.set_xlabel('k')
        ax.set_ylabel('E/E_h')
        ax.legend()
        plt.show()
    
    def _add_phase_integrals(self, k, m, n, hamilton=False):
        if hamilton:
            sum_index = np.shape(self.integrals.H_mat[:,m,n])[0]
            prefactors = [np.exp(1j * k * R) for R in range(sum_index)]
            mat_elem = np.dot(prefactors, self.integrals.H_mat[:,m,n])
        else:
            sum_index = np.shape(self.integrals.S_mat[:,m,n])[0]
            prefactors = [np.exp(1j * k * R) for R in range(sum_index)]
            mat_elem = np.dot(prefactors, self.integrals.S_mat[:,m,n])
        return mat_elem

    def _make_k_block(self, k, hamilton=False):
        mat = np.zeros((self.m_max, self.m_max), dtype='complex')
        for m in range(self.m_max):
            for n in range(self.m_max):
                mat[m, n] = self._add_phase_integrals(k=k, m=m, n=n, hamilton=hamilton)
        return mat




    

if __name__ == "__main__":
    crystal = Crystal(a=3, n_points=100, delta_k=0.01, cached_int=False)
    crystal.get_interals()
    crystal.solve_k_blocks()
    crystal.plot_bands()
