import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eig, eigh_tridiagonal
import numpy as np
from LCAO_on_numerov import poeschl_teller, multiwell_pot, make_space

N_CENTERS = 4
D = 2
N_POINTS = 2000


def make_ham_diagonals(x_space, V):
    h = x_space[1] - x_space[0]
    diag = V + 1 / h**2
    offdiag = -0.5 * np.ones(len(x_space)-1) / h**2
    return  diag, offdiag

if __name__ == "__main__":
    x_space = make_space(n_centers=N_CENTERS, dist_wells=D, n_points=N_POINTS)
    pot = multiwell_pot(poeschl_teller, n_centers=N_CENTERS, d=D, x=x_space)
    diag, offdiag = make_ham_diagonals(x_space=x_space, V=pot)
    E, vec = eigh_tridiagonal(diag, offdiag)
    vec = vec.T
    # for i,vec in enumerate(vec):
    #     plt.plot(x_space, vec+E[i])
    plt.plot(x_space, pot)
    plt.plot(x_space, 4*vec[0])
    # plt.hlines(E, xmin =-3, xmax=3)
    plt.show()
    

