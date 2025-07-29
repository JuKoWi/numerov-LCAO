import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eigh
import numpy as np


#claculation parameters
D = 2
N_SINGLE = 200
N_CENTERS = 30   
N_STATES = 5  # number of atomic states used for every center


#set up space and potential
def make_space(n_centers, dist_wells, n_points):
    x_max = n_centers * dist_wells * 0.5 + 3 
    x_space = np.linspace(-x_max, x_max, n_points)
    return x_space


def poeschl_teller(xs, lam=5, a=1):
    return -lam * (lam + 1) * a**2 / (2 * np.cosh(a * xs) ** 2)

def multiwell_pot(pot_funct, n_centers, d, x):
    V_tot = np.zeros((len(x)))
    gerade = (n_centers%2 == 0)
    if gerade:
        for i in range(int(n_centers/2)):
            V_tot += pot_funct(x - (i+0.5)*d)
            V_tot += pot_funct(x + (i+0.5)*d)
    if not gerade:
        V_tot += pot_funct(x)
        for i in range(1,int(n_centers//2+1)):
            V_tot += pot_funct(x + i*d)
            V_tot += pot_funct(x - i*d)
    return V_tot

def half_multiwell(pot_funct, n_centers, d, x_half):
    V_tot = np.zeros((len(x_half)))
    gerade = (n_centers%2 == 0)
    if gerade:
        for i in range(int(n_centers/2)):
            V_tot += pot_funct(x_half + (i+0.5)*d)
    if not gerade:
        V_tot += pot_funct(x_half)
        for i in range(1, int(n_centers//2+1)):
            V_tot += pot_funct(x_half + i*d)
    return V_tot




def move_center(psi, d, x, xg):
    xg = xg + d
    spline = UnivariateSpline(xg, psi, s=0)
    psi_shift = spline(x)
    return psi_shift



# plt.plot(x_space, phi_ao[0])
# plt.plot(x_space, phi_ao[1])
# plt.plot(x_space, phi_ao[2])
# plt.plot(x_space, phi_ao[3])
# plt.plot(x_space, phi_ao[4])
# plt.plot(x_space, pot)
# plt.show()


def inner_prod(psi1, psi2, x_space):
    h = x_space[1]- x_space[0]
    return np.trapezoid(psi1 * psi2, dx=h)

def calc_S(phi_ao, x_space):
    n_basis = np.shape(phi_ao)[0]
    S = np.zeros((n_basis, n_basis))

    for j in range(n_basis):
        for i in range(n_basis): 
            S[i,j] = inner_prod(phi_ao[i], phi_ao[j], x_space)
    return S


#useful functions
# why are there nonzero values at the boundaries?
# alternatively instead of zero padding 
def grid_laplacian(wf, x_space):
    h = x_space[1] - x_space[0]
    laplacian = np.zeros(np.shape(wf))
    wf = np.pad(wf, pad_width=1, mode="constant", constant_values=0)
    laplacian = (wf[:-2] + wf[2:] - 2 * wf[1:-1])/h**2
    return laplacian
      
def apply_hamiltonian(pot, phi, x_space):
    H = -0.5 * grid_laplacian(phi, x_space) + (pot * phi)
    return H

def calc_H(pot, phi_ao, x_space):
    n_basis = np.shape(phi_ao)[0]
    H = np.zeros((n_basis, n_basis))

    for j in range(n_basis):
        for i in range(n_basis): 
            ket = apply_hamiltonian(pot, phi_ao[j], x_space)
            H[i,j] = inner_prod(phi_ao[i], ket, x_space)
    return H



if __name__ == "__main__":

    x_space = make_space(N_CENTERS, D, N_SINGLE)
    pot = multiwell_pot(poeschl_teller, N_CENTERS, D, x_space)

    #set up basis functions
    xg = np.load("numerov-grid.npy") #load the gridpoints that were used to caluclate the basis functions
    n_basis = N_STATES * N_CENTERS 
    phi_ao = np.zeros((N_STATES, N_CENTERS, np.shape(x_space)[0]))
    ao = np.load("numerov-five.npy")
    gerade = (N_CENTERS%2 == 0)
    for i in range(N_STATES):
        psi = ao[i]
        count = 1
        if gerade:
            for j in range(N_CENTERS):
                abs_shift = j // 2
                count *= -1
                phi_ao[i,j] = move_center(psi, (abs_shift + 0.5) * D* count, x_space, xg=xg)
        if not gerade:
            for j in range(N_CENTERS):
                count *= -1
                if j == 0:
                    phi_ao[i,j] = move_center(psi, 0, x_space, xg=xg)
                else:
                    abs_shift = (j+1) // 2
                    phi_ao[i,j] = move_center(psi, abs_shift * D * count, x_space, xg=xg)

    phi_ao = np.reshape(phi_ao, (n_basis, np.shape(x_space)[0]))
    
    #actual LCAO calculation
    H = calc_H(pot, phi_ao, x_space)
    S = calc_S(phi_ao, x_space)
    print(S)

    E, c_vec = eigh(H, S)
    E = np.real(E)
    mol_orb = c_vec.T @ phi_ao
    print(E)

    #visualize
    show_orb = range(N_CENTERS * 3)
    # for i in show_orb:
    #     plt.plot(x_space, mol_orb[i] + E[i], label=f"{E[i]:.2f}")
    plt.plot(x_space, mol_orb[0] + E[0])
    plt.plot(x_space, pot, ls="--")
    print(phi_ao.shape[0])
    # plt.legend()
    plt.show()