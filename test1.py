# This is a test file to test the deltaE vs n_g and the E vs n_g plots for only the transmon and cavity case

import numpy as np
import matplotlib.pyplot as plt
import time
from transmon import Transmon
from cavity import Cavity
from chain import Chain
from data import Data

# Transmon parameters
E_C = 1  # charging energy
n_0 = 10  # 10 # number of CP
E_J_max = 5  # (Wq+E_C)**2/8/E_C
d = 0  # 0.35 #squid asymmetry
flux_0 = 1  # 2.067833 * 10 ** (-15)
size_of_transmon_subspace = 0
plasma_energy = np.sqrt(8 * E_C * E_J_max)
ratio = E_J_max / E_C

# Cavity parameters
g = 1  # 0.08 #transmon cavity coupling
Wq0 = 3.474  # dressed qubit frequency
Wc0 = 7.192  # dressed cavity frequency
lamb_shift = g ** 2 / (Wc0 - Wq0)
Wc = Wc0 - lamb_shift  # bare qubit
# Wq = Wq0 + lamb_shift  # bare cavity
max_num_photons = 6  # number of photons

# Chain parameters
N = 0  # number of chain fermions\
t = 0.1  # chain hopping
epsilon_r = 0.1  # on site energy in the chain
gamma_L = 0.1  # chain coupling to the left josephson junction
gamma_R = gamma_L * (1 - d) / (1 + d)  # chain coupling to the right josephson junction

# Drive parameter
g_d = 1

# General parameters
steps = 200
n_g = 0
flux_array = np.linspace(-flux_0, flux_0, steps)
n_total = (2 * n_0 + 1) * (max_num_photons + 1) * (2 ** N)
n_g_array = np.linspace(-2, 2, steps)

# Instantiate classes
# transmon1 = Transmon(E_C, n_0, E_J_max, d, flux_0, size_of_transmon_subspace)
# cavity1 = Cavity(Wc, max_num_photons)
# H_transmon = transmon1.compute_hamiltonian(n_g=1)
# H_cavity = cavity1.compute_hamiltonian()
plot_data = Data(E_C=E_C, n_0=n_0, E_J_max=E_J_max, d=d, flux_0=flux_0, Wc=Wc, max_num_photons=max_num_photons, N=N,
                 t=t, epsilon_r=epsilon_r, g=g, gamma_L=gamma_L, gamma_R=gamma_R,
                 flux_array=flux_array, n_g_array=n_g_array)

eigenvalues_n_g, eigenvectors_n_g = plot_data.eigen_for_each_n_g()
energy_diff_n_g = plot_data.energy_diff_n_g()


def plot_energy_diff_vs_n_g(amount=eigenvalues_n_g.shape[1]):
    for i in range(amount):
        plt.plot(n_g_array, abs(energy_diff_n_g[:, i]))
    plt.xlabel(r'${n_g}$')
    plt.ylabel('Energy differences')
    plt.axhline(y=5.5, color='r', linestyle='--')
    # plt.title('Energy differences from GS (asymmetric transmon, cavity and chain)')

    # Save the figure as an image (e.g., PNG)
    filename = f'diff_n_g_only_cav_transmon_amount_of_energies_diff_{amount}_CPnum_{n_0}_PhotonsNum_{max_num_photons}.png'
    plt.savefig(filename)
    plt.show()


def plot_energy_vs_n_g(amount):
    for i in range(amount):
        plt.plot(n_g_array, eigenvalues_n_g[:, i] - eigenvalues_n_g[:, 0])
    plt.xlabel(r'${n_g}$')
    plt.ylabel('Energy')
    plt.axhline(y=5.5, color='r', linestyle='--')
    # plt.title('Energy differences from GS (asymmetric transmon, cavity and chain)')

    # Save the figure as an image (e.g., PNG)
    filename = f'n_g_only_cav_transmon_amount_of_energies_{amount}_CPnum_{n_0}_PhotonsNum_{max_num_photons}.png'
    plt.savefig(filename)
    plt.show()


plot_energy_diff_vs_n_g(4)
plot_energy_vs_n_g(4)
