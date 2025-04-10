import numpy as np
import matplotlib.pyplot as plt
from transmon import Transmon
from cavity import Cavity
from spinchain import SpinChain
from data import Data
import os

"""
Simple example of uses for the classes Transmon Cavity SpinChain. For a more complex one see "suppresed_our_model" and "suppresed_article_model"
"""

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

# SpinChain parameters
N_sites = 2  # number of chain fermions\
t = 0.1  # chain hopping
epsilon_r = 0  # on site energy in the chain
gamma_L = 0.1  # chain coupling to the left josephson junction
gamma_R = gamma_L * (1 - d) / (1 + d)  # chain coupling to the right josephson junction

# Drive parameter
g_d = 1

# General parameters
steps = 200
flux_array = np.linspace(-flux_0, flux_0, steps)
n_total = (2 * n_0 + 1) * (max_num_photons + 1) * (2 ** N_sites)
n_g_array = np.linspace(-2, 2, steps)

# Instantiate classes
transmon1 = Transmon(E_C, n_0, E_J_max, d, flux_0, size_of_transmon_subspace)
cavity1 = Cavity(Wc, max_num_photons)
H_transmon = transmon1.compute_hamiltonian(n_g=1)
H_cavity = cavity1.compute_hamiltonian()
chain1 = SpinChain(N_sites=N_sites, t=t, epsilon_r=epsilon_r)
plot_data = Data(E_C=E_C, n_0=n_0, E_J_max=E_J_max, d=d, flux_0=flux_0, Wc=Wc, max_num_photons=max_num_photons, N=N_sites,
                 t=t, epsilon_r=epsilon_r, g=g, gamma_L=gamma_L, gamma_R=gamma_R,
                 flux_array=flux_array, n_g_array=n_g_array)

eigenvalues_n_g, eigenvectors_n_g = plot_data.eigen_for_each_n_g()
energy_diff_n_g = plot_data.energy_diff_n_g()


def plot_y_vs_x(y_data, x_data, xlabel, ylabel, filename, amount):
    """
    Plots y_data vs x_data for a specified number of data series.

    Args:
        y_data (ndarray): The y-axis data to plot (e.g., energies or energy differences).
        x_data (ndarray): The x-axis data to plot against (e.g., n_g_array).
        filename (str): The filename to save the plot.
        amount (int): The number of series to plot.
    """
    for i in range(amount):
        plt.plot(x_data, y_data[:, i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()



if __name__ == '__main__':
    # Plot energy differences vs. n_g
    plot_y_vs_x(
        y_data=abs(energy_diff_n_g),
        x_data=n_g_array,
        xlabel=r'${n_g}$',
        ylabel='Energy differences',
        filename=os.path.join("data2", f'diff_n_g_only_cav_transmon_amount_of_energies_diff_4_CPnum_{n_0}_PhotonsNum_{max_num_photons}.png'),
        amount=4
    )

    # Plot energies vs. n_g
    plot_y_vs_x(
        y_data=eigenvalues_n_g - eigenvalues_n_g[:, 0][:, np.newaxis],
        x_data=n_g_array,
        xlabel=r'${n_g}$',
        ylabel='Energy',
        filename=os.path.join("data2", f'n_g_only_cav_transmon_amount_of_energies_4_CPnum_{n_0}_PhotonsNum_{max_num_photons}.png'),
        amount=4
    )
