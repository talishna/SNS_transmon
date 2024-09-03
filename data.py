import numpy as np
import matplotlib.pyplot as plt
import time
from total_h import TotalHamiltonian


class Data:
    def __init__(self, E_C, n_0, E_J_max, d, flux_0, Wc, max_num_photons, N, t, epsilon_r, g, gamma_L, gamma_R,
                 flux_array, n_g_array, cutoff_transmon=False, size_subspace_transmon=None):
        self.flux_array = flux_array
        self.n_g_array = n_g_array
        self.cutoff_transmon = cutoff_transmon
        self.size_subspace_transmon = size_subspace_transmon
        self.total_dim = (2*n_0+1)*(max_num_photons+1)*(2**N)
        self.H_total = TotalHamiltonian(E_C=E_C, n_0=n_0, E_J_max=E_J_max, d=d, flux_0=flux_0,
                                        Wc=Wc, max_num_photons=max_num_photons, N=N, t=t, epsilon_r=epsilon_r, g=g,
                                        gamma_L=gamma_L, gamma_R=gamma_R, cutoff_transmon=cutoff_transmon,
                                        size_subspace_transmon=size_subspace_transmon)
        self.eigenvalues_n_g = None

    def eigen_for_each_n_g(self):
        eigenvalues_n_g = np.zeros((self.n_g_array.shape[0], self.total_dim))
        eigenvectors_n_g = np.zeros((self.n_g_array.shape[0], self.total_dim, self.total_dim), dtype=complex)
        for i in range(self.n_g_array.shape[0]):
            H = self.H_total.total_hamiltonian(n_g=self.n_g_array[i])
            current_eigenvalues, current_eigenvectors = np.linalg.eigh(H)
            print("eigen number:", i)
            eigenvalues_n_g[i, :] = current_eigenvalues  # the difference from the GS
            eigenvectors_n_g[i, :, :] = current_eigenvectors
        self.eigenvalues_n_g = eigenvalues_n_g
        return eigenvalues_n_g, eigenvectors_n_g

    def energy_diff_n_g(self):
        amount_of_energies_n_g = self.total_dim
        amount_of_energy_diff = np.sum(list(range(amount_of_energies_n_g)))
        delta_energy_n_g_temp = np.zeros((self.n_g_array.shape[0], amount_of_energies_n_g, amount_of_energies_n_g))  # contains the energy differences, 0 axis in the size of n_g
        # array, 1 axis and 2 axis in size of amount of energies. so in each [:,i,j] i will the diff E_i-E_j
        for i in range(amount_of_energies_n_g):  # a loop that iterates from 0 to 6 including
            print("i =", i)
            for j in range(i + 1, amount_of_energies_n_g):  # a loop that iterates from 0 to 6 including
                diff = self.eigenvalues_n_g[:, i] - self.eigenvalues_n_g[:, j]  # should be an array with number of
                # rows as "steps" (and flux_array) and one column
                delta_energy_n_g_temp[:, i, j] = diff
        delta_energy_n_g = np.zeros((self.n_g_array.shape[0], amount_of_energy_diff))
        upper_triangle_indices = np.triu_indices(delta_energy_n_g_temp.shape[1], k=1)
        for i in range(self.n_g_array.shape[0]):
            delta_energy_n_g[i, :] = delta_energy_n_g_temp[i][upper_triangle_indices]
        return delta_energy_n_g


    def plot_energy_diff_vs_n_g(self, amount):
        # Implementation of plot_energy_diff_vs_n_g function
        pass

    def plot_energy_vs_n_g(self, amount):
        # Implementation of plot_energy_vs_n_g function
        pass
