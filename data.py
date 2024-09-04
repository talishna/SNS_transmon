import numpy as np
from total_hamiltonian import TotalHamiltonian
a = 5

class Data:
    """
    Represents a data handler for a quantum system with transmons, cavities, and spin chains.

    Attributes:
        flux_array (ndarray): Array of flux values.
        n_g_array (ndarray): Array of gate charge values.
        cutoff_transmon (bool): Indicates whether to apply a cutoff for the transmon.
        size_subspace_transmon (int): Size of the computational subspace for the transmon.
        total_dim (int): Total dimension of the combined Hilbert space.
        H_total (TotalHamiltonian): Instance of the TotalHamiltonian class representing the total Hamiltonian of the
        system.
        eigenvalues_n_g (ndarray): Array to store eigenvalues for each gate charge value.

    Methods:
        eigen_for_each_n_g(): Computes eigenvalues and eigenvectors for each gate charge value in n_g_array.
        energy_diff_n_g(): Computes the energy differences between eigenstates for each gate charge value.
        plot_energy_diff_vs_n_g(amount): Plots the energy differences as a function of gate charge for a specified
        number of energy levels.
        plot_energy_vs_n_g(amount): Plots the energy levels as a function of gate charge for a specified number of
        energy levels.
    """
    def __init__(self, E_C, n_0, E_J_max, d, flux_0, Wc, max_num_photons, N, t, epsilon_r, g, gamma_L, gamma_R,
                 flux_array, n_g_array, cutoff_transmon=False, size_subspace_transmon=None):
        """
        Initializes the Data object.

        Args:
            E_C (float): Charging energy.
            n_0 (int): Number of Cooper pairs.
            E_J_max (float): Maximum Josephson energy.
            d (float): Squid asymmetry parameter.
            flux_0 (float): Flux quantum.
            Wc (float): Cavity frequency.
            max_num_photons (int): Maximum number of photons in the cavity.
            N (int): Number of spins in the chain.
            t (float): Hopping parameter in the chain.
            epsilon_r (float): On-site energy in the chain.
            g (float): Coupling strength between the transmon and the cavity.
            gamma_L (float): Left coupling rate.
            gamma_R (float): Right coupling rate.
            flux_array (ndarray): Array of flux values.
            n_g_array (ndarray): Array of gate charge values.
            cutoff_transmon (bool, optional): Whether to apply a cutoff for the transmon. Default is False.
            size_subspace_transmon (int, optional): Size of the computational subspace for the transmon. Default is None.
        """
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
        """
        Computes eigenvalues and eigenvectors for each gate charge value in n_g_array.

        Returns:
            tuple: A tuple containing the eigenvalues and eigenvectors for each gate charge value.
        """
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
        """
        Computes the energy differences between eigenstates for each gate charge value.

        Returns:
            ndarray: Array of energy differences for each gate charge value.
        """
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
