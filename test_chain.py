import unittest
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sympy.physics.quantum.sho1d import Hamiltonian

from spinchain import SpinChain
from transmon import Transmon
from utils import is_hermitian, matrix_element_Mfi, delta_energies, upper_triangle, plot_x_y_color, colored_line

E_J = np.array([1,5,10,50])
E_C = 1
N = 5
number_of_energies = 5

class TestChain(unittest.TestCase):
    def setUp(self):
        self.N = 4
        self.t = 1
        self.epsilon_r = 1
        self.chain = SpinChain(self.N, self.t, self.epsilon_r)
        self.hamiltonian = self.chain.compute_hamiltonian()
        self.number_of_energies = 3
        self.steps = 200
        self.k = np.linspace(-0.5, 0.5, self.steps)

    def test_compute_hamiltonian(self):
        self.assertIsInstance(self.hamiltonian, np.ndarray)  # Test matrix type is ndarray
        self.assertEqual(self.hamiltonian.shape,
                         (self.chain.dimension, self.chain.dimension))  # Test matrix size is as dimension
        self.assertTrue(is_hermitian(self.hamiltonian), "Hamiltonian is not Hermitian")  # Test if hermitian



    # def test_plot_energy_levels_vs_ng(self):
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     plot_filename = os.path.join("transmon_test_plots", f"energy_levels_vs_ng_2x2_{timestamp}.png")
    #
    #     # Create a 2x2 grid of subplots
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #     axes = axes.flatten()  # Flatten the 2D array for easier indexing
    #
    #     for idx, E_J in enumerate(self.E_J_values):
    #         # Create a new Transmon object for each E_J
    #         transmon = Transmon(self.E_C, self.n_0, E_J)
    #
    #         # Compute energy levels
    #         energy_levels = []
    #         for n_g in self.n_g_array:
    #             hamiltonian = transmon.compute_hamiltonian(n_g=n_g)
    #             eigenvalues = np.linalg.eigvalsh(hamiltonian)
    #             energy_levels.append(eigenvalues[:self.number_of_energies])  # Store first 5 energy levels
    #
    #         energy_levels = np.reshape(np.array(energy_levels), (self.steps, self.number_of_energies))
    #
    #         # Plot energy levels on the corresponding subplot
    #         ax = axes[idx]
    #         for i in range(self.number_of_energies):
    #             ax.plot(self.n_g_array, energy_levels[:, i], label=f"Level {i}")
    #         ax.plot(self.n_g_array, np.full_like(self.n_g_array, np.sqrt(8 * E_J * E_C)), linestyle='--', label=r'$\sqrt{8 E_J E_C}$')
    #         ax.set_title(f'$E_J/E_C={int(E_J/E_C)}$')
    #         ax.set_xlabel("$n_g$")
    #         ax.set_ylabel("Energy")
    #         ax.legend()
    #         ax.grid()
    #
    #     # Adjust layout and save the figure
    #     plt.tight_layout()
    #     plt.savefig(plot_filename)
    #     plt.close()
    #
    #     # Assert the plot file is created
    #     self.assertTrue(os.path.exists(plot_filename), f"Plot file {plot_filename} was not created.")
    #
    def test_plot_eigenfunctions_vs_k(self):
        """
        Test plotting the first 5 eigenfunctions of the Hamiltonian as a function of phase phi
        for 4 different E_J/E_C ratios when n_g = 0. Includes real and imaginary parts for all functions.
        """

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join("chain_test_plots", f"eigenfunctions_vs_k_{timestamp}.png")

        fig, ax = plt.subplots()

        # compute eigenvalues and eigenvectors
        _, eigenvectors = np.linalg.eigh(self.hamiltonian)

        # compute the eigenfunctions
        func_mat = np.zeros((self.steps, self.number_of_energies), dtype=complex)  # Rows: k values, Columns: eigenfunctions
        for f in range(self.number_of_energies):  # Loop over the first 5 eigenfunctions
            for i in range(self.steps):  # Loop over phi values
                func_mat[i, f] = sum(
                    (1 / np.sqrt(2 * np.pi)) *
                    eigenvectors[n, f] *
                    np.exp(1j * (n) * self.k[i])  # The n-n_0 is so that the exponent will go from
                    # -n_0 to n_0
                    for n in range(self.chain.dimension)
                )

        # plot the eigenfunctions
        for f in range(self.number_of_energies):  # Plot only the first 5 eigenfunctions
            ax.plot(self.k, np.real(func_mat[:, f]), linestyle='-', label=f"Re(Function {f})")
            ax.plot(self.k, np.imag(func_mat[:, f]), linestyle='--', label=f"Im(Function {f})")

        ax.set_title(f"$functions of chain$")
        ax.set_xlabel("$\\k$")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.show()

        # Assert the plot file is created
        self.assertTrue(os.path.exists(plot_filename), f"Plot file {plot_filename} was not created.")

    #
    # def test_plot_delta_energy_vs_ng_dipole_transitions(self):
    #     length = np.sum(range(self.number_of_energies))
    #     big_M = np.zeros((self.n_g_array.shape[0], length))
    #     big_delta = np.zeros_like(big_M)
    #     n_operator = self.transmon.n_hat
    #
    #     # compute M and delta for each n_g
    #     for i, n_g in enumerate(self.n_g_array):
    #         H = self.transmon.compute_hamiltonian(n_g=n_g)
    #         eigenvalues, eigenvectors = np.linalg.eigh(H)
    #         M = matrix_element_Mfi(n_operator, eigenvectors, self.number_of_energies)
    #         delta = delta_energies(eigenvalues, self.number_of_energies)
    #         big_M[i, :] = upper_triangle(M)
    #         big_delta[i, :] = upper_triangle(delta)
    #
    #     big_delta = -1*big_delta # minus so the energies will be positive
    #     big_M = abs(big_M)**2
    #
    #     # Combine color ranges for normalization
    #     vmin = big_M.min()
    #     vmax = big_M.max()
    #     norm = Normalize(vmin=vmin, vmax=vmax)
    #
    #     # Create the plot
    #     fig, ax = plt.subplots()
    #
    #     # create the description for the graph from the upper triangle indices
    #     upper_tri = np.triu_indices(self.number_of_energies, k=1)
    #     indices = [(y, x) for x, y in zip(*upper_tri)]
    #     descriptions = [f'E{x}-E{y}' for x, y in indices]
    #
    #     for i in range(length):
    #         line = colored_line(self.n_g_array, big_delta[:, i], big_M[:, i], ax, linewidth=2, cmap="plasma", norm=norm)
    #
    #         # annotate the line with its description
    #         mid_idx = len(self.n_g_array)//2
    #         ax.text(self.n_g_array[mid_idx], big_delta[mid_idx, i], descriptions[i], fontsize=8, color="black", ha="left")
    #
    #         # Add a single colorbar
    #         if i==(length-1):
    #             fig.colorbar(line, ax=ax)
    #
    #     ax.set_title("Transmon Transitions due to Dipole")
    #     ax.set_xlabel("n_g")
    #     ax.set_ylabel("Delta Energy")
    #     ax.set_xlim(self.n_g_array.min(), self.n_g_array.max())
    #     ax.set_ylim(big_delta.min(), big_delta.max())
    #     plt.tight_layout()
    #
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     plot_filename = os.path.join("transmon_test_plots", f"Transmon Transitions due to Dipole {timestamp}.png")
    #     plt.savefig(plot_filename, format='svg')
    #     plt.show()
    #
    #     # plot_x_y_color(big_M, self.n_g_array, big_delta, "n_g", "Delta Energy", 'Transmon Transitions due to Dipole', path=path)
    #


if __name__ == '__main__':
    unittest.main()
