import unittest
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum.sho1d import Hamiltonian
from transmon import Transmon
from utils import is_hermitian, matrix_element_Mfi, delta_energies, upper_triangle, plot_x_y_color

E_J = np.array([1,5,10,50])
E_C = 1
N = 5
number_of_energies = 5

class TestTransmon(unittest.TestCase):
    def setUp(self):
        self.E_C = 1
        self.E_J_max = 1
        self.n_0 = 5
        self.E_J_values = np.array([1, 5, 10, 50])  # Different E_J values
        self.number_of_energies = 5
        self.steps = 200
        self.n_g_array = np.linspace(-2, 2, self.steps)
        self.phi = np.linspace(-np.pi, np.pi, self.steps)
        self.transmon = Transmon(self.E_C, self.n_0, self.E_J_max)
        self.hamiltonian = self.transmon.compute_hamiltonian()
        self.sin_phi_half = self.transmon.compute_sin_phi_half()
        self.cos_phi_half = self.transmon.compute_cos_phi_half()
    def test__compute_creation(self):
        self.assertIsInstance(self.transmon.creation, np.ndarray)  # Test matrix type is ndarray
        self.assertEqual(self.transmon.creation.shape,
                         (self.transmon.dimension, self.transmon.dimension))  # Test matrix size is as dimension

    def test_annihilation(self):
        exp_phi_minus = np.eye(self.transmon.dimension, k=1)
        np.testing.assert_array_equal(self.transmon.annihilation, exp_phi_minus)  # Test the annihilation operator is correct

    def test__compute_n_operator(self):
        self.assertIsInstance(self.transmon.n_hat, np.ndarray)  # Test matrix type is ndarray
        self.assertEqual(self.transmon.n_hat.shape,
                         (self.transmon.dimension, self.transmon.dimension))  # Test matrix size is as dimension

    def test_compute_hamiltonian(self):
        self.assertIsInstance(self.hamiltonian, np.ndarray)  # Test matrix type is ndarray
        self.assertEqual(self.hamiltonian.shape,
                         (self.transmon.dimension, self.transmon.dimension))  # Test matrix size is as dimension
        self.assertTrue(is_hermitian(self.hamiltonian), "Hamiltonian is not Hermitian")  # Test if hermitian

    def test_compute_sin_phi_half(self):
        self.assertIsInstance(self.sin_phi_half, np.ndarray)  # Test matrix type is ndarray
        self.assertEqual(self.sin_phi_half.shape,
                         (self.transmon.dimension, self.transmon.dimension))  # Test matrix size is as dimension

    def test_compute_cos_phi_half(self):
        self.assertIsInstance(self.cos_phi_half, np.ndarray)  # Test matrix type is ndarray
        self.assertEqual(self.cos_phi_half.shape,
                         (self.transmon.dimension, self.transmon.dimension))  # Test matrix size is as dimension

    def test_plot_energy_levels_vs_ng(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join("transmon_test_plots", f"energy_levels_vs_ng_2x2_{timestamp}.png")

        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten the 2D array for easier indexing

        for idx, E_J in enumerate(self.E_J_values):
            # Create a new Transmon object for each E_J
            transmon = Transmon(self.E_C, self.n_0, E_J)

            # Compute energy levels
            energy_levels = []
            for n_g in self.n_g_array:
                hamiltonian = transmon.compute_hamiltonian(n_g=n_g)
                eigenvalues = np.linalg.eigvalsh(hamiltonian)
                energy_levels.append(eigenvalues[:self.number_of_energies])  # Store first 5 energy levels

            energy_levels = np.reshape(np.array(energy_levels), (self.steps, self.number_of_energies))

            # Plot energy levels on the corresponding subplot
            ax = axes[idx]
            for i in range(self.number_of_energies):
                ax.plot(self.n_g_array, energy_levels[:, i], label=f"Level {i}")
            ax.plot(self.n_g_array, np.full_like(self.n_g_array, np.sqrt(8 * E_J * E_C)), linestyle='--', label=r'$\sqrt{8 E_J E_C}$')
            ax.set_title(f'$E_J/E_C={int(E_J/E_C)}$')
            ax.set_xlabel("$n_g$")
            ax.set_ylabel("Energy")
            ax.legend()
            ax.grid()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()

        # Assert the plot file is created
        self.assertTrue(os.path.exists(plot_filename), f"Plot file {plot_filename} was not created.")

    def test_plot_eigenfunctions_vs_phi_2x2(self):
        """
        Test plotting the first 5 eigenfunctions of the Hamiltonian as a function of phase phi
        for 4 different E_J/E_C ratios when n_g = 0. Includes real and imaginary parts for all functions.
        """
        E_J_ratios = self.E_J_values/self.E_C  # Ratios of E_J/E_C
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join("transmon_test_plots", f"eigenfunctions_vs_phi_2x2_{timestamp}.png")

        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten the 2D array for easier indexing

        for idx, E_J_ratio in enumerate(E_J_ratios):
            E_J = E_J_ratio / self.E_C  # Calculate E_J based on the ratio
            transmon = Transmon(self.E_C, self.n_0, E_J)
            hamiltonian = transmon.compute_hamiltonian(n_g=0)

            # Compute eigenvalues and eigenvectors
            _, eigenvectors = np.linalg.eigh(hamiltonian)

            # Compute the eigenfunctions
            func_mat = np.zeros((self.steps, self.number_of_energies), dtype=complex)  # Rows: phi values, Columns: eigenfunctions
            for f in range(self.number_of_energies):  # Loop over the first 5 eigenfunctions
                for i in range(self.steps):  # Loop over phi values
                    func_mat[i, f] = sum(
                        (1 / np.sqrt(2 * np.pi)) *
                        eigenvectors[n, f] *
                        np.exp(1j * (n - transmon.n_0) * self.phi[i])  # The n-n_0 is so that the exponent will go from
                        # -n_0 to n_0
                        for n in range(transmon.dimension)
                    )

            # Plot the eigenfunctions
            ax = axes[idx]
            for f in range(self.number_of_energies):  # Plot only the first 5 eigenfunctions
                ax.plot(self.phi, np.real(func_mat[:, f]), linestyle='-', label=f"Re(Function {f})")
                ax.plot(self.phi, np.imag(func_mat[:, f]), linestyle='--', label=f"Im(Function {f})")

            ax.set_title(f"$E_J/E_C = {E_J_ratio}$")
            ax.set_xlabel("$\\phi$")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()

        # Assert the plot file is created
        self.assertTrue(os.path.exists(plot_filename), f"Plot file {plot_filename} was not created.")


    def test_plot_delta_energy_vs_ng_dipole_transitions(self):
        n_operator = self.transmon.n_hat
        eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
        M = matrix_element_Mfi(n_operator, eigenvectors, self.number_of_energies)
        delta = delta_energies(eigenvalues, self.number_of_energies)
        M = upper_triangle(M)
        delta = upper_triangle(delta)



if __name__ == '__main__':
    unittest.main()
