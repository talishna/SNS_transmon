import unittest
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum.sho1d import Hamiltonian
from transmon import Transmon
from utils import is_hermitian

E_J = np.array([1,5,10,50])
E_C = 1
N = 5

class TestTransmon(unittest.TestCase):
    def setUp(self):
        self.E_C = 1
        self.E_J_max = 1
        self.n_0 = 5
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
        steps = 200
        number_of_energies = 5
        n_g_array = np.linspace(-2, 2, steps)
        E_J_values = [1, 5, 10, 50]  # Different E_J values
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join("transmon_test_plots", f"energy_levels_vs_ng_2x2_{timestamp}.png")

        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten the 2D array for easier indexing

        for idx, E_J in enumerate(E_J_values):
            # Create a new Transmon object for each E_J
            transmon = Transmon(self.E_C, self.n_0, E_J)

            # Compute energy levels
            energy_levels = []
            for n_g in n_g_array:
                hamiltonian = transmon.compute_hamiltonian(n_g=n_g)
                eigenvalues = np.linalg.eigvalsh(hamiltonian)
                energy_levels.append(eigenvalues[:number_of_energies])  # Store first 5 energy levels

            energy_levels = np.reshape(np.array(energy_levels), (steps, number_of_energies))

            # Plot energy levels on the corresponding subplot
            ax = axes[idx]
            for i in range(number_of_energies):
                ax.plot(n_g_array, energy_levels[:, i], label=f"Level {i}")
            ax.set_title(f"$E_J = {E_J}$")
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


if __name__ == '__main__':
    unittest.main()
