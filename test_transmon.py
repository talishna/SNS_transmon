import unittest
import numpy as np
from sympy.physics.quantum.sho1d import Hamiltonian

from transmon import Transmon
from test_utils import is_hermitian

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


if __name__ == '__main__':
    unittest.main()
