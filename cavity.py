import numpy as np
import matplotlib.pyplot as plt
import time


class Cavity:
    """
    Class representing a cavity.
    Attributes:
        Wc (float): Cavity frequency.
        max_num_photons (int): Maximum number of photons.
        dimension (int): Dimension of the Hilbert space.
        occupation_array (ndarray): Array representing the occupation numbers.
        creation (ndarray): Creation operator.
        annihilation (ndarray): Annihilation operator.
        occupation_operator (ndarray): Occupation operator.
    """

    def __init__(self, Wc, max_num_photons):
        self.Wc = Wc
        self.max_num_photons = max_num_photons
        self.dimension = max_num_photons + 1
        self.occupation_array = np.arange(1, self.dimension)

        # Precompute creation and annihilation operators
        self.creation = self._compute_creation()
        self.annihilation = self.creation.conj().T
        self.occupation_operator = self.creation @ self.annihilation

    def _compute_creation(self):
        a_up = np.diag(np.sqrt(self.occupation_array), k=-1)
        return a_up

    def compute_hamiltonian(self):
        return self.Wc * self.occupation_operator

    # def annihilation(self):
    #     a_down = np.diag(np.sqrt(self.occupation_array), k=1)
    #     return a_down

    # def occupation_operator(self):
    #     return self.creation() @ self.annihilation()
