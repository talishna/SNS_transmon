import numpy as np
a = 5

class Cavity:
    """
    Quantum optical cavity with a given frequency and maximum number of photons.

    Attributes:
        Wc (float): The cavity frequency.
        max_num_photons (int): The maximum number of photons in the cavity.
        dimension (int): The dimension of the Hilbert space, which is max_num_photons + 1.
        occupation_array (ndarray): Array of occupation numbers ranging from 1 to max_num_photons.
        creation (ndarray): Creation operator matrix.
        annihilation (ndarray): Annihilation operator matrix, conjugate transpose of the creation operator.
        occupation_operator (ndarray): Occupation operator matrix, computed as the product of creation and annihilation operators.

    Methods:
        compute_hamiltonian():
            Computes and returns the Hamiltonian matrix of the cavity system.
    """

    def __init__(self, Wc, max_num_photons):
        """
        Initializes the Cavity with given parameters.

        Args:
            Wc (float): The cavity frequency.
            max_num_photons (int): The maximum number of photons in the cavity.
        """
        self.Wc = Wc
        self.max_num_photons = max_num_photons
        self.dimension = max_num_photons + 1
        self.occupation_array = np.arange(1, self.dimension)

        # Precompute creation and annihilation operators
        self.creation = self._compute_creation()
        self.annihilation = self.creation.conj().T
        self.occupation_operator = self.creation @ self.annihilation

    def _compute_creation(self):
        """
        Computes the creation operator matrix.

        Returns:
            ndarray: The creation operator matrix.
        """
        a_up = np.diag(np.sqrt(self.occupation_array), k=-1)
        return a_up

    def compute_hamiltonian(self):
        """
        Computes the Hamiltonian of the cavity system.

        Returns:
            ndarray: The Hamiltonian matrix, which is the product of the cavity frequency and the occupation operator.
        """
        return self.Wc * self.occupation_operator
