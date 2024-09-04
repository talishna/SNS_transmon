import numpy as np

class Transmon:
    """
    Represents a transmon qubit.

    Attributes:
        E_C (float): Charging energy.
        n_0 (int): Number of Cooper pairs.
        E_J_max (float): Maximum Josephson energy.
        d (float): Squid asymmetry parameter.
        flux_0 (float): Flux quantum.
        size_of_subspace (int): Size of the computational subspace.
        dimension (int): Dimension of the Hilbert space.
        creation (ndarray): Creation operator.
        annihilation (ndarray): Annihilation operator.
        n_hat (ndarray): Number operator.

    Methods:
        compute_hamiltonian(flux=0, n_g=0, cutoff_transmon=False): Computes the Hamiltonian of the transmon qubit.
        compute_sin_phi_half(): Computes the matrix representation of the sin(phi/2) operator.
        compute_cos_phi_half(): Computes the matrix representation of the cos(phi/2) operator.
    """

    def __init__(self, E_C, n_0, E_J_max, d=0, flux_0=1, size_of_subspace=None):
        """
        Initializes the Transmon object.

        Args:
            E_C (float): Charging energy.
            n_0 (int): Number of Cooper pairs.
            E_J_max (float): Maximum Josephson energy.
            d (float, optional): Squid asymmetry parameter. Default is 0.
            flux_0 (float, optional): Flux quantum. Default is 1.
            size_of_subspace (int, optional): Size of the computational subspace. Default is None.
        """
        self.E_C = E_C
        self.n_0 = n_0
        self.E_J_max = E_J_max
        self.d = d
        self.flux_0 = flux_0
        self.size_of_subspace = size_of_subspace
        self.dimension = 2 * int(self.n_0) + 1

        # Precompute creation and annihilation operators
        self.creation = self._compute_creation()
        self.annihilation = self.creation.conj().T
        self.n_hat = self._compute_n_operator()

    # Defining the creation and annihilation operators for the SCs
    def _compute_creation(self):
        """
        Computes the creation operator matrix.

        Returns:
            ndarray: The creation operator matrix.
        """
        exp_phi_plus = np.eye(self.dimension, k=-1)
        return exp_phi_plus

    def _compute_n_operator(self):
        """
        Computes the relative number of cooper pairs operator.

        Returns:
            ndarray: The number operator matrix.
        """
        n_values = np.array([-self.n_0 + i for i in range(self.dimension)])
        n = np.diag(n_values)
        return n

    def compute_hamiltonian(self, flux=0, n_g=0, cutoff_transmon=False):
        """
        Computes the Hamiltonian of the transmon qubit.

        Args:
            flux (float, optional): Flux. Default is 0.
            n_g (float, optional): Gate charge. Default is 0.
            cutoff_transmon (bool, optional): Whether to apply a cutoff for the transmon. Default is False.

        Returns:
            ndarray: The Hamiltonian matrix.
        """
        D = 4 * self.E_C * (self.n_hat - n_g * np.eye(self.dimension)) ** 2
        asymmetry = (self.E_J_max / 2) * np.sqrt(np.cos(np.pi * flux / self.flux_0) ** 2 +
                                                 (self.d ** 2) * np.sin(np.pi * flux / self.flux_0) ** 2)
        H_temp = D - asymmetry * (self.creation + self.annihilation)
        return H_temp


    def compute_sin_phi_half(self):
        """
        Computes the matrix representation of the sin(phi/2) operator.

        Returns:
            ndarray: The sin(phi/2) operator matrix.
        """
        sin_phi_half = np.zeros((self.dimension, self.dimension), dtype=complex)
        n_values = np.array([-self.n_0 + i for i in range(self.dimension)])
        for n in range(self.dimension):
            for m in range(self.dimension):
                sin_phi_half[n, m] = -1 / (2 * np.pi * ((n_values[m]-n_values[n]) ** 2 - 1/4))
        return sin_phi_half


    def compute_cos_phi_half(self):
        """
        Computes the matrix representation of the cos(phi/2) operator.

        Returns:
            ndarray: The cos(phi/2) operator matrix.
        """
        cos_phi_half = np.zeros((self.dimension, self.dimension), dtype=complex)
        n_values = np.array([-self.n_0 + i for i in range(self.dimension)], dtype=complex)
        for n in range(self.dimension):
            for m in range(self.dimension):
                cos_phi_half[n, m] = 1j * (n_values[m]-n_values[n]) / (np.pi * ((n_values[m]-n_values[n]) ** 2 - 1/4))
        return cos_phi_half