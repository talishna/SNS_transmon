import numpy as np


class Chain:
    """
    Represents a chain of spins with a specified interaction and on-site energy.

    Attributes:
        N (int): Number of spins in the chain.
        t (float): Coupling strength between spins.
        epsilon_r (float): On-site energy.
        dimension (int): Dimension of the Hilbert space.

        s_x (ndarray): Spin operator for the x direction.
        s_y (ndarray): Spin operator for the y direction.
        s_z (ndarray): Spin operator for the z direction.
        s_plus (ndarray): Spin raising operator.
        s_minus (ndarray): Spin lowering operator.

    Methods:
        compute_hamiltonian(): Computes the Hamiltonian of the spin chain.
    """

    def __init__(self, N, t, epsilon_r):
        """
        Initializes the Chain with given parameters.

        Args:
            N (int): Number of sites in the chain.
            t (float): Hopping parameter.
            epsilon_r (float): On-site energy.
        """
        self.N = N
        self.t = t
        self.epsilon_r = epsilon_r
        self.dimension = 2 ** self.N

        # Defining the single spin operators s_x, s_y, s_z:
        self.s_x = 0.5 * np.array([[0, 1], [1, 0]])
        self.s_y = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.s_z = 0.5 * np.array([[1, 0], [0, -1]])
        self.s_plus = np.array([[0, 1], [0, 0]])  # self.s_x + 1j * self.s_y
        self.s_minus = np.array([[0, 0], [1, 0]])  # self.s_x - 1j * self.s_y

    def compute_hamiltonian(self):
        """
        Computes the Hamiltonian matrix for the chain.

        Returns:
            ndarray: The Hamiltonian matrix.
        """
        H = np.zeros((2 ** self.N, 2 ** self.N))
        if self.N < 2:
            return H
        for i in range(self.N - 2):
            term = np.kron(np.eye(2 ** ((i + 1) - 1)),
                           np.kron(self.s_plus,
                                   np.kron(-self.s_z,
                                           np.kron(self.s_minus, np.eye(2 ** (self.N - ((i + 1) + 2)))))))
            # I am adding +1 in all the dimensions because the range starts from 0 and the spin index start from 1
            H += -self.t * (term + term.conj().T)
        for i in range(self.N):
            on_site_energy = self.epsilon_r * np.kron(np.eye(2 ** ((i + 1) - 1)),
                                                      np.kron(self.s_z + 0.5 * np.eye(2),
                                                              np.eye(2 ** (self.N - (i + 1)))))
            H += on_site_energy
        return H
