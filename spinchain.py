import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

class SpinChain:
    """
    Represents a chain of spins with a specified interaction and on-site energy.

    Attributes:
        N_sites (int): Number of spins in the chain.
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

    def __init__(self, N_sites, t, epsilon_r):
        """
        Initializes the SpinChain with given parameters.

        Args:
            N_sites (int): Number of sites (spins including up and dowm) in the chain.
            t (float): Hopping parameter.
            epsilon_r (float): On-site energy.
        """
        self.N_sites = N_sites
        self.N_spins = 2*N_sites  # because of the spin we have 2 times more
        self.t = t
        self.epsilon_r = epsilon_r
        self.dimension = 2 ** self.N_spins

        # Defining the single spin operators s_x, s_y, s_z:
        self.s_x = csr_matrix([[0, 0.5], [0.5, 0]], dtype=np.float32)
        self.s_y = csr_matrix([[0, -0.5j], [0.5j, 0]], dtype=np.complex64)  # Only complex where needed
        self.s_z = csr_matrix([[0.5, 0], [0, -0.5]], dtype=np.float32)
        self.s_plus = csr_matrix([[0, 1], [0, 0]], dtype=np.float32)
        self.s_minus = csr_matrix([[0, 0], [1, 0]], dtype=np.float32)

    def compute_hamiltonian(self):
        """
        Computes the Hamiltonian matrix for the chain.

        Returns:
            scipy.sparse.csr_matrix: The sparse Hamiltonian matrix.
        """
        H = sp.csr_matrix((self.dimension, self.dimension), dtype=np.float32)  # Initialize sparse matrix
        s_z_mod = self.s_z + 0.5 * sp.eye(2, format="csr")

        for i in range(1, self.N_spins + 1):
            I_left =sp.eye(2 ** (i-1), format="csr")
            I_right =sp.eye(2 ** (self.N_spins - i), format="csr")
            on_site_energy = sp.kron(I_left, sp.kron(s_z_mod, I_right))
            H += self.epsilon_r * on_site_energy

            # Hopping term only if there's room for two more sites and N<4
            if self.N_spins >= 4 and (i < self.N_spins - 1):
                I_right2 = sp.eye(2 ** (self.N_spins - (i + 2)), format="csr")
                term = sp.kron(I_left,
                               sp.kron(self.s_plus,
                                       sp.kron(-self.s_z,
                                               sp.kron(self.s_minus, I_right2, format="csr"))), format="csr")
                H += -2 * self.t * (term + term.getH())

        # for i in range(1, self.N_spins + 1):
        #     on_site_energy = np.kron(np.eye(2 ** (i - 1)),
        #                                               np.kron(self.s_z + 0.5 * np.eye(2),
        #                                                       np.eye(2 ** (self.N_spins - i))))
        #     H += self.epsilon_r * on_site_energy
        # if self.N_spins < 4:
        #     return H
        # for i in range(1, self.N_spins - 1):
        #     term = np.kron(np.eye(2 ** (i - 1)),
        #                    np.kron(self.s_plus,
        #                            np.kron(-self.s_z,
        #                                    np.kron(self.s_minus, np.eye(2 ** (self.N_spins - (i + 2)))))))
        #
        #     H += -2 * self.t * (term + term.conj().T)
        return H

    def apply_operator_at_site(self, operator, site_index):
        """Applies an operator at a specific site in the chain.
        site_index should be between [0,self.N_spins] so it should take the spin into account.
        Returns: scipy.sparse.csr_matrix: Resulting sparse matrix after applying the operator.
        """
        I_left = sp.eye(2 ** (site_index - 1), format="csr")
        I_right = sp.eye(2 ** (self.N_spins - site_index), format="csr")
        return sp.kron(I_left, sp.kron(operator, I_right), format="csr")
