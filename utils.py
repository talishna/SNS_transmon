import numpy as np
from parameters import num_of_lines


def is_hermitian(matrix):
    """Custom assertion to check if a matrix is Hermitian."""
    is_square = matrix.shape[0] == matrix.shape[1]
    is_hermitian = np.allclose(matrix, matrix.conj().T)
    return is_square and is_hermitian


def matrix_element_Mfi(operator: np.ndarray, eigenvectors: np.ndarray,
                       amount: int = num_of_lines) -> np.ndarray:
    """
    Create the M_fi matrix element, for given operator and eigenvectors.
    |M_fi|^2 is proportional to the FGR transition probability from i to f.
    Parameters:
        operator (np.ndarray): Operator matrix for transitions with shape(total_dim, total_dim).
        eigenvectors (np.ndarray): Eigenvectors matrix with shape (total_dim, total_dim).
        amount (int): Number of eigenvalues and eigenvectors to consider.
    Returns:
        np.ndarray: M (amount, amount) matrix element for the interaction.
        Where M_10 is the transition from 0 to 1
    """
    M = np.zeros((amount, amount), dtype=complex)
    for i in range(amount):
        vec_i = eigenvectors[:, i]
        operator_vec_i = operator @ vec_i  # Precompute for efficiency
        for f in range(amount):
            vec_f = eigenvectors[:, f]
            M[f, i] = vec_f.conj() @ operator_vec_i

    # M = eigenvectors[:, :amount].T.conj() @ operator @ eigenvectors[:, :amount]  # better for large amount
    return M

def delta_energies(eigenenergies: np.ndarray, amount: int = num_of_lines) -> np.ndarray:
    """
    Calculate the delta energies of a given eigenenergies array.
    Parameters:
        eigenenergies (np.ndarray): 1d array of eigenenergies.
        amount (int): Number of eigenvalues and eigenvectors to consider.
    Returns:
        np.ndarray: delta_energies 2D array with shape (amount, amount), where delta_energies_fi is E_i-E_f.
        So if the eigenenergies are in ascending order the upper triangle will be positive and the lower negative.
        So delta_energies_10 is E_0-E_1
    """
    # Consider only the first 'amount' eigenenergies
    eigenenergies = eigenenergies[:amount]

    # Use broadcasting to compute the differences
    delta_energies = eigenenergies[np.newaxis, :] - eigenenergies[:, np.newaxis]

    return delta_energies


import numpy as np


def upper_triangle(matrix: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    """
    Extract the upper triangle of a 2D matrix.

    Parameters:
        matrix (np.ndarray): Input 2D array.
        include_diagonal (bool): Whether to include the diagonal elements in the output.

    Returns:
        np.ndarray: 1D array containing the upper triangular elements.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Determine the offset for the diagonal
    k = 0 if include_diagonal else 1

    # Extract the upper triangular elements
    upper_tri = matrix[np.triu_indices(matrix.shape[0], k=k)]

    return upper_tri


