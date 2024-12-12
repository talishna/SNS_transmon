import numpy as np

def is_hermitian(matrix):
    """Custom assertion to check if a matrix is Hermitian."""
    is_square = matrix.shape[0] == matrix.shape[1]
    is_hermitian = np.allclose(matrix, matrix.conj().T)
    return is_square and is_hermitian
