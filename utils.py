import numpy as np
import time
import os
from parameters import num_of_lines
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection


def is_hermitian(matrix):
    """Custom assertion to check if a matrix is Hermitian."""
    is_square = matrix.shape[0] == matrix.shape[1]
    is_hermitian = np.allclose(matrix, matrix.conj().T)
    return is_square and is_hermitian


def matrix_element_Mfi(operator: np.ndarray, eigenvectors: np.ndarray,
                       amount: int = num_of_lines) -> np.ndarray:
    """
    Create the M_fi=<v_f|M|V_i> matrix element, for given operator and eigenvectors.
    |M_fi|^2 is proportional to the FGR transition probability from i to f.
    Parameters:
        operator (np.ndarray): Operator matrix for transitions with shape(total_dim, total_dim).
        eigenvectors (np.ndarray): Eigenvectors matrix with shape (total_dim, total_dim).
        amount (int): Number of eigenvalues and eigenvectors to consider.
    Returns:
        np.ndarray: M (amount, amount) matrix element for the interaction.
        Where M_10=<v_1|M|v_0> is the transition from 0 to 1
    """
    # M = np.zeros((amount, amount), dtype=complex)
    # for i in range(amount):
    #     vec_i = eigenvectors[:, i]
    #     operator_vec_i = operator @ vec_i  # Precompute for efficiency
    #     for f in range(amount):
    #         vec_f = eigenvectors[:, f]
    #         M[f, i] = vec_f.conj() @ operator_vec_i

    M2 = eigenvectors[:, :amount].T.conj() @ operator @ eigenvectors[:, :amount]  # better for large amount
    # print("are M and M2 close?" + str(np.allclose(M, M2, 1e-5, 1e-8)))  # check if M and M2 are close
    return M2

def delta_energies(eigenenergies: np.ndarray, amount: int = num_of_lines) -> np.ndarray:
    """
    Calculate the delta energies of a given eigenenergies array.
    Parameters:
        eigenenergies (np.ndarray): 1d array of eigenenergies.
        amount (int): Number of eigenvalues and eigenvectors to consider.
    Returns:
        np.ndarray: delta_energies 2D array with shape (amount, amount), where delta_energies_fi is E_f-E_i.
        So if the eigenenergies are in ascending order the upper triangle will be positive and the lower negative.
        So delta_energies_10 is E_1-E_0 and delta_energies_01 = E0-E1
    """
    # Consider only the first 'amount' eigenenergies
    eigenenergies = eigenenergies[:amount]

    # Use broadcasting to compute the differences
    delta_energies = eigenenergies[:, np.newaxis] - eigenenergies[np.newaxis, :]

    return delta_energies



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

def plot_x_y_color(color_values, x, y, xlabel, ylabel, title, path = None, wo_small_values=True):
    """
    Plots x and y values with a color gradient based on color_values.

    Parameters:
        color_values (np.ndarray): Array of values used to color the line segments. Should match the second dimension of y.
        x (np.ndarray): Array of x values.
        y (np.ndarray): 2D array of y values. Each column represents a different line segment.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        wo_small_values (bool): If True, sets color values below 1e-14 to zero.

    Returns:
        None: Displays the plot.
    """

    if wo_small_values:
        color_values = np.where(color_values < 1e-14, 0, color_values)

    # linear transformation to map all color_values to the range 0 to 1
    # min_val = np.min(color_values)
    # max_val = np.max(color_values)
    # transformed_color_values = (color_values - min_val) / (max_val - min_val)

    norm = colors.Normalize(vmin=np.min(color_values), vmax=np.max(color_values), clip=False)
    normalized_color_values = norm(color_values)

    # debug
    print(f"Color values before normalization: min = {np.min(color_values)}, max = {np.max(color_values)}")
    print(
        f"Normalized color values: min = {np.min(normalized_color_values)}, max = {np.max(normalized_color_values)}")

    cmap = plt.get_cmap('seismic')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # creates a figure and axes objects. figure contains all the elements of a
    # plot - subplots,titles,labels, legends. axes is an individual plotting area within the fig, this is the plot
    # itself. fig contains the axes (subplots)

    for i in range(color_values.shape[1]):
        points = np.array([x, y[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
        lc.set_array(normalized_color_values[:, i])
        ax.add_collection(lc)  # add the line collection to the ax

        if i == 0:
            # create the colorbar based on the first line collection to ensure it appears once
            cb = plt.colorbar(lc, ax=ax)
            cb.set_label('Dipole operator transition amplitude')
            print("Colorbar limits:", lc.norm.vmin, lc.norm.vmax)  # print colorbar limits


    plt.ylim(7,8)
    ax.autoscale()  # adjusts the axis limits to fit the data in the subplot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)  # Add grid lines
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if path:
        plot_filename = os.path.join(str(path), f'{str(title)}_{timestamp}.svg')
        plt.savefig(plot_filename, format='svg')
    else:
        plt.savefig(f'{str(title)}_{timestamp}.svg', format='svg')
