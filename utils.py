import numpy as np
import time
import os
from parameters import num_of_lines
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import warnings
import scipy.sparse as sp


def is_hermitian(matrix):
    """Custom assertion to check if a matrix is Hermitian."""
    is_square = matrix.shape[0] == matrix.shape[1]
    tol = 1e-10
    if isinstance(matrix, np.ndarray):
        is_hermitian = np.allclose(matrix, matrix.conj().T, atol=tol)
    elif sp.issparse(matrix):
        is_hermitian = np.allclose(matrix.data, matrix.getH().data, atol=tol)
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


def transform_operator(operator, eigenvectors):
    """
    Transform the given operator using the provided eigenvectors.

    Parameters:
    operator (np.ndarray): The operator to be transformed. Can be 2D or 3D.
    eigenvectors (np.ndarray): The eigenvectors used for the transformation. Should match the dimensionality of n_operator.

    Returns:
    np.ndarray: The transformed operator.
    """
    if operator.ndim == 3:  # for array inputs
        transformed_operator = np.zeros_like(operator)
        for i in range(operator.shape[0]):
            transformed_operator[i, :, :] = eigenvectors[i, :, :] @ operator[i, :, :] @ eigenvectors[i, :, :].conj().T
    elif operator.ndim == 2:
        transformed_operator = eigenvectors @ operator @ eigenvectors.conj().T
    else:  # for scalar inputs
        raise ValueError("The operator must be 2D or 3D if order to do the basis transformation")
    return transformed_operator


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


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value..

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection.
        This may include the `norm` keyword for consistent color mapping.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    # Create the LineCollection object with the provided kwargs
    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def plot_x_y_color(color_values, x, y, xlabel, ylabel, title, descriptions=None, wo_small_values=True):
    """
    Plots x and y values with a color gradient based on color_values.

    Parameters:
        color_values (np.ndarray): Array of values used to color the line segments. Should match the dimensions of y.
        x (np.ndarray): Array of x values.
        y (np.ndarray): 2D array of y values. Each column represents a different line segment.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        wo_small_values (bool): If True, sets color values below 1e-14 to zero.

    Returns:
        None: Displays the plot.
    """
    # Copy to avoid modifying original color_values
    color_values = color_values.copy()

    if wo_small_values:
        color_values = np.where(color_values < 1e-14, 0, color_values)

    # linear transformation to map all color_values to the range 0 to 1
    # min_val = np.min(color_values)
    # max_val = np.max(color_values)
    # transformed_color_values = (color_values - min_val) / (max_val - min_val)
    # Combine color ranges for normalization
    norm = colors.Normalize(vmin=np.min(color_values), vmax=np.max(color_values), clip=False)
    normalized_color_values = norm(color_values)

    # debug
    print(f"Color values before normalization: min = {np.min(color_values)}, max = {np.max(color_values)}")
    print(
        f"Normalized color values: min = {np.min(normalized_color_values)}, max = {np.max(normalized_color_values)}")

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # creates a figure and axes objects. figure contains all the elements of a
    # plot - subplots,titles,labels, legends. axes is an individual plotting area within the fig, this is the plot
    # itself. fig contains the axes (subplots)

    # Define distinct legend colors
    distinct_colors = list(colors.TABLEAU_COLORS.values())  # 10 distinct colors
    # List to store legend entries
    legend_entries = []

    for i in range(color_values.shape[1]):
        points = np.array([x, y[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
        lc.set_array(normalized_color_values[:, i])
        ax.add_collection(lc)  # add the line collection to the ax

        # Get a distinct color for the legend
        legend_color = distinct_colors[i % len(distinct_colors)]
        # Add a small dot at the first point using the distinct legend color
        ax.scatter(x[0], y[0, i], color=legend_color, s=30, edgecolor='black', zorder=3, label=descriptions[i])
        # If descriptions exist, add them to the legend
        if descriptions is not None and i < len(descriptions):
            legend_entries.append(mlines.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=legend_color, markersize=6, label=descriptions[i]))

        if i == 0:
            # create the colorbar based on the first line collection to ensure it appears once
            cb = plt.colorbar(lc, ax=ax)
            cb.set_label('Dipole operator transition amplitude')
            print("Colorbar limits:", lc.norm.vmin, lc.norm.vmax)  # print colorbar limits

    # Add legend if descriptions are provided
    if descriptions:
        ax.legend(handles=legend_entries, loc="upper right", fontsize=8, frameon=True)

    plt.ylim(7, 8)
    # plt.savefig(f'{str(title)}.svg', format='svg')
    # plt.xlim(-2,2)
    ax.autoscale()  # adjusts the axis limits to fit the data in the subplot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)  # Add grid lines
    plt.show()
