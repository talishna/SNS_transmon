import numpy as np
import matplotlib.pyplot as plt
from transmon import Transmon
import os
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import math
import parameters

# Parameters
E_C = parameters.E_C
n_0_int = parameters.n_0_int
E_J_max = parameters.E_J_max
d = parameters.d
flux_0 = parameters.flux_0
size_of_transmon_subspace = parameters.size_of_transmon_subspace
plasma_energy = parameters.plasma_energy
ratio = parameters.ratio
gamma = parameters.gamma
gamma_L = parameters.gamma_L
gamma_R = parameters.gamma_R
n_0_half_int = parameters.n_0_half_int
E_C_tag = parameters.E_C_tag
F = parameters.F

steps = parameters.steps
n_g_array = parameters.n_g_array
N_g_array = parameters.N_g_array
total_dim = parameters.total_dim

# these are parameters for the plots
num_of_lines = parameters.num_of_lines
labels_one_dataset = parameters.labels_one_dataset
amount_of_energy_diff = parameters.amount_of_energy_diff

# Define range for N_g and n_g for the graph of both n_g and N_g varying
n_g_range = parameters.n_g_range
N_g_range = parameters.N_g_range


# noinspection PyTypeChecker
def h_tot_n_tot_N_g_n_g(n_g=0, N_g=0, even=False, odd=False):  # the hamiltonian of the coulomb interaction model
    transmon0 = Transmon(E_C, n_0_int, E_J_max, d, flux_0)
    H_transmon01 = transmon0.compute_hamiltonian(n_g=n_g)
    n_hat_int = transmon0.n_hat
    E_C_tag_N_g_0 = E_C_tag * N_g ** 2 * np.eye(H_transmon01.shape[0])
    E_C_tag_N_g_1 = E_C_tag * (0.5 + N_g) ** 2 * np.eye(H_transmon01.shape[0])
    H_transmon0 = H_transmon01 + E_C_tag_N_g_0 - F * n_hat_int * N_g  # this is the mat of subspace n_d = 0
    H_transmon1 = H_transmon01 + E_C_tag_N_g_1 - F * n_hat_int * (0.5 + N_g)  # this is the mat of subspace n_d = 1
    transmon2 = Transmon(E_C, n_0_half_int, E_J_max, d, flux_0)
    H_transmon2 = transmon2.compute_hamiltonian(n_g=n_g)  # this is the mat of subspace n_d = 2
    n_hat_half_int = transmon2.n_hat
    E_C_tag_N_g_2 = E_C_tag * (1 + N_g) ** 2 * np.eye(H_transmon2.shape[0])
    H_transmon2 += E_C_tag_N_g_2 - F * n_hat_half_int * (1 + N_g)
    # E_C_tag_N_g_2 = E_C_tag * (1 + N_g) ** 2 * np.eye(H_transmon01.shape[0])
    # H_transmon2 = H_transmon01 + E_C_tag_N_g_2
    # gamma_mat = gamma * (np.eye(H_transmon01.shape[0], k=1) + np.eye(H_transmon01.shape[0], k=-1))
    gamma_mat = gamma_L * np.eye(H_transmon01.shape[0]) + gamma_R * np.eye(H_transmon01.shape[0], k=-1)
    zero = np.zeros_like(H_transmon0)
    if even:
        H_tot = np.block([[H_transmon0, gamma_mat],
                          [gamma_mat.conj().T, H_transmon2]])
        # n_total = np.block([[n_hat_int, zero], [zero, n_hat_int]])
        n_total = np.block([[n_hat_int, zero],
                             [zero, n_hat_half_int]])
    elif odd:
        H_tot = np.block([[H_transmon1, zero],
                          [zero, H_transmon1]])
        n_total = np.block([[n_hat_int, zero],
                            [zero, n_hat_int]])
    else:
        H_tot = np.block([[H_transmon0, zero, zero, gamma_mat],
                          [zero, H_transmon1, zero, zero],
                          [zero, zero, H_transmon1, zero],
                          [gamma_mat, zero, zero, H_transmon2]])
        n_total = np.block([[n_hat_int, zero, zero, zero],
                            [zero, n_hat_int, zero, zero],
                            [zero, zero, n_hat_int, zero],
                            [zero, zero, zero, n_hat_half_int]])
        # n_total = np.block([[n_hat_int, zero, zero, zero],
        #                     [zero, n_hat_int, zero, zero],
        #                     [zero, zero, n_hat_int, zero],
        #                     [zero, zero, zero, n_hat_int]])

    # Check if the matrix is Hermitian
    # is_hermitian = np.allclose(n_total, n_total.conj().T)
    # print("Matrix is Hermitian:", is_hermitian)
    return H_tot, n_total


def dispersion(array, which_eigen):
    max = np.max(array[:, which_eigen])
    min = np.min(array[:, which_eigen])
    dispersion = (max - min)
    return dispersion



def plot_data_vs_x(amount, x_array, y_matrices, num_datasets, xlabel, ylabel, title, labels=None, linestyle='-'):
    """
    Plot data against x values.

    Parameters:
        amount (int): Number of lines to plot.
        x_array (array-like): Array of x values.
        y_matrices (list of 2D array-like): List of matrices of y values. Each matrix represents a set of data.
        num_datasets (int): Number of datasets to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title for the plot.
        labels (list of str, optional): Labels for each line. If None, labels will not be shown. Default is None.
        linestyle (str, optional): Linestyle for the plot. Default is '-'.
    """
    for i in range(amount):
        if num_datasets == 1:
            plt.plot(x_array, y_matrices[0][:, i], label=labels[0][i] if labels else None, linestyle=linestyle)
        elif num_datasets == 2:
            plt.plot(x_array, y_matrices[0][:, i], label=labels[0][i] if labels else None, linestyle=linestyle)
            plt.plot(x_array, y_matrices[1][:, i], linestyle='--', label=labels[1][i] if labels else None)
        else:
            raise ValueError("Number of datasets must be either 1 or 2.")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels:
        plt.legend()  # Show legend with labels
    plt.grid(True)  # Add grid lines
    plt.show()


def compute_eigenvalues_and_operators(n_g=None, N_g=None, even=False, odd=False):
    """
    Compute eigenvalues and eigenvectors of the Hamiltonian, and the n operator.

    Parameters:
    n_g (float or np.ndarray): The n_g parameter, can be a scalar or an array.
    N_g (float or np.ndarray): The N_g parameter, can be a scalar or an array.
    even (bool): Whether to consider even states.
    odd (bool): Whether to consider odd states.

    Returns:
    tuple: Eigenvalues, eigenvectors, and n operator.
    """
    # Helper function to compute the Hamiltonian and operator
    def compute_hamiltonian_and_operator(n_g, N_g, even, odd):
        H, current_n_operator = h_tot_n_tot_N_g_n_g(n_g=n_g, N_g=N_g, even=even, odd=odd)
        current_eigenvalues, current_eigenvectors = np.linalg.eigh(H)
        return current_eigenvalues, current_eigenvectors, current_n_operator, H.shape[0]

    # get the dimentions of the hamiltonian
    if isinstance(n_g, np.ndarray):
        ng_sample = n_g[0]
    else:
        ng_sample = n_g
    if isinstance(N_g, np.ndarray):
        Ng_sample = N_g[0]
    else:
        Ng_sample = N_g

    _, _, _, total_dim = compute_hamiltonian_and_operator(ng_sample, Ng_sample, even, odd)

    # Handle the case where both n_g and N_g are arrays
    if isinstance(n_g, np.ndarray) and isinstance(N_g, np.ndarray):
        eigenvalues = np.zeros((n_g.shape[0], int(total_dim)))
        eigenvectors = np.zeros((n_g.shape[0], int(total_dim), int(total_dim)))
        n_operator = np.zeros((n_g.shape[0], int(total_dim), int(total_dim)))
        for i, (ng, Ng) in enumerate(zip(n_g, N_g)):
            ev, evec, n_op, _ = compute_hamiltonian_and_operator(ng, Ng, even, odd)
            eigenvalues[i, :] = ev
            eigenvectors[i, :, :] = evec
            n_operator[i, :, :] = n_op

    # Handle the case where only n_g is an array
    elif isinstance(n_g, np.ndarray):
        eigenvalues = np.zeros((n_g.shape[0], int(total_dim)))
        eigenvectors = np.zeros((n_g.shape[0], int(total_dim), int(total_dim)))
        n_operator = np.zeros((n_g.shape[0], int(total_dim), int(total_dim)))
        for i in range(n_g.shape[0]):
            ev, evec, n_op, _ = compute_hamiltonian_and_operator(n_g[i], N_g, even, odd)
            eigenvalues[i, :] = ev
            eigenvectors[i, :, :] = evec
            n_operator[i, :, :] = n_op

    # Handle the case where only N_g is an array
    elif isinstance(N_g, np.ndarray):
        eigenvalues = np.zeros((N_g.shape[0], total_dim))
        eigenvectors = np.zeros((N_g.shape[0], total_dim, total_dim))
        n_operator = np.zeros((N_g.shape[0], int(total_dim), int(total_dim)))
        for i in range(N_g.shape[0]):
            ev, evec, n_op, _ = compute_hamiltonian_and_operator(n_g, N_g[i], even, odd)
            eigenvalues[i, :] = ev
            eigenvectors[i, :, :] = evec
            n_operator[i, :, :] = n_op

    # Handle the case where both n_g and N_g are scalars
    else:
        eigenvalues = np.zeros(int(total_dim))
        eigenvectors = np.zeros((int(total_dim), int(total_dim)))
        ev, evec, n_op, _ = compute_hamiltonian_and_operator(n_g, N_g, even, odd)
        eigenvalues[:] = ev
        eigenvectors[:, :] = evec
        n_operator = n_op

    return eigenvalues, eigenvectors, n_operator


def transform_operator(operator, eigenvectors):
    """
    Transform the given operator using the provided eigenvectors.

    Parameters:
    n_operator (np.ndarray): The operator to be transformed. Can be 2D or 3D.
    eigenvectors (np.ndarray): The eigenvectors used for the transformation. Should match the dimensionality of n_operator.

    Returns:
    np.ndarray: The transformed operator.
    """
    if operator.ndim == 3:  # for array inputs
        transformed_n_operator = np.zeros_like(operator)
        for i in range(operator.shape[0]):
            transformed_n_operator[i, :, :] = eigenvectors[i, :, :] @ operator[i, :, :] @ eigenvectors[i, :, :].conj().T
    elif operator.ndim == 2:
        transformed_n_operator = eigenvectors @ operator @ eigenvectors.conj().T
    else:  # for scalar inputs
        raise ValueError("The operator must be 2D or 3D if order to do the basis transformation")
    return transformed_n_operator


# in this code i want to create a graph of energy transitions as a result of the n operator
# in the even subspace of the transmon +dot + coulomb int. i want the line to be colored by the transition probability


def create_M_and_delta(operator: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                       amount: int = num_of_lines) -> (np.ndarray, np.ndarray):
    """
    Create the M matrix and delta_energy matrix for given operator, eigenvalues, and eigenvectors.

    Parameters:
        operator (np.ndarray): Operator matrix for transitions.
        eigenvalues (np.ndarray): Eigenvalues matrix with shape (steps, total_dim).
        eigenvectors (np.ndarray): Eigenvectors matrix with shape (steps, total_dim, total_dim).
        amount (int): Number of eigenvalues and eigenvectors to consider.

    Returns:
        tuple: M (np.ndarray) - Transition probability matrix.
               delta_energy (np.ndarray) - Energy difference matrix.
    """
    num_steps = eigenvalues.shape[0]
    delta_energy = np.zeros((num_steps, amount, amount), dtype=complex)
    M = np.zeros((num_steps, amount, amount), dtype=complex)

    for i in range(amount):
        # print("i =", i)
        for j in range(amount):
            delta_energy[:, i, j] = eigenvalues[:, i] - eigenvalues[:, j]
            for step in range(num_steps):
                vec_i = eigenvectors[step, :, i]
                vec_j = eigenvectors[step, :, j]

                # Transform the operator to the basis of current step eigenvectors
                # operator_transformed = eigenvectors[step, :, :].T.conj() @ operator[step, :, :] @ eigenvectors[step, :, :]
                M[step, i, j] = vec_i.conj().dot(operator[step, :, :].dot(vec_j))

    return M, delta_energy



def create_upper_triangle_of_3d_array(array):
    # M and delta energy are (200,7,7) mat where each layer is <i|a_up+a_down|j> i want to get only the upper triangle
    # of each layer (k=1) and make it 1D and put in a matrix of (200,21)
    new_dim = np.sum(list(range(array.shape[1])))  # array.shape[1] = amount_of_energies
    new_array = np.zeros((array.shape[0], new_dim))
    upper_triangle_indices = np.triu_indices(array.shape[1], k=1)  # gets the indices of the upper triangle
    for i in range(array.shape[0]):  # steps
        new_array[i, :] = array[i][upper_triangle_indices]
    return new_array


def plot_x_y_color(color_values, x, y, xlabel, ylabel, title, wo_small_values=True):
    """
    Plots x and y values with a color gradient based on color_values.

    Parameters:
    - color_values: Array of values used to color the line segments.
    - x: Array of x values.
    - y: Array of y values.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - wo_small_values: If True, sets color values below 1e-14 to zero.
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
    # plt.savefig(f'{str(title)}.svg', format='svg')
    # plt.xlim(-2,2)
    ax.autoscale()  # adjusts the axis limits to fit the data in the subplot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)  # Add grid lines
    plt.show()


def epsilon_m(m):
    return -(-1)**m * E_C * 2 ** (4 * m + 5) / np.math.factorial(m) * np.sqrt(E_J_max / (2 * E_C)) ** (
                m / 2 + 3 / 4) * np.exp(-np.sqrt(8 * E_J_max / E_C))  # disspersion


def arctan2_positive_radians(x, y):
    """
    Calculate the arctangent of y / x in radians, ensuring the result is between 0 and 2*pi.

    Parameters:
    x (float): The x-coordinate.
    y (float): The y-coordinate.

    Returns:
    float: The angle in radians, adjusted to be in the range [0, 2*pi].
    """
    # Calculate the arctangent in radians
    radians_result = math.atan2(y, x)

    # Ensure the result is positive (between 0 and 2*pi radians)
    positive_radians_result = (radians_result + 2 * math.pi) % (2 * math.pi)

    return positive_radians_result


def analytical_eigen_and_dipole_operator1(array, total_dim=num_of_lines, change_n_g=True):
    """
    Calculate analytical eigenvalues, eigenvectors, and dipole operators for given parameters.

    Parameters:
        array (numpy.ndarray): Array of either n_g or N_g values.
        total_dim (int): Total dimension for the eigenvalues and eigenvectors.
        change_n_g (bool): If True, the array represents n_g values; if False, it represents N_g values.

    Returns:
        eigenvalues_analytical (numpy.ndarray): Analytical eigenvalues.
        eigenvectors_analytical (numpy.ndarray): Analytical eigenvectors.
        dipole_analytical (numpy.ndarray): Analytical dipole operators.
    """
    eigenvalues_analytical = np.zeros((array.shape[0], total_dim))
    eigenvectors_analytical = np.zeros((array.shape[0], total_dim, total_dim))
    dipole_analytical = np.zeros((array.shape[0], total_dim, total_dim))

    E0 = -E_J_max + plasma_energy * (0 + 0.5) - (E_C / 12) * (0 + 0 + 3)
    E1 = -E_J_max + plasma_energy * (1 + 0.5) - (E_C / 12) * (6 + 6 + 3)
    t0 = 0.5 * epsilon_m(0)
    t1 = 0.5 * epsilon_m(1)
    g = 1

    epsilonx0 = 2 * gamma * np.exp(-(flux_0 ** 2) / 32)
    epsilonx1 = 2 * gamma * np.exp(-(flux_0 ** 2) / 32) * (1 - (flux_0 ** 2) / 16)

    for i in range(array.shape[0]):
        if change_n_g:
            n_g = array[i]
            N_g = -0.5
        else:
            n_g = 0
            N_g = array[i]

        epsilon00 = E0 + E_C_tag * (N_g + 0.5) ** 2 + E_C_tag / 4
        epsilon01 = E1 + E_C_tag * (N_g + 0.5) ** 2 + E_C_tag / 4
        epsilonz0 = t0 * np.cos(2 * np.pi * n_g) - E_C_tag * (N_g + 0.5)
        epsilonz1 = t1 * np.cos(2 * np.pi * n_g) - E_C_tag * (N_g + 0.5)
        theta0 = np.arctan2(epsilonx0, epsilonz0)
        theta1 = np.arctan2(epsilonx1, epsilonz1)

        E0plus = epsilon00 + np.sqrt(epsilonx0 ** 2 + epsilonz0 ** 2)
        E0minus = epsilon00 - np.sqrt(epsilonx0 ** 2 + epsilonz0 ** 2)
        E1plus = epsilon01 + np.sqrt(epsilonx1 ** 2 + epsilonz1 ** 2)
        E1minus = epsilon01 - np.sqrt(epsilonx1 ** 2 + epsilonz1 ** 2)

        arg = (theta1 - theta0) / 2
        current_eigenvalues = [E0minus, E0plus, E1minus, E1plus]
        current_eigenvectors = np.eye(total_dim)
        current_dipole = (g / flux_0) * np.array(
            [[0, 0, np.cos(arg), np.sin(arg)],
             [0, 0, -np.sin(arg), np.cos(arg)],
             [np.cos(arg), -np.sin(arg), 0, 0],
             [np.sin(arg), np.cos(arg), 0, 0]]
        )

        # here i switched the cos and sin looks better in the graphs of analytical vs numerical for
        # the matrix elements - but not sure why. might be the dividing by 2 in the arg
        # for the thetas
        # current_dipole = (g / flux_0) * np.array(
        #     [[0, 0, np.sin(arg), np.cos(arg)],
        #      [0, 0, np.cos(arg), np.sin(arg)],
        #      [np.sin(arg), np.cos(arg), 0, 0],
        #      [np.cos(arg), np.sin(arg), 0, 0]]
        # )


        eigenvalues_analytical[i, :] = current_eigenvalues
        eigenvectors_analytical[i, :, :] = current_eigenvectors
        dipole_analytical[i, :, :] = current_dipole

    return eigenvalues_analytical, eigenvectors_analytical, dipole_analytical


def create_M_and_delta_analytical(operator, eigenvalues, eigenvectors):
    amount_of_energies = eigenvalues.shape[1]
    # operator should be of appropriate size to eigenvectors
    delta_energy = np.zeros((eigenvalues.shape[0], amount_of_energies, amount_of_energies),
                            dtype=complex)  # contains the energy
    # differences, so in each [:,i,j] i will the diff E_i-E_j
    M = np.zeros_like(delta_energy, dtype=complex)  # contains the transition probability due to operator, here
    # i will put the probabilities. It will contain <i|operator|j> in the i'th row and j'th column

    for i in range(amount_of_energies):  # a loop that iterates from 0 to 6 including
        print("i =", i)
        for j in range(amount_of_energies):  # a loop that iterates from 0 to 6 including
            diff = (eigenvalues[:, i]
                    - eigenvalues[:, j])  # should be an array with number of rows as steps and one column
            delta_energy[:, i, j] = diff
            for step in range(eigenvalues.shape[0]):
                vec_i = eigenvectors[step, :, i]
                vec_j = eigenvectors[step, :, j]
                # operator_temp = np.dot(np.dot(np.linalg.inv(eigenvectors[step, :, :]), operator[step, :, :]),
                #                        eigenvectors[step, :, :])  # here i move the operator to the relevant basis
                M_ij = vec_i.conj().T @ operator[step, :, :] @ vec_j  # here I want to save the transition probability from j to i for each step because each
                # step have different eigenvectors
                M[step, i, j] = M_ij
    return M, delta_energy


# eigenvalues_N_g, eigenvectors_N_g, n_operator_N_g = compute_eigenvalues_and_operators(N_g=N_g_array, n_g=0)
# plot_data_vs_x(amount=num_of_lines, x_array=N_g_array, y_matrices=[eigenvalues_N_g,0], num_datasets=1,
#                xlabel=r'${N_g}$', ylabel='Energy',
#                title='Numerical \n Energy vs N_g both even and uneven subspaces n_g=0',
#                labels=[labels_one_dataset, 0])
#
#
# # these are the plots of the energy vs n_g for this system with the dispersions of each energy level
# # for N_g=0
# eigenvalues_n_g_N_g_0, eigenvectors_n_g_N_g_0, n_operator_n_g_N_g_0 = (
#     compute_eigenvalues_and_operators(N_g=0, n_g=n_g_array))
# plot_data_vs_x(amount=num_of_lines, x_array=n_g_array, y_matrices=[eigenvalues_n_g_N_g_0,0], num_datasets=1,
#                xlabel=r'${n_g}$', ylabel='Energy',
#                title='Numerical \n Energy vs n_g both even and uneven subspaces N_g=0',
#                labels=[labels_one_dataset, 0])
#
# # for N_g=-0.5
# eigenvalues_n_g_N_g_half, eigenvectors_n_g_N_g_half, n_operator_n_g_N_g_half = (
#     compute_eigenvalues_and_operators(N_g=-0.5, n_g=n_g_array))
# plot_data_vs_x(amount=num_of_lines, x_array=n_g_array, y_matrices=[eigenvalues_n_g_N_g_half,0], num_datasets=1,
#                xlabel=r'${n_g}$', ylabel='Energy',
#                title='Numerical \n Energy vs n_g both even and uneven subspaces N_g=-0.5',
#                labels=[labels_one_dataset, 0])

# # dispersion for N_g=0
# dispersion_0_N_g_0 = dispersion(array=eigenvalues_n_g_N_g_0, which_eigen=0)
# print("dispersion_0_N_g_0:", dispersion_0_N_g_0)
# dispersion_1_N_g_0 = dispersion(array=eigenvalues_n_g_N_g_0, which_eigen=1)
# print("dispersion_1_N_g_0:", dispersion_1_N_g_0)
# dispersion_2_N_g_0 = dispersion(array=eigenvalues_n_g_N_g_0, which_eigen=2)
# print("dispersion_2_N_g_0:", dispersion_2_N_g_0)
# dispersion_3_N_g_0 = dispersion(array=eigenvalues_n_g_N_g_0, which_eigen=3)
# print("dispersion_3_N_g_0:", dispersion_3_N_g_0)
#
# # dispersion for N_g=-0.5
# dispersion_0_N_g_half = dispersion(array=eigenvalues_n_g_N_g_half, which_eigen=0)
# print("dispersion_0_N_g_half:", dispersion_0_N_g_half)
# dispersion_1_N_g_half = dispersion(array=eigenvalues_n_g_N_g_half, which_eigen=1)
# print("dispersion_1_N_g_half:", dispersion_1_N_g_half)
# dispersion_2_N_g_half = dispersion(array=eigenvalues_n_g_N_g_half, which_eigen=2)
# print("dispersion_2_N_g_half:", dispersion_2_N_g_half)
# dispersion_3_N_g_half = dispersion(array=eigenvalues_n_g_N_g_half, which_eigen=3)
# print("dispersion_3_N_g_half:", dispersion_3_N_g_half)


# Data numerical:

# Both n_g and N_g change even subspace
eigenvalues_both_even, eigenvectors_both_even, n_operator_both_even = (
    compute_eigenvalues_and_operators(n_g=n_g_range, N_g=N_g_range, even=True))
M_numerical_both, delta_energy_numerical_both = (
    create_M_and_delta(operator=n_operator_both_even, eigenvalues=eigenvalues_both_even,
                       eigenvectors=eigenvectors_both_even, amount=num_of_lines))
M_numerical_both = np.abs(M_numerical_both) ** 2
delta_energy_numerical_both = np.abs(delta_energy_numerical_both)
unravel_M_numerical_both = create_upper_triangle_of_3d_array(M_numerical_both)
unravel_delta_energy_numerical_both = create_upper_triangle_of_3d_array(delta_energy_numerical_both)

# n_g varies and N_g=-0.5 even subspace
eigenvalues_n_g_N_g_half_even, eigenvectors_n_g_N_g_half_even, n_operator_n_g_N_g_half_even = (
    compute_eigenvalues_and_operators(n_g=n_g_array, N_g=-0.5, even=True))
M_numerical_n_g_N_g_half, delta_energy_numerical_n_g_N_g_half = (
    create_M_and_delta(operator=n_operator_n_g_N_g_half_even, eigenvalues=eigenvalues_n_g_N_g_half_even,
                       eigenvectors=eigenvectors_n_g_N_g_half_even, amount=num_of_lines))
M_numerical_n_g_N_g_half = np.abs(M_numerical_n_g_N_g_half)
delta_energy_numerical_n_g_N_g_half = np.abs(delta_energy_numerical_n_g_N_g_half)
unravel_M_numerical_n_g_N_g_half = create_upper_triangle_of_3d_array(M_numerical_n_g_N_g_half)
unravel_delta_energy_numerical_n_g_N_g_half = create_upper_triangle_of_3d_array(delta_energy_numerical_n_g_N_g_half)

# n_g=0 and N_g varies even subspace
eigenvalues_N_g_n_g_0_even, eigenvectors_N_g_n_g_0_even, n_operator_N_g_n_g_0_even = (
        compute_eigenvalues_and_operators(n_g=0, N_g=N_g_array, even=True))
M_numerical_N_g_n_g_0, delta_energy_numerical_N_g_n_g_0 = (
    create_M_and_delta(operator=n_operator_N_g_n_g_0_even, eigenvalues=eigenvalues_N_g_n_g_0_even,
                       eigenvectors=eigenvectors_N_g_n_g_0_even, amount=num_of_lines))
M_numerical_N_g_n_g_0 = np.abs(M_numerical_N_g_n_g_0)
delta_energy_numerical_N_g_n_g_0 = np.abs(delta_energy_numerical_N_g_n_g_0)
unravel_M_numerical_N_g_n_g_0 = create_upper_triangle_of_3d_array(M_numerical_N_g_n_g_0)
unravel_delta_energy_numerical_N_g_n_g_0 = create_upper_triangle_of_3d_array(delta_energy_numerical_N_g_n_g_0)

# Data analytical:

# n_g varies and N_g=-0.5 even subspace
eigenvalues_analytical_N_g_half, eigenvectors_analytical_N_g_half, dipole_analytical_N_g_half = (
    analytical_eigen_and_dipole_operator1(array=n_g_array, change_n_g=True))
M_analytical_N_g_half, delta_energy_analytical_N_g_half = create_M_and_delta_analytical(operator=dipole_analytical_N_g_half,
                                                                          eigenvalues=eigenvalues_analytical_N_g_half,
                                                                          eigenvectors=eigenvectors_analytical_N_g_half)
M_analytical_N_g_half = np.abs(M_analytical_N_g_half)**2
delta_energy_analytical_N_g_half = np.abs(delta_energy_analytical_N_g_half)
unravel_M_analytical_N_g_half = create_upper_triangle_of_3d_array(M_analytical_N_g_half)
unravel_delta_energy_analytical_N_g_half = create_upper_triangle_of_3d_array(delta_energy_analytical_N_g_half)

# n_g=0 and N_g varies even subspace
eigenvalues_analytical_n_g_0, eigenvectors_analytical_n_g_0, dipole_analytical_n_g_0 = (
    analytical_eigen_and_dipole_operator1(array=N_g_array, change_n_g=False))
M_analytical_n_g_0, delta_energy_analytical_n_g_0 = create_M_and_delta_analytical(operator=dipole_analytical_n_g_0,
                                                                          eigenvalues=eigenvalues_analytical_n_g_0,
                                                                          eigenvectors=eigenvectors_analytical_n_g_0)
M_analytical_n_g_0 = np.abs(M_analytical_n_g_0)**2
delta_energy_analytical_n_g_0 = np.abs(delta_energy_analytical_n_g_0)
unravel_M_analytical_n_g_0 = create_upper_triangle_of_3d_array(M_analytical_n_g_0)
unravel_delta_energy_analytical_n_g_0 = create_upper_triangle_of_3d_array(delta_energy_analytical_n_g_0)

if __name__ == "__main__":



    # in this code i want to create a graph of energy transitions as a result of the n operator
    # in the even subspace of the transmon +dot + coulomb int for N_g=-0.5
    # I want the line to be colored by the transition amplitude

    plot_x_y_color(color_values=unravel_M_numerical_n_g_N_g_half, x=n_g_array, y=unravel_delta_energy_numerical_n_g_N_g_half,
                   xlabel=r'$n_g$', ylabel='Energy differences',
                   title='Numerical \n Energy diff vs n_g even subspace N_g=-0.5')

    # same as a function of N_g
    plot_x_y_color(unravel_M_numerical_N_g_n_g_0, N_g_array, unravel_delta_energy_numerical_N_g_n_g_0, xlabel=r'$N_g$',
                   ylabel='Energy differences',
                   title='Numerical \n Energy diff vs N_g even subspace n_g=0')


    # same as a function of N_g when both N_g and n_g change
    plot_x_y_color(unravel_M_numerical_both, N_g_range, unravel_delta_energy_numerical_both, xlabel=r'$N_g$',
                   ylabel='Energy differences',
                   title='Numerical \n Energy diff vs N_g even subspace varying n_g')


    # Numerical and Analytical together
    # Analytical data
    # now the colored plots for analytical energy vs n_g Even subspace only with N_g=-0.5:
    plot_x_y_color(unravel_M_analytical_N_g_half, n_g_array, unravel_delta_energy_analytical_N_g_half, xlabel=r'$n_g$',
                   ylabel='Energy differences',
                   title='Analytical \n Energy diff vs n_g even subspace N_g=-0.5')


    # now the colored plots for analytical energy vs N_g Even subspace only with n_g=0:
    plot_x_y_color(unravel_M_analytical_n_g_0, n_g_array, unravel_delta_energy_analytical_n_g_0, xlabel=r'$N_g$',
                   ylabel='Energy differences',
                   title='Analytical \n Energy diff vs N_g even subspace n_g=0')

