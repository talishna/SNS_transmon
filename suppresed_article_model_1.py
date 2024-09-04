import numpy as np
import matplotlib.pyplot as plt
from transmon import Transmon
import math
from suppresed_our_model import create_M_and_delta
from suppresed_our_model import create_upper_triangle_of_3d_array
from suppresed_our_model import plot_x_y_color
import parameters
from scipy.optimize import brentq
a = 5

# page 167 in my notability note - this is the model from the suppressed paper eq (2)
# in this code I want to create the hamiltonian for the system in the charge basis for each n_g, diagonalize it, plot
# the energy diff as a function of n_g and paint it by the dipole's transition probability

# Fit parameters from article
# Charging energy Ec (GHz), Res. 1 Gamma/2 (GHz), Res. 2 Gamma/2 (GHz)
# 5.393879300028895374e-01 3.602147184310360473e+01 2.983756389878811177e+01

# Parameters
E_C = parameters.E_C
n_0_int = parameters.n_0_int
# delta_tilda = 1
# r = 1
max_n_g = 10
n_g_array = np.linspace(-max_n_g, max_n_g, 800)
xr = 5

# Parameters from the article
E_C = 5.393879300028895374e-01  # from article
gap = 45  # from article
Gamma = 3.602147184310360473e+01  # from article

# these are parameters for the plots
num_of_lines = 4  # parameters.num_of_lines


def has_non_zero_complex_part(matrix):
    """
    Check if a matrix has any non-zero complex part.

    Args:
        matrix (ndarray): Input matrix.

    Returns:
        bool: True if any element has a non-zero imaginary part, False otherwise.
    """
    return np.any(np.imag(matrix) != 0)


def bound_state_equation(x, phi, gamma1, gamma2, gap, xr):  # Eq. 1 in the paper
    """
    Bound state equation as defined in the paper.

    Args:
        x (float): Energy value.
        phi (float): Phase value.
        gamma1 (float): Coupling parameter gamma1.
        gamma2 (float): Coupling parameter gamma2.
        gap (float): Gap parameter.
        xr (float): Epsilon_r parameter.

    Returns:
        float: Value of the bound state equation.
    """
    gamma = gamma1 + gamma2
    omega = ((gap**2 - x**2) * (x**2 - xr**2 - gamma**2/4) + gap**2 * gamma1 * gamma2 * np.sin(phi/2)**2)
    return omega + gamma * x**2 * np.sqrt(gap**2 - x**2)


def T_BW(gamma1, gamma2, xr):  # extract transmission based on \epsilon_r and Gammas
    """
    Extract transmission based on epsilon_r and Gammas.

    Args:
        gamma1 (float): Coupling parameter gamma1.
        gamma2 (float): Coupling parameter gamma2.
        xr (float): Epsilon_r parameter.

    Returns:
        float: Transmission value.
    """
    T = gamma1*gamma2/(xr**2+1/4*(gamma1+gamma2)**2)
    return T


def Delta_tilde(gamma1, gamma2, gap, xr):
    """
    Solves Eq. 1 at zero phase to find E(0) = Delta_tilde for given Gammas, gap, and epsilon_r.

    Args:
        gamma1 (float): Coupling parameter gamma1.
        gamma2 (float): Coupling parameter gamma2.
        gap (float): Gap parameter.
        xr (float): Epsilon_r parameter.

    Returns:
        float: Delta_tilde value.
    """

    # Define the function to find the root of, rescaling parameters by the gap
    def func(x):
        return bound_state_equation(x, 0, gamma1 / gap, gamma2 / gap, 1, xr / gap)

    # Use Brent's method to find the root of the function in the interval (0, 1-1e-3)
    sol = brentq(func, 0, 1 - 1e-3, full_output=False) * gap

    # Return the solution scaled by the gap
    return sol


# noinspection PyTypeChecker
def hamiltonian_and_n_operator(E_C=E_C, n_0_int=n_0_int, n_g=0, gamma1=Gamma,
                               gamma2=Gamma, gap=gap, xr=xr):
    """
    Creates the Hamiltonian and the n operator for the system.

    Args:
        E_C (float): Charging energy.
        n_0_int (int): Integer value of n_0.
        n_g (float): Value of n_g.
        gamma1 (float): Coupling parameter gamma1.
        gamma2 (float): Coupling parameter gamma2.
        gap (float): Gap parameter.
        xr (float): Epsilon_r parameter.

    Returns:
        tuple: Hamiltonian matrix and n operator matrix.
    """
    delta_tilda = Delta_tilde(gamma1, gamma2, gap, xr)
    r = np.sqrt(1 - T_BW(gamma1, gamma2, xr))
    transmon_0 = Transmon(E_C, n_0_int, 0)
    H_0 = transmon_0.compute_hamiltonian(n_g=n_g)
    H_0 = H_0 + 1j * np.zeros_like(H_0)  # Cast to complex for the cos
    cos_phi_half = transmon_0.compute_cos_phi_half()
    # print("Does cos have complex part?:", has_non_zero_complex_part(cos_phi_half))
    sin_phi_half = transmon_0.compute_sin_phi_half()
    B1 = H_0 + delta_tilda * cos_phi_half
    B2 = delta_tilda * r * sin_phi_half
    B3 = delta_tilda * r * sin_phi_half
    B4 = H_0 - delta_tilda * cos_phi_half
    H = np.block([[B1, B2],
                  [B3, B4]])
    zero_mat = np.zeros_like(H_0)
    n = np.block([[transmon_0.n_hat, zero_mat],
                  [zero_mat, transmon_0.n_hat]])
    return H, n

def compute_eigenvalues_and_operators(n_g=None):
    """
    Compute eigenvalues and eigenvectors of the Hamiltonian, and the n operator.

    Parameters:
    n_g (float or np.ndarray): The n_g parameter, can be a scalar or an array.

    Returns:
    tuple: Eigenvalues, eigenvectors, and n operator.
    """
    # Helper function to compute the Hamiltonian and operator
    def compute_hamiltonian_and_operator(n_g):
        H, current_n_operator = hamiltonian_and_n_operator(n_g=n_g)
        current_eigenvalues, current_eigenvectors = np.linalg.eigh(H)
        return current_eigenvalues, current_eigenvectors, current_n_operator, H.shape[0]

    # get the dimensions of the hamiltonian
    if isinstance(n_g, np.ndarray):
        ng_sample = n_g[0]
    else:
        ng_sample = n_g

    _, _, _, total_dim = compute_hamiltonian_and_operator(ng_sample)

    # Handle the case where n_g is an array
    if isinstance(n_g, np.ndarray):
        eigenvalues = np.zeros((n_g.shape[0], int(total_dim)), dtype=complex)
        eigenvectors = np.zeros((n_g.shape[0], int(total_dim), int(total_dim)), dtype=complex)
        n_operator = np.zeros((n_g.shape[0], int(total_dim), int(total_dim)), dtype=complex)
        for i in range(n_g.shape[0]):
            ev, evec, n_op, _ = compute_hamiltonian_and_operator(n_g[i])
            eigenvalues[i, :] = ev
            eigenvectors[i, :, :] = evec
            n_operator[i, :, :] = n_op

    # Handle the case where n_g is scalar
    else:
        eigenvalues = np.zeros(int(total_dim), dtype=complex)
        eigenvectors = np.zeros((int(total_dim), int(total_dim)), dtype=complex)
        ev, evec, n_op, _ = compute_hamiltonian_and_operator(n_g)
        eigenvalues[:] = ev
        eigenvectors[:, :] = evec
        n_operator = n_op

    return eigenvalues, eigenvectors, n_operator


def dispersion_per_interval(delta_E, intervals):
    """
    Calculate the dispersion for each interval of delta_E.

    Args:
        delta_E (ndarray): Energy differences.
        intervals (int): Number of intervals to divide delta_E into.

    Returns:
        ndarray: Dispersion values for each interval.
    """
    num_of_point = math.floor(delta_E.shape[0] / intervals)
    dispersion = np.zeros(intervals)
    for i in range(intervals):
        delta_interval = delta_E[i * num_of_point:(i + 1)*num_of_point]
        dispersion[i] = np.max(delta_interval)-np.min(delta_interval)
    return dispersion


def average_per_interval(a, intervals):
    """
    Calculate the average for each interval of array a.

    Args:
        a (ndarray): Input array.
        intervals (int): Number of intervals to divide the array into.

    Returns:
        ndarray: Average values for each interval.
    """
    num_of_point = math.floor(a.shape[0] / intervals)
    average = np.zeros(intervals)
    for i in range(intervals):
        a_interval = a[i * num_of_point:(i + 1) * num_of_point]
        average[i] = np.average(a_interval)
    return average

if __name__ == "__main__":

    H, n = hamiltonian_and_n_operator()
    print("Does H have complex part?:", has_non_zero_complex_part(H))
    print("Does n have complex part?:", has_non_zero_complex_part(n))
    eigenvalues, eigenvectors, n_operator = compute_eigenvalues_and_operators(n_g=n_g_array)
    print("Does eigenvalues have complex part?:", has_non_zero_complex_part(eigenvalues))

    M, delta_E = create_M_and_delta(operator=n_operator, eigenvalues=eigenvalues, eigenvectors=eigenvectors, amount=num_of_lines)
    M, delta_E = np.abs(M)**2, np.abs(delta_E)
    M_from_gs = M[:, 0, 1:]  # taking only the transitions from the GS and excluding the gs->gs "transition"
    delta_E_from_gs = delta_E[:, 0, 1:]  # taking only the transitions from the GS and excluding the gs->gs "transition"  # create_upper_triangle_of_3d_array(delta_E)
    plot_x_y_color(color_values=M_from_gs, x=n_g_array, y=delta_E_from_gs, xlabel=r"$n_g$", ylabel=r"$\Delta_E$",
                   title="Suppressed their model 1 - from GS")

    M_all, delta_E_all = create_upper_triangle_of_3d_array(M), create_upper_triangle_of_3d_array(delta_E)
    plot_x_y_color(color_values=M_all, x=n_g_array, y=delta_E_all, xlabel=r"$n_g$", ylabel=r"$\Delta_E$",
                   title="Suppressed their model 1 - all")

    intervals = 2 * max_n_g
    dispersion_array = dispersion_per_interval(delta_E_from_gs, intervals)
    qubit_energy = average_per_interval(delta_E_from_gs, intervals)
    plt.plot(qubit_energy, dispersion_array)
    plt.xlabel("qubit energy")
    plt.ylabel("dispersion")
    plt.title("dispersion vs qubit energy - from gs only")
    plt.grid(True)
    plt.show()

    intervals = 2*max_n_g
    dispersion_array = dispersion_per_interval(delta_E_all, intervals)
    qubit_energy = average_per_interval(delta_E_all, intervals)
    plt.plot(qubit_energy, dispersion_array)
    plt.xlabel("qubit energy")
    plt.ylabel("dispersion")
    plt.title("dispersion vs qubit energy - all energies")
    plt.grid(True)
    plt.show()
