import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import kron, eye, csr_matrix, diags, spdiags, kronsum
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig, eigh
import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from line_profiler_pycharm import profile

from transmon import Transmon
from spinchain import SpinChain
from utils import is_hermitian, matrix_element_Mfi, delta_energies, upper_triangle, plot_x_y_color, colored_line, transform_operator


# Parameters
# Transmon
E_C = 1  # charging energy
n_0_int_L = 6  # 10 # number of CP on the left
n_0_int_R = 6  # 10 # number of CP on the right
E_J = 30
E_J_max = 10  # (Wq+E_C)**2/8/E_C
d = 0  # 0.35 #squid asymmetry
flux_0 = 1  # 2.067833 * 10 ** (-15)
transmon_subspace_size = 6
plasma_energy = np.sqrt(8 * E_C * E_J_max)
ratio = E_J / E_C
gamma = 0.01
gamma_L = 0.01
gamma_R = 0.01
steps = 200
n_g_array = np.linspace(-2, 2, steps)
flux_array = np.linspace(0, flux_0, steps)
number_of_energies = 6

# SpinChain
N_sites = 2  # number of sites in the chain to be spins it's times 2 (handled in the code for the spinchain
t = 0.1
epsilon_r = 0
size_of_spinchain_subspace = 4

# basic objects
transmon_L = Transmon(E_C, n_0_int_L)
transmon_R = Transmon(E_C, n_0_int_L)
spinchain = SpinChain(N_sites, t, epsilon_r)



def diagonalize_and_truncate(H, size_of_subspace):
    """
        Diagonalizes a Hermitian matrix and truncates it to retain the lowest eigenvalues.

        Args:
            H (ndarray): The Hermitian matrix (Hamiltonian) to diagonalize. Must be a square 2D NumPy array.
            size_of_subspace (int): The number of lowest eigenvalues (and corresponding eigenvectors) to retain.

        Returns:
            tuple:
                - eigenvalues (ndarray): All eigenvalues of the input matrix, sorted in ascending order.
                - U (ndarray): The unitary matrix where columns are eigenvectors of the input matrix.
                - H_truncated (ndarray): A diagonal matrix containing the smallest `size_of_subspace` eigenvalues.

        Notes:
            - The input matrix `H` must be Hermitian. If it is not, the behavior of the function is undefined.
            - The truncated Hamiltonian is returned as a diagonal matrix, not in its original basis.

        """
    if sparse.issparse(H):
        eigenvalues, U = eigs(H)  # note that U is not sparse
        H_truncated = diags(eigenvalues[:size_of_subspace], offsets=0, format='csr')
    else:
        eigenvalues, U = np.linalg.eigh(H)
        H_truncated = np.diag(eigenvalues[:size_of_subspace])
    return eigenvalues, U, H_truncated



@profile
def double_transmon_chain_interaction(E_J=E_J, flux=0, flux_0=flux_0, gamma_L=gamma_L, gamma_R=gamma_R,
                                      transmon_L=transmon_L, transmon_R=transmon_R, spinchain=spinchain,
                                      transmon_subspace_size_L=None, n_g=0):
    #  hamiltonians
    H_L = transmon_L.compute_hamiltonian(n_g=n_g)
    H_R = transmon_R.compute_hamiltonian(n_g=-n_g)
    H_spinchain = spinchain.compute_hamiltonian()  # this is already sparse
    n_L = transmon_L.n_hat
    n_R = transmon_R.n_hat
    n_tot = np.kron(n_L, np.eye(n_R.shape[0])) - np.kron(np.eye(n_L.shape[0]), n_R)

    #  operators
    e_phi_L = transmon_L.creation
    e_phi_R = transmon_R.creation
    s_minus = spinchain.s_minus

    flux_exp_plus = np.exp(1j * np.pi * flux / (flux_0))
    flux_exp_minus = np.exp(-1j * np.pi * flux / (flux_0))
    flux_exp_plus_half = np.exp(1j * np.pi * flux / (2 * flux_0))
    flux_exp_minus_half = np.exp(-1j * np.pi * flux / (2 * flux_0))
    J_term = -(E_J / 2) * (flux_exp_minus * np.kron(e_phi_L, e_phi_R.conj().T) +
                           flux_exp_plus * np.kron(e_phi_L.conj().T, e_phi_R))
    H_transmon = np.kron(H_L, np.eye(H_R.shape[0])) + np.kron(np.eye(H_L.shape[0]), H_R) + J_term
    e_phi_L = np.kron(e_phi_L, np.eye(int(H_R.shape[0])))
    e_phi_R = np.kron(np.eye(int(H_L.shape[0])), e_phi_R)

    if transmon_subspace_size_L:
        _, U, H_transmon = diagonalize_and_truncate(H_transmon, transmon_subspace_size_L)
        e_phi_L = transform_operator(e_phi_L, U)[:transmon_subspace_size_L, :transmon_subspace_size_L]
        e_phi_R = transform_operator(e_phi_R, U)[:transmon_subspace_size_L, :transmon_subspace_size_L]
        n_tot = transform_operator(n_tot, U)[:transmon_subspace_size_L, :transmon_subspace_size_L]

    H_transmon = csr_matrix(H_transmon)
    e_phi_L = csr_matrix(e_phi_L)
    e_phi_R = csr_matrix(e_phi_R)
    L_term = (flux_exp_plus_half * kron(e_phi_L,
                                        kron(s_minus,
                                             kron(s_minus, eye(int(H_spinchain.shape[0] * 2 ** (-2)), format='csr')))))
    R_term = (flux_exp_minus_half * kron(e_phi_R,
                                         kron(eye(int(H_spinchain.shape[0] * 2 ** (-2)), format='csr'),
                                              kron(s_minus, s_minus))))
    H_int = gamma_L * (L_term + L_term.getH()) + gamma_R * (R_term + R_term.getH())
    H = (kron(H_transmon, eye(H_spinchain.shape[0], format='csr')) +
         kron(eye(H_transmon.shape[0], format='csr'), H_spinchain) + csr_matrix(H_int))
    return H, np.kron(n_tot, np.eye(H_spinchain.shape[0]))


if __name__ == '__main__':


    # def eigen_per_flux():
    #     H0, _ = double_transmon_chain_interaction(flux=0, transmon_subspace_size_L=transmon_subspace_size)
    #     dimension = H0.shape[0]
    #     # Initialize arrays to store eigenstate energies and eigenvectors
    #     eigenvectors_h = np.zeros((len(flux_array), dimension, number_of_energies), dtype=complex)
    #     eigenvalues_h = np.zeros((len(flux_array), number_of_energies), dtype=float)
    #     n_h = np.zeros((len(flux_array), dimension, dimension), dtype=complex)
    #     v0 = None  # Store eigenvectors for reuse
    #
    #     # Start timing
    #     start_time = time.time()
    #
    #     # Calculate eigenstates for each flux value
    #     for i, flux in enumerate(flux_array):
    #         # Recalculate H for the current flux value
    #         H, n_tot = double_transmon_chain_interaction(flux=flux, transmon_subspace_size_L=transmon_subspace_size)
    #
    #         # Calculate eigenstates and eigenvalues
    #         try:
    #             if i == 0:
    #                 eigenvals, eigenvecs = sparse.linalg.eigsh(H, k=number_of_energies, which='SM')
    #             else:
    #                 eigenvals, eigenvecs = sparse.linalg.eigsh(H, k=number_of_energies, which='SM', v0=v0)
    #
    #         except sparse.linalg.ArpackNoConvergence:
    #             print(f"ARPACK did not converge at flux={flux}. Using the last v0.")
    #             v0 = eigenvectors_h[max(i-1, 0), :, 0]  # Use the last known eigenvector
    #             eigenvals, eigenvecs = sparse.linalg.eigs(H, k=number_of_energies, which='SM', v0=v0)
    #
    #         # Store the first x eigenstate energies
    #         sorted_indices = np.argsort(eigenvals.real)
    #         sorted_eigenvals = eigenvals[sorted_indices]
    #         sorted_eigenvecs = eigenvecs[:, sorted_indices]
    #         eigenvalues_h[i, :] = sorted_eigenvals
    #         eigenvectors_h[i, :, :] = sorted_eigenvecs
    #         v0 = np.mean(sorted_eigenvecs[:, :min(2, sorted_eigenvecs.shape[1])], axis=1)
    #         n_h[i, :, :] = n_tot
    #         isherm = is_hermitian(H)
    #         print(f"i={i}, flux={flux}, isherm={isherm}")
    #
    #     # End timing
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Loop finished in {elapsed_time:.2f} seconds.")
    #     return eigenvalues_h, eigenvectors_h, n_h
    #
    #
    # eigenvalues_flux, eigenvectors_flux, n_operator_flux = eigen_per_flux()
    # np.save("eigenvalues_SNS_double_transmon_chain_flux.npy", eigenvalues_flux)
    # np.save("eigenvectors_SNS_double_transmon_chain_flux.npy", eigenvectors_flux)
    # eigenvalues_flux = np.load("eigenvalues_SNS_double_transmon_chain_flux.npy")
    # eigenvectors_flux = np.load("eigenvectors_SNS_double_transmon_chain_flux.npy")
    #
    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # for i in range(number_of_energies):
    #     plt.plot(flux_array, eigenvalues_flux[:, i], label=f'Eigenstate {i}')
    #
    # plt.xlabel('Flux (Φ/Φ0)')
    # plt.ylabel('Energy')
    # plt.title(f'First {number_of_energies} Eigenstates vs Flux')
    # plt.legend()
    # plt.grid(True)
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # plot_filename = os.path.join("SNS_double_transmon_chain_plots",
    #                              f"Transmon energies vs flux {timestamp}.png")
    # plt.savefig(plot_filename, format='png')
    # plt.show()
    #
    #
    # def plot_delta_energy_vs_flux_dipole_transitions():
    #     length = np.sum(range(number_of_energies))
    #     big_M = np.zeros((flux_array.shape[0], length), dtype=complex)
    #     big_delta = np.zeros_like(big_M)
    #
    #     # compute M and delta for each flux
    #     for i, flux in enumerate(flux_array):
    #         M = eigenvectors_flux[i, :, :].T.conj() @ n_operator_flux[i, :, :] @ eigenvectors_flux[i, :, :]
    #         delta = eigenvalues_flux[i, :, np.newaxis] - eigenvalues_flux[i, np.newaxis, :]
    #         big_M[i, :] = upper_triangle(M)
    #         big_delta[i, :] = upper_triangle(delta)
    #
    #     big_delta = -1 * big_delta  # minus so the energies will be positive
    #     big_M = abs(big_M) ** 2
    #
    #     # Combine color ranges for normalization
    #     vmin = big_M.min()
    #     vmax = big_M.max()
    #     norm = Normalize(vmin=vmin, vmax=vmax)
    #
    #     # Create the plot
    #     fig, ax = plt.subplots()
    #
    #     # create the description for the graph from the upper triangle indices
    #     upper_tri = np.triu_indices(number_of_energies, k=1)
    #     indices = [(y, x) for x, y in zip(*upper_tri)]
    #     descriptions = [f'E{x}-E{y}' for x, y in indices]
    #
    #     for i in range(length):
    #         line = colored_line(flux_array, big_delta[:, i], big_M[:, i], ax, linewidth=2, cmap="plasma", norm=norm)
    #
    #         # annotate the line with its description
    #         mid_idx = len(flux_array) // 2
    #         ax.text(flux_array[mid_idx], big_delta[mid_idx, i], descriptions[i], fontsize=8, color="black", ha="left")
    #
    #         # Add a single colorbar
    #         if i == (length - 1):
    #             fig.colorbar(line, ax=ax)
    #
    #     ax.set_title("Transmon Transitions due to Dipole")
    #     ax.set_xlabel("Flux (Φ/Φ0)")
    #     ax.set_ylabel("Delta Energy")
    #     ax.set_xlim(flux_array.min(), flux_array.max())
    #     ax.set_ylim(big_delta.min(), big_delta.max())
    #     plt.tight_layout()
    #
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     plot_filename = os.path.join("SNS_double_transmon_chain_plots",
    #                                  f"Transmon Transitions vs flux due to Dipole {timestamp}.png")
    #     plt.savefig(plot_filename, format='png')
    #     plt.show()
    #
    # plot_delta_energy_vs_flux_dipole_transitions()
    #
    #
    # # now for n_g
    #
    # def eigen_per_n_g():
    #     H0, _ = double_transmon_chain_interaction(flux=0, transmon_subspace_size_L=transmon_subspace_size)
    #     dimension = H0.shape[0]
    #     # Initialize arrays to store eigenstate energies and eigenvectors
    #     eigenvectors_n_g = np.zeros((len(n_g_array), dimension, number_of_energies), dtype=complex)
    #     eigenvalues_n_g = np.zeros((len(n_g_array), number_of_energies), dtype=float)
    #     n_n_g = np.zeros((len(n_g_array), dimension, dimension), dtype=complex)
    #     v0 = None  # Store eigenvectors for reuse
    #
    #     # Start timing
    #     start_time = time.time()
    #
    #     # Calculate eigenstates for each flux value
    #     for i, n_g in enumerate(n_g_array):
    #         # Recalculate H for the current flux value
    #         H, n_tot = double_transmon_chain_interaction(n_g=n_g, transmon_subspace_size_L=transmon_subspace_size)
    #
    #         # Calculate eigenstates and eigenvalues
    #         try:
    #             if i == 0:
    #                 eigenvals, eigenvecs = sparse.linalg.eigs(H, k=number_of_energies, which='SM')
    #             else:
    #                 eigenvals, eigenvecs = sparse.linalg.eigs(H, k=number_of_energies, which='SM', v0=v0)
    #
    #         except sparse.linalg.ArpackNoConvergence:
    #             print(f"ARPACK did not converge at flux={n_g}. Using the last v0.")
    #             v0 = eigenvectors_n_g[max(i-1, 0), :, 0]  # Use the last known eigenvector
    #             eigenvals, eigenvecs = sparse.linalg.eigs(H, k=number_of_energies, which='SM', v0=v0)
    #
    #         # Store the first x eigenstate energies
    #         sorted_indices = np.argsort(eigenvals.real)
    #         sorted_eigenvals = eigenvals[sorted_indices]
    #         sorted_eigenvecs = eigenvecs[:, sorted_indices]
    #         eigenvalues_n_g[i, :] = sorted_eigenvals
    #         eigenvectors_n_g[i, :, :] = sorted_eigenvecs
    #         v0 = np.mean(sorted_eigenvecs[:, :min(2, sorted_eigenvecs.shape[1])], axis=1)
    #         n_n_g[i, :, :] = n_tot
    #         isherm = is_hermitian(H)
    #         print(f"i={i}, n_g={n_g}, isherm={isherm}")
    #
    #     # End timing
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Loop finished in {elapsed_time:.2f} seconds.")
    #     return eigenvalues_n_g, eigenvectors_n_g, n_n_g
    #
    #
    # eigenvalues_n_g, eigenvectors_n_g, n_operator_n_g = eigen_per_n_g()
    # np.save("eigenvalues_SNS_double_transmon_chain_n_g.npy", eigenvalues_n_g)
    # np.save("eigenvectors_SNS_double_transmon_chain_n_g.npy", eigenvectors_n_g)
    # eigenvalues = np.load("eigenvalues_SNS_double_transmon_chain_n_g.npy")
    # eigenvectors = np.load("eigenvectors_SNS_double_transmon_chain_n_g.npy")
    #
    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # for i in range(number_of_energies):
    #     plt.plot(flux_array, eigenvalues[:, i], label=f'Eigenstate {i}')
    #
    # plt.xlabel('n_g')
    # plt.ylabel('Energy')
    # plt.title(f'First {number_of_energies} Eigenstates vs n_g')
    # plt.legend()
    # plt.grid(True)
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # plot_filename = os.path.join("SNS_double_transmon_chain_plots",
    #                              f"Transmon energies vs n_g {timestamp}.png")
    # plt.savefig(plot_filename, format='png')
    # plt.show()
    #
    #
    # def plot_delta_energy_vs_flux_dipole_transitions():
    #     length = np.sum(range(number_of_energies))
    #     big_M = np.zeros((n_g_array.shape[0], length), dtype=complex)
    #     big_delta = np.zeros_like(big_M)
    #
    #     # compute M and delta for each flux
    #     for i, n_g in enumerate(n_g_array):
    #         M = eigenvectors_n_g[i, :, :].T.conj() @ n_operator_n_g[i, :, :] @ eigenvectors_n_g[i, :, :]
    #         delta = eigenvalues_n_g[i, :, np.newaxis] - eigenvalues_n_g[i, np.newaxis, :]
    #         big_M[i, :] = upper_triangle(M)
    #         big_delta[i, :] = upper_triangle(delta)
    #
    #     big_delta = -1 * big_delta  # minus so the energies will be positive
    #     big_M = abs(big_M) ** 2
    #
    #     # Combine color ranges for normalization
    #     vmin = big_M.min()
    #     vmax = big_M.max()
    #     norm = Normalize(vmin=vmin, vmax=vmax)
    #
    #     # Create the plot
    #     fig, ax = plt.subplots()
    #
    #     # create the description for the graph from the upper triangle indices
    #     upper_tri = np.triu_indices(number_of_energies, k=1)
    #     indices = [(y, x) for x, y in zip(*upper_tri)]
    #     descriptions = [f'E{x}-E{y}' for x, y in indices]
    #
    #     for i in range(length):
    #         line = colored_line(n_g_array, big_delta[:, i], big_M[:, i], ax, linewidth=2, cmap="plasma", norm=norm)
    #
    #         # annotate the line with its description
    #         mid_idx = len(n_g_array) // 2
    #         ax.text(n_g_array[mid_idx], big_delta[mid_idx, i], descriptions[i], fontsize=8, color="black", ha="left")
    #
    #         # Add a single colorbar
    #         if i == (length - 1):
    #             fig.colorbar(line, ax=ax)
    #
    #     ax.set_title("Transmon Transitions due to Dipole")
    #     ax.set_xlabel("n_g")
    #     ax.set_ylabel("Delta Energy")
    #     ax.set_xlim(n_g_array.min(), n_g_array.max())
    #     ax.set_ylim(big_delta.min(), big_delta.max())
    #     plt.tight_layout()
    #
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     plot_filename = os.path.join("SNS_double_transmon_chain_plots",
    #                                  f"Transmon Transitions vs n_g due to Dipole {timestamp}.png")
    #     plt.savefig(plot_filename, format='png')
    #     plt.show()
    #
    # plot_delta_energy_vs_flux_dipole_transitions()

    H_spinchain = spinchain.compute_hamiltonian()
    eigenvals, eigenvecs = sparse.linalg.eigs(H_spinchain, k=number_of_energies, which='SM')
    eigenvals = 10*eigenvals
    print(eigenvals)
    H = H_spinchain.toarray()
    eigenvals2, eigenvecs2 = np.linalg.eigh(H)
    print(eigenvals2)







