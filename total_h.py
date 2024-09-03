import numpy as np
import matplotlib.pyplot as plt
import time
from transmon import Transmon
from cavity import Cavity
from chain import Chain


class TotalHamiltonian:
    def __init__(self, E_C, n_0, E_J_max, d, flux_0, Wc, max_num_photons, N, t, epsilon_r, g, gamma_L, gamma_R,
                 cutoff_transmon=False, size_subspace_transmon=None):
        self.g = g
        self.gamma_L = gamma_L
        self.gamma_R = gamma_R
        # self.flux = flux
        # self.n_g = n_g
        self.cutoff_transmon = cutoff_transmon
        self.size_of_subspace_T = size_subspace_transmon
        self.transmon = Transmon(E_C=E_C, n_0=n_0, E_J_max=E_J_max, d=d, flux_0=flux_0,
                                 size_of_subspace=size_subspace_transmon)  # This is a specific instance of the transmon class
        self.cavity = Cavity(Wc=Wc, max_num_photons=max_num_photons)  # This is a specific instance of the cavity class
        self.chain = Chain(N=N, t=t, epsilon_r=epsilon_r)

    def hamiltonian_int_cavity_transmon(self, n_g=0, cutoff_transmon=False, size_subspace_transmon=None, U=None):
        H = np.kron(self.g * (self.cavity.annihilation + self.cavity.creation),
                    (self.transmon.n_hat - n_g * np.eye(self.transmon.dimension)))
        B = np.sum(np.abs(H - np.conj(H.T)))
        print("hermitian of H_int_cav_transmon:", B)
        return H

    def hamiltonian_int_transmon_chain(self, flux=0, cutoff_transmon=False, size_subspace_transmon=None, U=None):
        exp_flux_plus = np.exp(1j * np.pi * flux / (2 * self.transmon.flux_0))  # should be next to phi_L and -phi_R
        exp_flux_minus = np.conj(exp_flux_plus)  # should be next to -phi_L and phi_R
        H = np.zeros((self.transmon.dimension * self.chain.dimension, self.transmon.dimension * self.chain.dimension))
        if self.chain.N < 2:
            return H
        left_term = np.kron(exp_flux_plus * self.transmon.creation,
                            np.kron(self.chain.s_minus,
                                    np.kron(self.chain.s_minus, np.eye(2 ** (self.chain.N - 2)))))
        right_term = np.kron(exp_flux_minus * self.transmon.creation,
                             np.kron(np.eye(2 ** (self.chain.N - 2)),
                                     np.kron(self.chain.s_minus, self.chain.s_minus)))
        H = self.gamma_L * (left_term + left_term.conj().T) + \
            self.gamma_R * (right_term + right_term.conj().T)
        B = np.sum(np.abs(H - np.conj(H.T)))
        print("hermitian of H_int_trans_chain:", B)
        return H

    def total_hamiltonian(self, flux=0, n_g=0, cutoff_transmon=False):
        H_transmon = self.transmon.compute_hamiltonian(flux=flux, n_g=n_g)
        H_cavity = self.cavity.compute_hamiltonian()
        H_cav_tran = self.hamiltonian_int_cavity_transmon(n_g=n_g)
        H_chain = self.chain.compute_hamiltonian()
        H_tran_chain = self.hamiltonian_int_transmon_chain(flux=flux)
        H = (np.kron(np.eye(self.cavity.dimension), np.kron(H_transmon, np.eye(self.chain.dimension))) +
             np.kron(H_cavity, np.eye(self.transmon.dimension * self.chain.dimension)) +
             np.kron(H_cav_tran, np.eye(self.chain.dimension)) +
             np.kron(np.eye(self.cavity.dimension * self.transmon.dimension), H_chain) +
             np.kron(np.eye(self.cavity.dimension), H_tran_chain))
        return H
