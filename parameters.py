import numpy as np

"""
This file contains the parameters for the model Tali Shnaider and Eytan grosfeld developed for the SNS transmon with dot 
system.
"""

# Parameters
E_C = 1  # charging energy
n_0_int = 20  # 10 # number of CP
E_J_max = 10  # (Wq+E_C)**2/8/E_C
d = 0  # 0.35 #squid asymmetry
flux_0 = 1  # 2.067833 * 10 ** (-15)
size_of_transmon_subspace = 0
plasma_energy = np.sqrt(8 * E_C * E_J_max)
ratio = E_J_max / E_C
gamma = 0.05
gamma_L = 0.05
gamma_R = 0.05
n_0_half_int = n_0_int + 0.5
E_C_tag = 0
F = 0

U = 0.1
E_X = 0.1
E_C_tag = E_C + 2 * U - 2 * E_X
epsilon = 0.5
const = 0 # (U-2*epsilon)**2/(4*E_C_tag**2)

steps = 300
n_g_array = np.linspace(-0.5, 0.5, steps)
N_g_array = np.linspace(-2, 2, steps)
total_dim = 4 * (2 * n_0_int + 1)  # 4 because of the dot

# these are parameters for the plots
num_of_lines = 4  # eigenvalues_sys.shape[1] #6*2**N_sites
labels_one_dataset = [f'lvl{i + 1}' for i in range(num_of_lines)]
amount_of_energy_diff = np.sum(list(range(num_of_lines)))

# Define range for N_g and n_g for the graph of both n_g and N_g varying
n_g_range = np.linspace(-4, 4, 400)
N_g_range = np.linspace(-2, 2, 400)
