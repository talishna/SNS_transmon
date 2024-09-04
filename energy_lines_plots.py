from suppresed_our_model import create_M_and_delta
from suppresed_our_model import create_upper_triangle_of_3d_array
from suppresed_our_model import plot_x_y_color
from suppresed_our_model import plot_data_vs_x
from suppresed_our_model import eigenvalues_both_even
from suppresed_our_model import eigenvalues_analytical_N_g_half
from suppresed_our_model import eigenvalues_n_g_N_g_half_even
from suppresed_our_model import eigenvalues_analytical_n_g_0
from suppresed_our_model import eigenvalues_N_g_n_g_0_even

import parameters
a = 5
n_g_array = parameters.n_g_array
N_g_array = parameters.N_g_array
num_of_lines = parameters.num_of_lines
N_g_range = parameters.N_g_range
labels_one_dataset = parameters.labels_one_dataset

if __name__ == "__main__":
    # now i want to write a code that plots the energy vs N_g when N_g is in [-2,2] but also when n_g changes rapidly in
    # [-4,4]
    plot_data_vs_x(amount=num_of_lines, x_array=N_g_range, y_matrices=[eigenvalues_both_even, 0], num_datasets=1,
                   xlabel=r'${N_g}$', ylabel='Energy',
                   title='Numerical \n Energy vs N_g even subspace varying n_g',
                   labels=[labels_one_dataset, 0])

    # Numerical and Analytical together
    # Analytical data
    # the two plots of the numerical and analytical together. energy vs n_g Even subspace only with N_g=-0.5.
    labels_two_datasets = [["analytical-lvl1", "analytical-lvl2", "analytical-lvl3", "analytical-lvl4"],
                           ["numerical-lvl1", "numerical-lvl2", "numerical-lvl3", "numerical-lvl4"]]
    plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
                   y_matrices=[eigenvalues_analytical_N_g_half, eigenvalues_n_g_N_g_half_even],
                   num_datasets=2, xlabel=r'$n_g$', ylabel='Energy',
                   title='Numerical and Analytical \n Energy vs n_g even subspace N_g=-0.5',
                   labels=labels_two_datasets)


    # the two plots of the numerical and analytical together. energy vs N_g Even subspace only with n_g=0.
    plot_data_vs_x(amount=num_of_lines, x_array=N_g_array,
                   y_matrices=[eigenvalues_analytical_n_g_0, eigenvalues_N_g_n_g_0_even],
                   num_datasets=2, xlabel=r'$N_g$', ylabel='Energy',
                   title='Numerical and Analytical \n Energy vs N_g even subspace n_g=0',
                   labels=labels_two_datasets)