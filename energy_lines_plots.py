from suppresed_our_model import create_M_and_delta
from suppresed_our_model import create_upper_triangle_of_3d_array
from suppresed_our_model import plot_x_y_color
from suppresed_our_model import plot_data_vs_x

from suppresed_our_model import eigenvalues_n_g_N_g_0_even, eigenvalues_analytical_N_g_0

from suppresed_our_model import eigenvalues_n_g_N_g_half_even, eigenvalues_analytical_N_g_half

from suppresed_our_model import eigenvalues_both_even, eigenvalues_analytical_both

from suppresed_our_model import eigenvalues_N_g_n_g_0_even, eigenvalues_analytical_n_g_0

from suppresed_our_model import eigenvalues_N_g_n_g_quarter_even, eigenvalues_analytical_n_g_quarter

import parameters

n_g_array = parameters.n_g_array
N_g_array = parameters.N_g_array
num_of_lines = parameters.num_of_lines
N_g_range = parameters.N_g_range
labels_one_dataset = parameters.labels_one_dataset

if __name__ == "__main__":


    # Numerical and Analytical together
    # Analytical data
    # the two plots of the numerical and analytical together. energy vs n_g Even subspace only with N_g=-0.5.
    # labels_two_datasets = [[r"analytical-${\left| 0,- \right\rangle}$", r"analytical-${\left| 0,+ \right\rangle}$",
    #                         r"analytical-${\left| 1,- \right\rangle}$", r"analytical-${\left| 1,+ \right\rangle}$"],
    #                        [r"numerical-${\left| 0,- \right\rangle}$", r"numerical-${\left| 0,+ \right\rangle}$",
    #                         r"numerical-${\left| 1,- \right\rangle}$", r"numerical-${\left| 1,+ \right\rangle}$"]]

    labels_two_datasets = [[r"${\left| 0,- \right\rangle}$", r"${\left| 0,+ \right\rangle}$",
                            r"${\left| 1,- \right\rangle}$", r"${\left| 1,+ \right\rangle}$"],
                           [r"${\left| 0,- \right\rangle}$", r"${\left| 0,+ \right\rangle}$",
                            r"${\left| 1,- \right\rangle}$", r"${\left| 1,+ \right\rangle}$"]]
    # n_g varies and N_g=0
    plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
                   y_matrices=[eigenvalues_analytical_N_g_0, eigenvalues_n_g_N_g_0_even],
                   num_datasets=2, xlabel=r'${n_g}$', ylabel='Energy',
                   title=r'Numerical Solution' '\n' 'Energy vs $n_g$ even subspace $N_g$=0',
                   labels=labels_two_datasets)

    plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
                   y_matrices=[eigenvalues_analytical_N_g_half, eigenvalues_n_g_N_g_half_even],
                   num_datasets=2, xlabel=r'$n_g$', ylabel='$E$',
                   title=r'Numerical and Analytical' '\n' 'Energy vs $n_g$ even subspace $N_g$=-0.5',
                   labels=labels_two_datasets)


    # # the two plots of the numerical and analytical together. energy vs N_g Even subspace only with n_g=0.
    # plot_data_vs_x(amount=num_of_lines, x_array=N_g_array,
    #                y_matrices=[eigenvalues_analytical_n_g_0, eigenvalues_N_g_n_g_0_even],
    #                num_datasets=2, xlabel=r'$N_g$', ylabel='$E$',
    #                title=r'Numerical and Analytical' '\n' 'Energy vs $N_g$ even subspace $n_g=0$',
    #                labels=labels_two_datasets)
    #
    # # the two plots of the numerical and analytical together. energy vs N_g Even subspace only with n_g=0.25.
    # plot_data_vs_x(amount=num_of_lines, x_array=N_g_array,
    #                y_matrices=[eigenvalues_analytical_n_g_quarter, eigenvalues_N_g_n_g_quarter_even],
    #                num_datasets=2, xlabel=r'$N_g$', ylabel='$E$',
    #                title=r'Numerical and Analytical' '\n' 'Energy vs $N_g$ even subspace $n_g=0.25$',
    #                labels=labels_two_datasets)
    #
    # # now i want to write a code that plots the energy vs N_g when N_g is in [-2,2] but also when n_g changes rapidly in
    # # [-4,4]
    # plot_data_vs_x(amount=num_of_lines, x_array=N_g_range,
    #                y_matrices=[eigenvalues_both_even, eigenvalues_analytical_both], num_datasets=2,
    #                xlabel=r'${N_g}$', ylabel='$E$',
    #                title=r'Numerical and Analytical' '\n' 'Energy vs $N_g$ even subspace varying $n_g$',
    #                labels=labels_two_datasets)