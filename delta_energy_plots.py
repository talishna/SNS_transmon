from suppresed_our_model import plot_x_y_color
from suppresed_our_model import (unravel_M_numerical_n_g_N_g_half,  unravel_delta_energy_numerical_n_g_N_g_half,
                                 descriptions_numerical_n_g_N_g_half)
from suppresed_our_model import (unravel_M_analytical_N_g_half,  unravel_delta_energy_analytical_N_g_half,
                                 descriptions_analytical_N_g_half)

from suppresed_our_model import (unravel_M_numerical_N_g_n_g_0,  unravel_delta_energy_numerical_N_g_n_g_0,
                                 descriptions_numerical_N_g_n_g_0)
from suppresed_our_model import (unravel_M_analytical_n_g_0,  unravel_delta_energy_analytical_n_g_0,
                                 descriptions_analytical_n_g_0)

from suppresed_our_model import (unravel_M_numerical_N_g_n_g_quarter,  unravel_delta_energy_numerical_N_g_n_g_quarter,
                                 descriptions_numerical_N_g_n_g_quarter)
from suppresed_our_model import (unravel_M_analytical_n_g_quarter,  unravel_delta_energy_analytical_n_g_quarter,
                                 descriptions_analytical_n_g_quarter)

from suppresed_our_model import (unravel_M_numerical_both,  unravel_delta_energy_numerical_both,
                                 descriptions_numerical_both)
from suppresed_our_model import (unravel_M_analytical_both,  unravel_delta_energy_analytical_both,
                                 descriptions_analytical_both)

from suppresed_our_model import (unravel_M_numerical_n_g_N_g_0, unravel_delta_energy_numerical_n_g_N_g_0,
                                 descriptions_numerical_n_g_N_g_0)
from suppresed_our_model import (unravel_M_analytical_N_g_0, unravel_delta_energy_analytical_N_g_0,
                                 descriptions_analytical_N_g_0)

import parameters

n_g_array = parameters.n_g_array
N_g_array = parameters.N_g_array
num_of_lines = parameters.num_of_lines
N_g_range = parameters.N_g_range
labels_one_dataset = parameters.labels_one_dataset


def format_energy_labels(labels):
    """
    Replaces energy level notations (E0, E1, E2, E3) with their LaTeX equivalents.

    Parameters:
        labels (list of str): List of energy difference labels.

    Returns:
        list of str: Formatted labels with LaTeX notation.
    """
    mapping = {
        'E0': r'${\left| 0,- \right\rangle}$',
        'E1': r'${\left| 0,+ \right\rangle}$',
        'E2': r'${\left| 1,- \right\rangle}$',
        'E3': r'${\left| 1,+ \right\rangle}$'
    }

    formatted_labels = []
    for label in labels:
        for key, value in mapping.items():
            label = label.replace(key, value)
        formatted_labels.append(label)

    return formatted_labels

if __name__ == "__main__":
    # in this code i want to create a graph of energy transitions as a result of the n operator
    # in the even subspace of the transmon +dot + coulomb int for N_g=-0.5
    # I want the line to be colored by the transition amplitude


    descriptions_numerical_n_g_N_g_half = format_energy_labels(descriptions_numerical_n_g_N_g_half)
    descriptions_analytical_N_g_half = format_energy_labels(descriptions_analytical_N_g_half)
    descriptions_numerical_N_g_n_g_0 = format_energy_labels(descriptions_numerical_N_g_n_g_0)
    descriptions_analytical_n_g_0 = format_energy_labels(descriptions_analytical_n_g_0)
    descriptions_numerical_both = format_energy_labels(descriptions_numerical_both)
    descriptions_analytical_both = format_energy_labels(descriptions_analytical_both)
    descriptions_numerical_N_g_n_g_quarter = format_energy_labels(descriptions_numerical_N_g_n_g_quarter)
    descriptions_analytical_n_g_quarter = format_energy_labels(descriptions_analytical_n_g_quarter)

    # Numerical data
    # delta energy as a function of n_g with N_g=-0.5 colored by dipole transition
    plot_x_y_color(color_values=unravel_M_numerical_n_g_N_g_half, x=n_g_array,
                   y=unravel_delta_energy_numerical_n_g_N_g_half,
                   xlabel=r'$n_g$', ylabel=r'$\Delta E$',
                   title=r'Numerical Solution' '\n' '$\Delta E$ vs $n_g$ even subspace $N_g$=-0.5',
                   descriptions=descriptions_numerical_n_g_N_g_half)

    # Analytical data
    # now the colored plots for analytical energy vs n_g Even subspace only with N_g=-0.5:
    plot_x_y_color(unravel_M_analytical_N_g_half, n_g_array, unravel_delta_energy_analytical_N_g_half, xlabel=r'$n_g$',
                   ylabel=r'$\Delta E$',
                   title=r'Analytical Solution' '\n' '$\Delta E$ vs $n_g$ even subspace $N_g$=-0.5',
                   descriptions=descriptions_analytical_N_g_half)



    # # Numerical data
    # # delta energy as a function of N_g with n_g=0
    # plot_x_y_color(unravel_M_numerical_N_g_n_g_0, N_g_array, unravel_delta_energy_numerical_N_g_n_g_0, xlabel=r'$N_g$',
    #                ylabel=r'$\Delta E$',
    #                title=r'Numerical Solution' '\n' '$\Delta E$ vs $N_g$ even subspace $n_g=0$',
    #                descriptions=descriptions_numerical_N_g_n_g_0)
    #
    # # Analytical data
    # # now the colored plots for analytical energy vs N_g Even subspace only with n_g=0:
    # plot_x_y_color(unravel_M_analytical_n_g_0, N_g_array, unravel_delta_energy_analytical_n_g_0, xlabel=r'$N_g$',
    #                ylabel=r'$\Delta E$',
    #                title=r'Analytical Solution' '\n' '$\Delta E$ vs $N_g$ even subspace $n_g$=0',
    #                descriptions=descriptions_analytical_n_g_0)
    #
    #
    #
    # # Numerical data
    # # delta energy as a function of N_g with n_g=0.25
    # plot_x_y_color(unravel_M_numerical_N_g_n_g_quarter, N_g_array, unravel_delta_energy_numerical_N_g_n_g_quarter, xlabel=r'$N_g$',
    #                ylabel=r'$\Delta E$',
    #                title=r'Numerical Solution' '\n' '$\Delta E$ vs $N_g$ even subspace $n_g=0.25$',
    #                descriptions=descriptions_numerical_N_g_n_g_quarter)
    #
    # # Analytical data
    # # now the colored plots for analytical energy vs N_g Even subspace only with n_g=0.25:
    # plot_x_y_color(unravel_M_analytical_n_g_quarter, N_g_array, unravel_delta_energy_analytical_n_g_quarter, xlabel=r'$N_g$',
    #                ylabel=r'$\Delta E$',
    #                title=r'Analytical Solution' '\n' '$\Delta E$ vs $N_g$ even subspace $n_g$=0.25',
    #                descriptions=descriptions_analytical_n_g_quarter)
    #



    # # Numerical data
    # # delta energy as a function of N_g when both N_g and n_g change
    # plot_x_y_color(unravel_M_numerical_both, N_g_range, unravel_delta_energy_numerical_both, xlabel=r'$N_g$',
    #                ylabel=r'$\Delta E$',
    #                title=r'Numerical Solution' '\n' '$\Delta E$ vs $N_g$ even subspace varying $n_g$',
    #                descriptions=descriptions_numerical_both)
    #
    # # Analytical data
    # # now the colored plots for analytical energy vs N_g when both N_g and n_g change:
    # plot_x_y_color(unravel_M_analytical_both, N_g_range, unravel_delta_energy_analytical_both, xlabel=r'$N_g$',
    #                ylabel=r'$\Delta E$',
    #                title=r'Analytical Solution' '\n' '$\Delta E$ vs $N_g$ even subspace varying $n_g$',
    #                descriptions=descriptions_analytical_both)

    # Numerical data
    # n_g varies and N_g=0 even subspace
    plot_x_y_color(color_values=unravel_M_numerical_n_g_N_g_0, x=n_g_array,
                   y=unravel_delta_energy_numerical_n_g_N_g_0,
                   xlabel=r'$n_g$', ylabel=r'$\Delta E$',
                   title=r'Numerical Solution' '\n' '$\Delta E$ vs $n_g$ even subspace $N_g$=0',
                   descriptions=descriptions_numerical_n_g_N_g_0)

    # Analytical data
    # n_g varies and N_g=0 even subspace
    plot_x_y_color(color_values=unravel_M_analytical_N_g_0, x=n_g_array,
                   y=unravel_delta_energy_analytical_N_g_0,
                   xlabel=r'$n_g$', ylabel=r'$\Delta E$',
                   title=r'Numerical Solution' '\n' '$\Delta E$ vs $n_g$ even subspace $N_g$=0',
                   descriptions=descriptions_analytical_N_g_0)
