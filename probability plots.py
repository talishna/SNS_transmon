import numpy as np
import matplotlib.pyplot as plt
from transmon import Transmon
import os
from matplotlib import colors
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
import math
import parameters

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

from suppresed_our_model import plot_data_vs_x
import parameters
from delta_energy_plots import format_energy_labels

n_g_array = parameters.n_g_array
N_g_array = parameters.N_g_array
num_of_lines = 6

descriptions_numerical_n_g_N_g_half = format_energy_labels(descriptions_numerical_n_g_N_g_half)
descriptions_analytical_N_g_half = format_energy_labels(descriptions_analytical_N_g_half)
descriptions_numerical_N_g_n_g_0 = format_energy_labels(descriptions_numerical_N_g_n_g_0)
descriptions_analytical_n_g_0 = format_energy_labels(descriptions_analytical_n_g_0)
descriptions_numerical_both = format_energy_labels(descriptions_numerical_both)
descriptions_analytical_both = format_energy_labels(descriptions_analytical_both)
descriptions_numerical_N_g_n_g_quarter = format_energy_labels(descriptions_numerical_N_g_n_g_quarter)
descriptions_analytical_n_g_quarter = format_energy_labels(descriptions_analytical_n_g_quarter)

def normalize(array):
    # Force normalization to range [0, 1]
    min_val, max_val = np.min(array), np.max(array)
    normalized_color_values = (array - min_val) / (max_val - min_val)
    return normalized_color_values

normalized_unravel_M_numerical_n_g_N_g_half = normalize(unravel_M_numerical_n_g_N_g_half)
normalized_unravel_M_analytical_N_g_half = normalize(unravel_M_analytical_N_g_half)

plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
               y_matrices=[normalized_unravel_M_numerical_n_g_N_g_half, 0],
               num_datasets=1, xlabel=r'$n_g$', ylabel='Probability',
               title=r'Numerical and Analytical' '\n' 'Energy vs $n_g$ even subspace $N_g$=-0.5',
               labels=[descriptions_numerical_n_g_N_g_half])

plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
               y_matrices=[normalized_unravel_M_analytical_N_g_half, 0],
               num_datasets=1, xlabel=r'$n_g$', ylabel='Probability',
               title=r'Numerical and Analytical' '\n' 'Energy vs $n_g$ even subspace $N_g$=-0.5',
               labels=[descriptions_analytical_N_g_half])
plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
               y_matrices=[unravel_M_numerical_n_g_N_g_half, 0],
               num_datasets=1, xlabel=r'$n_g$', ylabel='Probability',
               title=r'Numerical and Analytical' '\n' 'Energy vs $n_g$ even subspace $N_g$=-0.5',
               labels=[descriptions_numerical_n_g_N_g_half])

plot_data_vs_x(amount=num_of_lines, x_array=n_g_array,
               y_matrices=[unravel_M_analytical_N_g_half, 0],
               num_datasets=1, xlabel=r'$n_g$', ylabel='Probability',
               title=r'Numerical and Analytical' '\n' 'Energy vs $n_g$ even subspace $N_g$=-0.5',
               labels=[descriptions_analytical_N_g_half])