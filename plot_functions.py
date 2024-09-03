import numpy as np
import matplotlib.pyplot as plt
from transmon import Transmon
import os
from matplotlib import colors
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection


def plot_x_vs_y_amount_of_lines(amount, x_array, y_matrix, xlabel, ylabel, title):
    for i in range(amount):
        plt.plot(x_array, y_matrix[:, i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.axhline(y=5.5, color='r', linestyle='--')
    plt.title(title)

    # Save the figure as an image (e.g., PNG)
    # data_folder = 'data'
    # filename = os.path.join(data_folder, f'N_g_culomb_int_model_amount_{amount}_CPnum_{n_0_int}.png')
    # plt.savefig(filename)
    plt.show()

def two_plots_energy_vs_n_g(amount, x_array, y_matrix1, y_matrix2, xlabel, ylabel, title, label1, label2):
    """
    Plot energy versus n_g for two sets of eigenvalues.

    Parameters:
        amount (int): Number of eigenvalues to plot.
        n_g_array (array-like): Array of n_g values.
        eigenvalues1 (2D array-like): Eigenvalues for the first set.
        eigenvalues2 (2D array-like): Eigenvalues for the second set.
        label1 (str): Label for the first set of eigenvalues.
        label2 (str): Label for the second set of eigenvalues.
        N_g (int, optional): Value of N_g. Default is 0.
    """
    for i in range(amount):
        plt.plot(x_array, y_matrix1[:, i], label=label1)
        plt.plot(x_array, y_matrix2[:, i], linestyle='--', label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()  # Show legend with labels
    plt.grid(True)  # Add grid lines
    plt.show()


