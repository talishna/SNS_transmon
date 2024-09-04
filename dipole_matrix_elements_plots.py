from suppresed_our_model import create_M_and_delta
from suppresed_our_model import create_upper_triangle_of_3d_array
from suppresed_our_model import plot_x_y_color
from suppresed_our_model import plot_data_vs_x
from suppresed_our_model import M_analytical_N_g_half
from suppresed_our_model import M_numerical_n_g_N_g_half
from suppresed_our_model import M_analytical_n_g_0
from suppresed_our_model import M_numerical_N_g_n_g_0

import parameters
a = 5
n_g_array = parameters.n_g_array
N_g_array = parameters.N_g_array


if __name__ == "__main__":

    # plot M02 M03 M12 M13 as a function of n_g for both analytical and numerical models
    # M_analytical_N_g_half
    # M_numerical_n_g_N_g_half
    labels_two_datasets_M_plots = [["analytical"], ["numerical"]]
    # M02
    plot_data_vs_x(amount=1, x_array=n_g_array, y_matrices=[M_analytical_N_g_half[:, 0, 2].reshape(-1, 1),
                                                            M_numerical_n_g_N_g_half[:, 0, 2].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${n_g}$', ylabel='M02',
                   title='Numerical and analytical \n M02 vs n_g even subspace N_g=-0.5',
                   labels=labels_two_datasets_M_plots)
    # M03
    plot_data_vs_x(amount=1, x_array=n_g_array, y_matrices=[M_analytical_N_g_half[:, 0, 3].reshape(-1, 1),
                                                            M_numerical_n_g_N_g_half[:, 0, 3].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${n_g}$', ylabel='M03',
                   title='Numerical and analytical \n M03 vs n_g even subspace N_g=-0.5',
                   labels=labels_two_datasets_M_plots)
    # M12
    plot_data_vs_x(amount=1, x_array=n_g_array, y_matrices=[M_analytical_N_g_half[:, 1, 2].reshape(-1, 1),
                                                            M_numerical_n_g_N_g_half[:, 1, 2].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${n_g}$', ylabel='M12',
                   title='Numerical and analytical \n M12 vs n_g even subspace N_g=-0.5',
                   labels=labels_two_datasets_M_plots)
    #M13
    plot_data_vs_x(amount=1, x_array=n_g_array, y_matrices=[M_analytical_N_g_half[:, 1, 3].reshape(-1, 1),
                                                            M_numerical_n_g_N_g_half[:, 1, 3].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${n_g}$', ylabel='M13',
                   title='Numerical and analytical \n M13 vs n_g even subspace N_g=-0.5',
                   labels=labels_two_datasets_M_plots)




    # plot M02 M03 M12 M13 as a function of N_g for both analytical and numerical
    # M_analytical_n_g_0
    # M_numerical_N_g_n_g_0
    # M02
    plot_data_vs_x(amount=1, x_array=N_g_array, y_matrices=[M_analytical_n_g_0[:, 0, 2].reshape(-1, 1),
                                                            M_numerical_N_g_n_g_0[:, 0, 2].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${N_g}$', ylabel='M02',
                   title='Numerical and analytical \n M02 vs N_g even subspace n_g=0',
                   labels=labels_two_datasets_M_plots)
    # M03
    plot_data_vs_x(amount=1, x_array=N_g_array, y_matrices=[M_analytical_n_g_0[:, 0, 3].reshape(-1, 1),
                                                            M_numerical_N_g_n_g_0[:, 0, 3].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${N_g}$', ylabel='M03',
                   title='Numerical and analytical \n M03 vs N_g even subspace n_g=0',
                   labels=labels_two_datasets_M_plots)
    # M12
    plot_data_vs_x(amount=1, x_array=N_g_array, y_matrices=[M_analytical_n_g_0[:, 1, 2].reshape(-1, 1),
                                                            M_numerical_N_g_n_g_0[:, 1, 2].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${N_g}$', ylabel='M12',
                   title='Numerical and analytical \n M12 vs N_g even subspace n_g=0',
                   labels=labels_two_datasets_M_plots)
    # M13
    plot_data_vs_x(amount=1, x_array=N_g_array, y_matrices=[M_analytical_n_g_0[:, 1, 3].reshape(-1, 1),
                                                            M_numerical_N_g_n_g_0[:, 1, 3].reshape(-1, 1)],
                   num_datasets=2, xlabel=r'${N_g}$', ylabel='M13',
                   title='Numerical and analytical \n M13 vs N_g even subspace n_g=0',
                   labels=labels_two_datasets_M_plots)