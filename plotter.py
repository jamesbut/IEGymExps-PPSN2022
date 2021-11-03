import sys
import matplotlib.pyplot as plt
import numpy as np
from data import read_data

np.set_printoptions(suppress=True)


def _plot_data(train_phenotypes=None, params=None, test_phenotypes=None):

    if train_phenotypes is not None:
        if params is not None:
            plt.scatter(train_phenotypes[:, 0], train_phenotypes[:, 1],
                        c=params[:, 0], cmap='plasma')
            plt.colorbar()
        else:
            plt.scatter(train_phenotypes[:, 0], train_phenotypes[:, 1])

    if test_phenotypes is not None:
        plt.scatter(test_phenotypes[:, 0], test_phenotypes[:, 1])

    plt.show()


def _fitness_analysis(fitnesses, folder_paths):

    max_arg = np.argmax(fitnesses)
    max_fitness = fitnesses[max_arg]
    max_file = folder_paths[max_arg]

    print("Max fitness: {}              File: {}".format(max_fitness, max_file))

    min_arg = np.argmin(fitnesses)
    min_fitness = fitnesses[min_arg]
    min_file = folder_paths[min_arg]

    print("Min fitness: {}              File: {}".format(min_fitness, min_file))

    mean_fitness = np.mean(fitnesses)
    print("Mean fitness:", mean_fitness)


def read_and_plot(data_dir=None, test_data=None):

    # Read data
    if data_dir is not None:
        fitnesses, genos, phenos, params, folder_paths = read_data(data_dir)

    print("Fitnesses:\n", fitnesses)
    print("Genotypes:\n", genos)
    print("Phenotypes:\n", phenos)
    print("Params:\n", params[0])

    # Provide fitness analysis
    _fitness_analysis(fitnesses, folder_paths)

    # Plot training and/or test data
    _plot_data(phenos, params, test_data)


if __name__ == '__main__':

    read_and_plot(sys.argv[1])
