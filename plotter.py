import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from data import read_agent_data, read_evo_data, read_configs

np.set_printoptions(suppress=True)


def _plot_phenos_scatter(train_phenotypes=None, params=None, test_phenotypes=None):

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


def _plot_exp_evo_data(median_bests, median_means, lq_means, uq_means, colour):

    # Create x axis of generations
    gens = np.arange(1, median_bests.shape[0] + 1)

    # Prepare data for plotting
    plot_median_bests = np.column_stack((gens, median_bests))
    plot_median_means = np.column_stack((gens, median_means))
    plot_uq_means = np.column_stack((gens, uq_means))
    plot_lq_means = np.column_stack((gens, lq_means))
    plot_data = np.array([plot_median_bests, plot_median_means,
                          plot_uq_means, plot_lq_means])

    line_styles = ['--', '-', '--', '--']
    line_widths = [1., 1., 0.25, 0.25]

    for i in range(plot_data.shape[0]):
        plt.plot(plot_data[i, :, 0], plot_data[i, :, 1],
                 color=colour, linestyle=line_styles[i],
                 linewidth=line_widths[i])

    plt.fill_between(gens, lq_means, uq_means, color=colour, alpha=0.1)


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


def read_and_plot_phenos(exp_data_path=None, winner_file_name=None, test_data=None):

    # Read agent data
    if exp_data_path is not None:
        fitnesses, genos, phenos, params, folder_paths = \
            read_agent_data(exp_data_path, winner_file_name)

    print("Fitnesses:\n", fitnesses)
    print("Genotypes:\n", genos)
    print("Phenotypes:\n", phenos)
    print("Params:\n", params[0])

    # Provide fitness analysis
    _fitness_analysis(fitnesses, folder_paths)

    # Plot training and/or test data
    _plot_phenos_scatter(phenos, params, test_data)


def read_and_plot_evo_data(exp_data_dirs, data_dir_path):

    exp_plot_colours = ['b', 'r', 'g', 'm', 'y']
    legend_items = []

    # Append data dir path to experiment directories
    exp_data_paths = [data_dir_path + edd for edd in exp_data_dirs]

    for i, exp_data_path in enumerate(exp_data_paths):

        # Read experiment data
        mean_fitnesses, best_fitnesses = read_evo_data(exp_data_path)

        # Calculate statistics
        # mean_best_so_far_fitnesses = np.mean(best_fitnesses, axis=0)
        median_best_fitnesses = np.median(best_fitnesses, axis=0)
        median_mean_fitnesses = np.median(mean_fitnesses, axis=0)
        lq_mean_fitnesses = np.quantile(mean_fitnesses, 0.25, axis=0)
        uq_mean_fitnesses = np.quantile(mean_fitnesses, 0.75, axis=0)

        # Plot experiment data
        _plot_exp_evo_data(median_best_fitnesses, median_mean_fitnesses,
                           lq_mean_fitnesses, uq_mean_fitnesses, exp_plot_colours[i])

        legend_items.append(mpatches.Patch(color=exp_plot_colours[i],
                                           label=exp_data_dirs[i]))

    plt.legend(handles=legend_items)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    plt.show()


if __name__ == '__main__':

    config = read_configs(sys.argv)[0]

    # Plot phenotype data
    if '-pheno' in sys.argv:

        exp_dir = sys.argv[2]
        read_and_plot_phenos(config['logging']['data_dir_path'] + exp_dir,
                             config['logging']['winner_file_name'])

    # Plot evolutionary run data
    elif '-evo' in sys.argv:

        exp_data_dirs = sys.argv[2]
        # Split comma separated experiment directories
        exp_data_dirs = exp_data_dirs.split(',')

        read_and_plot_evo_data(exp_data_dirs, config['logging']['data_dir_path'])
