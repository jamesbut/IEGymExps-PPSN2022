import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from data import read_agent_data, read_evo_data, read_configs, get_sub_folders
from command_line import parse_axis_limits

np.set_printoptions(suppress=True)


def _plot_phenos_scatter(train_phenotypes=None, colour_vals=None, test_phenotypes=None,
                         plot_axis_lb=None, plot_axis_ub=None):

    if train_phenotypes is not None:
        if colour_vals is not None:
            plt.scatter(train_phenotypes[:, 0], train_phenotypes[:, 1],
                        c=colour_vals, cmap='plasma')
            plt.colorbar()
        else:
            plt.scatter(train_phenotypes[:, 0], train_phenotypes[:, 1])

    if test_phenotypes is not None:
        plt.scatter(test_phenotypes[:, 0], test_phenotypes[:, 1], alpha=0.1)

    # Set axis limit if given
    if plot_axis_lb and plot_axis_ub:
        plt.xlim([plot_axis_lb, plot_axis_ub])
        plt.ylim([plot_axis_lb, plot_axis_ub])

    plt.show()


def _plot_exp_evo_data(mean_bests, median_bests, lq_bests, uq_bests, median_means,
                       lq_means, uq_means, colour,
                       plot_q_means=True, plot_q_bests=True):

    # Create x axis of generations
    gens = np.arange(1, median_bests.shape[0] + 1)

    # Prepare data for plotting
    # plot_mean_bests = np.column_stack((gens, mean_bests))
    plot_median_bests = np.column_stack((gens, median_bests))
    plot_lq_bests = np.column_stack((gens, lq_bests))
    plot_uq_bests = np.column_stack((gens, uq_bests))

    plot_median_means = np.column_stack((gens, median_means))
    plot_uq_means = np.column_stack((gens, uq_means))
    plot_lq_means = np.column_stack((gens, lq_means))

    # Select data to plot
    plot_data = np.array([plot_median_bests, plot_median_means])
    line_styles = ['--', '-']
    line_widths = [1., 1.]
    if plot_q_bests:
        plot_data = np.concatenate((plot_data,
                                    np.array([plot_lq_bests, plot_uq_bests])))
        line_styles += ['--', '--']
        line_widths += [0.25, 0.25]
    if plot_q_means:
        plot_data = np.concatenate((plot_data,
                                    np.array([plot_lq_means, plot_uq_means])))
        line_styles += ['--', '--']
        line_widths += [0.25, 0.25]

    # Plot data
    for i in range(plot_data.shape[0]):
        plt.plot(plot_data[i, :, 0], plot_data[i, :, 1],
                 color=colour, linestyle=line_styles[i],
                 linewidth=line_widths[i])

    # Fill between IQRs
    if plot_q_means:
        plt.fill_between(gens, lq_means, uq_means, color=colour, alpha=0.1)
    if plot_q_bests:
        plt.fill_between(gens, lq_bests, uq_bests, color=colour, alpha=0.1)


def _fitness_analysis(fitnesses, folder_paths, verbosity) -> float:

    max_arg = np.argmax(fitnesses)
    max_fitness = fitnesses[max_arg]
    max_file = folder_paths[max_arg]

    if verbosity:
        print("Max fitness: {}              File: {}".format(max_fitness, max_file))

    mean_fitness = np.mean(fitnesses)
    if verbosity:
        print("Mean fitness:", mean_fitness)

    median_fitness = np.median(fitnesses)
    if verbosity:
        print("Median fitness:", median_fitness)

    min_arg = np.argmin(fitnesses)
    min_fitness = fitnesses[min_arg]
    min_file = folder_paths[min_arg]

    if verbosity:
        print("Min fitness: {}              File: {}".format(min_fitness, min_file))

    return max_fitness


# Calculate best fitnesses so far from the best fitnesses for each generation
def calculate_best_fitnesses_so_far(best_fitnesses):

    def create_best_fitnesses_so_far(run_best_fitnesses):
        best_fitnesses_so_far = np.empty_like(run_best_fitnesses)
        best_fitness_so_far = run_best_fitnesses[0]
        # Calculate best winner so far at each generation
        for i, f in enumerate(run_best_fitnesses):
            if f > best_fitness_so_far:
                best_fitness_so_far = f
            best_fitnesses_so_far[i] = best_fitness_so_far
        return best_fitnesses_so_far

    # For each evolutionary run
    return np.apply_along_axis(create_best_fitnesses_so_far, 1, best_fitnesses)


# Read pheno data of experiment
def _read_exp(exp_data_path, winner_file_name, verbosity, colour_params):

    # Read agent data
    if exp_data_path is not None:
        fitnesses, genos, phenos, params, folder_paths = \
            read_agent_data(exp_data_path, winner_file_name)

    if verbosity:
        print("Fitnesses:\n", fitnesses)
        print("Genotypes:\n", genos)
        print("Phenotypes:\n", phenos)
        print("Params:\n", params[0])

    # Flatten params for now - this might not work when training has been on
    # more than one param
    params = params.flatten()

    # Provide fitness analysis
    max_fitness = _fitness_analysis(fitnesses, folder_paths, verbosity)

    # Determine colour values for plotting
    colour_vals = fitnesses
    if colour_params:
        colour_vals = params

    return max_fitness, phenos, colour_vals


def read_and_plot_phenos(exp_data_path=None, winner_file_name=None, test_data=None,
                         group=False, colour_params=False, print_numpy_arrays=False,
                         verbosity=True, plot_axis_lb=None, plot_axis_ub=None):

    # Get all experiments from group
    if group:
        exp_data_paths = get_sub_folders(exp_data_path, recursive=False,
                                         sort_by_suffix_num=True)
    # Otherwise just process given experiment
    else:
        exp_data_paths = [exp_data_path]
        verbosity = True

    # Print full numpy arrays
    if print_numpy_arrays:
        np.set_printoptions(threshold=sys.maxsize)

    max_exp_fitnesses = []

    # For all experiments, collect pheno data
    for i, exp_data_path in enumerate(exp_data_paths):
        print(exp_data_path)

        # Read pheno data
        max_fitness, phenos, colour_vals = _read_exp(exp_data_path, winner_file_name,
                                                     verbosity, colour_params)
        # Plot pheno data
        if verbosity:
            _plot_phenos_scatter(phenos, colour_vals, test_data,
                                 plot_axis_lb, plot_axis_ub)

        # Keep track of max fitness
        max_exp_fitnesses.append(max_fitness)

    if group:
        # Find the experiment with the maximum fitness of the group and run again
        print('**********************************************')
        print('*   Experiment with max fitness in group     *')
        print('**********************************************')

        # Calculate exp with max fitness in group
        group_max_exp_fitness_arg = np.argmax(max_exp_fitnesses)

        # Read and plot
        _, phenos, colour_vals = _read_exp(
            exp_data_paths[group_max_exp_fitness_arg], winner_file_name,
            True, colour_params
        )
        _plot_phenos_scatter(phenos, colour_vals, test_data,
                             plot_axis_lb, plot_axis_ub)

        print('**********************************************')


def _prepare_evo_exp_data_paths(data_dirs, data_dir_path, winner_file_name, verbosity):

    exp_data_paths = []

    for data_dir in data_dirs:

        # Check string for experiment or group
        if 'exp' in data_dir:
            exp_data_paths.append(data_dir_path + data_dir)

        # Group of experiments
        else:

            # Get sub group paths
            group_exp_data_paths = get_sub_folders(data_dir_path + data_dir,
                                                   recursive=False,
                                                   sort_by_suffix_num=True)

            # Calculate max fitness of the experiments in the group
            max_fitnesses = []
            for edp in group_exp_data_paths:
                max_fitness, _, _ = _read_exp(edp, winner_file_name, verbosity, False)
                max_fitnesses.append(max_fitness)

            exp_data_paths.append(group_exp_data_paths[np.argmax(max_fitnesses)])

    return exp_data_paths


def read_and_plot_evo_data(exp_data_dirs, data_dir_path, winner_file_name,
                           plot_q_means=True, plot_q_bests=True, verbosity=False):

    exp_plot_colours = ['b', 'r', 'g', 'm', 'y']
    legend_items = []

    exp_data_paths = _prepare_evo_exp_data_paths(exp_data_dirs, data_dir_path,
                                                 winner_file_name, verbosity)
    print(exp_data_paths)

    for i, exp_data_path in enumerate(exp_data_paths):

        # Read experiment data
        mean_fitnesses, best_fitnesses = read_evo_data(exp_data_path)

        # Calculate best fitnesses so far
        best_fitnesses_so_far = calculate_best_fitnesses_so_far(best_fitnesses)

        # Print number of runs
        print('Number of runs:', len(best_fitnesses))

        # Calculate statistics
        mean_best_fitnesses = np.mean(best_fitnesses_so_far, axis=0)
        median_best_fitnesses = np.median(best_fitnesses_so_far, axis=0)
        lq_best_fitnesses = np.quantile(best_fitnesses_so_far, 0.25, axis=0)
        uq_best_fitnesses = np.quantile(best_fitnesses_so_far, 0.75, axis=0)

        median_mean_fitnesses = np.median(mean_fitnesses, axis=0)
        lq_mean_fitnesses = np.quantile(mean_fitnesses, 0.25, axis=0)
        uq_mean_fitnesses = np.quantile(mean_fitnesses, 0.75, axis=0)

        # Plot experiment data
        _plot_exp_evo_data(mean_best_fitnesses, median_best_fitnesses,
                           lq_best_fitnesses, uq_best_fitnesses,
                           median_mean_fitnesses, lq_mean_fitnesses, uq_mean_fitnesses,
                           exp_plot_colours[i], plot_q_means, plot_q_bests)

        # Set legend label
        legend_label = exp_data_path.replace(data_dir_path, '')
        legend_items.append(
            mpatches.Patch(color=exp_plot_colours[i], label=legend_label)
        )

    plt.legend(handles=legend_items)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    plt.show()


if __name__ == '__main__':

    config = read_configs(sys.argv)[0]

    # Can pass in entire experiment group
    if '-group' in sys.argv:
        exp_dir = sys.argv[3]
    # Or just single experiment
    else:
        exp_dir = sys.argv[2]

    # Plot phenotype data
    if '-pheno' in sys.argv:

        # Get axis limits from command line
        plot_axis_lb, plot_axis_ub = parse_axis_limits(sys.argv)

        read_and_plot_phenos(config['logging']['data_dir_path'] + exp_dir,
                             config['logging']['winner_file_name'],
                             group=True if '-group' in sys.argv else False,
                             colour_params=True if '-colour_params' in sys.argv
                                                else False,
                             print_numpy_arrays=True
                                if '-print_numpy_arrays' in sys.argv else False,
                             verbosity=True if '-verbosity' in sys.argv else False,
                             plot_axis_lb=plot_axis_lb,
                             plot_axis_ub=plot_axis_ub)

    # Plot evolutionary run data
    elif '-evo' in sys.argv:

        # Split comma separated experiment directories
        exp_data_dirs = exp_dir.split(',')

        read_and_plot_evo_data(exp_data_dirs, config['logging']['data_dir_path'],
                               config['logging']['winner_file_name'],
                               # Turns off the plotting of the inter-quartile ranges
                               False if '-q_means_off' in sys.argv else True,
                               False if '-q_bests_off' in sys.argv else True,
                               True if '-verbosity' in sys.argv else False)
