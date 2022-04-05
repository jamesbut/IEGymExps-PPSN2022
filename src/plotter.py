import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Optional
from data import read_agent_data, read_evo_data, get_sub_folders
from command_line import parse_axis_limits, read_configs, retrieve_flag_args

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


def _plot_exp_evo_data(mean_bests, median_bests, lq_bests, uq_bests, best_bests,
                       mean_means, median_means, lq_means, uq_means, colour,
                       plot_mean_bests: bool = False, plot_median_bests: bool = True,
                       plot_q_bests: bool = False, plot_best_bests: bool = True,
                       plot_mean_means: bool = True, plot_median_means: bool = False,
                       plot_q_means: bool = False, x_axis_max: Optional[int] = None):

    # Inner class to build plot data
    class PlotData:

        def __init__(self, x_axis: list, x_axis_max: Optional[int]):

            self.data = np.empty([0, len(x_axis), 2])
            self.line_styles: list[str] = []
            self.line_widths: list[float] = []
            self.gens = x_axis
            self.x_axis_max = x_axis_max

        def add_data(self, data: np.ndarray, line_style: str, line_width: float):

            # Prepare data for plotting
            data_stacked = np.column_stack((self.gens, data))

            # Concatenate with old data
            self.data = np.concatenate((self.data, np.array([data_stacked])))

            # Concatenate new line styles and widths with old
            self.line_styles += [line_style]
            self.line_widths += [line_width]

        def plot(self):
            # Plot data
            for i in range(self.data.shape[0]):
                plt.plot(self.data[i, :self.x_axis_max, 0],
                         self.data[i, :self.x_axis_max, 1],
                         color=colour, linestyle=self.line_styles[i],
                         linewidth=self.line_widths[i])

    # Create x axis of generations
    gens = np.arange(1, median_bests.shape[0] + 1)

    data = PlotData(gens, x_axis_max)

    # Plot best winner so far data
    if plot_mean_bests:
        data.add_data(mean_bests, '-', 1.)
    if plot_median_bests:
        data.add_data(median_bests, '-', 1.)
    if plot_q_bests:
        data.add_data(lq_bests, '--', 0.25)
        data.add_data(uq_bests, '--', 0.25)
    if plot_best_bests:
        data.add_data(best_bests, ':', 1.)

    # Plot population average data
    if plot_mean_means:
        data.add_data(mean_means, '--', 1.)
    if plot_median_means:
        data.add_data(median_means, '--', 1.)
    if plot_q_means:
        data.add_data(lq_means, '--', 0.25)
        data.add_data(uq_means, '--', 0.25)

    # Plot data
    data.plot()

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

    print('exp_data_path:', exp_data_path)
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


# Determines which experiment to plot when a sub group of experiments is given

# gen_one_max is a boolean which determines which experiment in the sub group to use.
# If gen_one_max is false, the experiment with the best winner so far with the highest
# fitness is chosen, otherwise the experiment with the highest best winner so far
# fitness at generation 1 is chosen
def _determine_experiment_to_plot(exp_data_path: str, winner_file_name: str,
                                  gen_one_max: bool, verbosity: bool) -> str:

    # Check whether exp_data_path is an experiment directory or subgroup directory
    if 'exp' in exp_data_path.split('/')[-1]:
        return exp_data_path

    # A subgroup directory
    else:

        # Get subgroup experiment paths
        group_exp_data_paths = get_sub_folders(exp_data_path,
                                               recursive=False,
                                               sort_by_suffix_num=True)

        max_fitnesses = []
        for edp in group_exp_data_paths:

            # Calculate run with greatest best fitness so far at generation 1
            if gen_one_max:

                _, best_fitnesses = read_evo_data(edp)
                max_fitnesses.append(np.max(best_fitnesses[:, 0]))

            # Calculate max fitness of best winners so far for the experiments
            # in the group
            else:

                max_fitness, _, _ = _read_exp(edp, winner_file_name, verbosity, False)
                max_fitnesses.append(max_fitness)

        if verbosity:
            print('Max fitnesses:', max_fitnesses)

        return group_exp_data_paths[np.argmax(max_fitnesses)]


# Prompt for legend lables
def _prompt_legend_labels(exp_data_dirs) -> list[str]:

    legend_labels: list[str] = []

    for exp_data_dir in exp_data_dirs:
        print(exp_data_dir)
        legend_labels.append(input('Legend label? '))

    return legend_labels


def read_and_plot_evo_data(exp_data_dirs, data_dir_path, winner_file_name,
                           gen_one_max: bool = False, plot_mean_bests: bool = False,
                           plot_median_bests: bool = True, plot_q_bests: bool = False,
                           plot_best_bests: bool = True, plot_mean_means: bool = True,
                           plot_median_means: bool = False, plot_q_means: bool = False,
                           verbosity: bool = False, prompt_legend_labels: bool = False,
                           x_axis_max: Optional[int] = None):

    # Prefix exp data directory with data path
    exp_data_paths = [data_dir_path + edd for edd in exp_data_dirs]

    exp_plot_colours = ['b', 'r', 'g', 'm', 'y', 'c']
    legend_items: list[str] = []
    # Prompt for legend labels
    legend_labels: Optional[list[str]] = None
    if prompt_legend_labels:
        legend_labels = _prompt_legend_labels(exp_data_dirs)

    for i, exp_data_path in enumerate(exp_data_paths):

        # Determine which experiment to plot
        exp_data_path = _determine_experiment_to_plot(exp_data_path, winner_file_name,
                                                      gen_one_max, verbosity)
        print('Experiment data path:', exp_data_path)

        # Read experiment data
        mean_fitnesses, best_fitnesses = read_evo_data(exp_data_path)
        if verbosity:
            print('Best fitnesses of run:\n', best_fitnesses)

        # Calculate best fitnesses so far
        best_fitnesses_so_far = calculate_best_fitnesses_so_far(best_fitnesses)

        # Print number of runs
        print('Number of runs:', len(best_fitnesses))

        # Calculate statistics
        mean_best_fitnesses = np.mean(best_fitnesses_so_far, axis=0)
        median_best_fitnesses = np.median(best_fitnesses_so_far, axis=0)
        lq_best_fitnesses = np.quantile(best_fitnesses_so_far, 0.25, axis=0)
        uq_best_fitnesses = np.quantile(best_fitnesses_so_far, 0.75, axis=0)
        best_best_fitnesses = np.max(best_fitnesses_so_far, axis=0)

        mean_mean_fitnesses = np.mean(mean_fitnesses, axis=0)
        median_mean_fitnesses = np.median(mean_fitnesses, axis=0)
        lq_mean_fitnesses = np.quantile(mean_fitnesses, 0.25, axis=0)
        uq_mean_fitnesses = np.quantile(mean_fitnesses, 0.75, axis=0)

        # Plot experiment data
        _plot_exp_evo_data(mean_best_fitnesses, median_best_fitnesses,
                           lq_best_fitnesses, uq_best_fitnesses, best_best_fitnesses,
                           mean_mean_fitnesses, median_mean_fitnesses,
                           lq_mean_fitnesses, uq_mean_fitnesses,
                           exp_plot_colours[i], plot_mean_bests, plot_median_bests,
                           plot_q_bests, plot_best_bests, plot_mean_means,
                           plot_median_means, plot_q_means, x_axis_max)

        # Set legend label
        if legend_labels:
            legend_label = legend_labels[i]
        else:
            legend_label = exp_data_path.replace(data_dir_path, '')
        legend_items.append(
            mpatches.Patch(color=exp_plot_colours[i], label=legend_label)
        )

        print('-' * 50)

    plt.legend(handles=legend_items)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    plt.show()


if __name__ == '__main__':

    config = read_configs(sys.argv)[0]

    # Can pass in entire experiment group
    if '--group' in sys.argv:
        exp_dir = sys.argv[3]
    # Or just single experiment
    else:
        exp_dir = sys.argv[2]

    # Plot phenotype data
    if '--pheno' in sys.argv:

        # Get axis limits from command line
        plot_axis_lb, plot_axis_ub = parse_axis_limits(sys.argv)

        read_and_plot_phenos(config['logging']['data_dir_path'] + exp_dir,
                             config['logging']['winner_file_name'],
                             group=True if '--group' in sys.argv else False,
                             colour_params=True if '--colour-params' in sys.argv
                                                else False,
                             print_numpy_arrays=True
                                if '--print-numpy-arrays' in sys.argv else False,
                             verbosity=True if '--verbosity' in sys.argv else False,
                             plot_axis_lb=plot_axis_lb,
                             plot_axis_ub=plot_axis_ub)

    # Plot evolutionary run data
    elif '--evo' in sys.argv:

        ######################
        # Parse what to plot #
        ######################

        # Plot either mean or median of best winners so far
        plot_mean_bests = True if '--mean-bests-on' in sys.argv else False
        plot_median_bests = not plot_mean_bests
        # Plot IQR of best winners so far
        plot_q_bests = True if '--q-bests-on' in sys.argv else False
        # Plot best run of best winners so far
        plot_b_bests = False if '--best-bests-off' in sys.argv else True

        # Plot either mean or median of average population fitness
        plot_mean_means = False if '--mean-means-off' in sys.argv else True
        plot_median_means = True if '--median-means-on' in sys.argv else False
        # Plot IQR of average population fitness
        plot_q_means = True if '--q-means-on' in sys.argv else False

        # Plot run with maximum best so far fitness as generation 1
        gen_one_max = True if '--gen-one-max' in sys.argv else False

        verbosity = True if '--verbosity' in sys.argv else False

        # Prompot for legend labels
        prompt_legend_labels = True if '--legend-labels' in sys.argv else False

        # x-axis max
        x_axis_max: Optional[list[str]] = retrieve_flag_args('--x-axis-max', sys.argv)
        # Convert to int
        if x_axis_max is not None:
            x_axis_max = int(x_axis_max[0])

        # Split comma separated experiment directories
        exp_data_dirs = exp_dir.split(',')

        read_and_plot_evo_data(
            exp_data_dirs, config['logging']['data_dir_path'],
            config['logging']['winner_file_name'],
            gen_one_max, plot_mean_bests, plot_median_bests, plot_q_bests, plot_b_bests,
            plot_mean_means, plot_median_means, plot_q_means, verbosity,
            prompt_legend_labels, x_axis_max)
