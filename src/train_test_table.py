#######################################################################################
# Create a train test table that tests a number of trained controllers on a selection #
# of test parameters                                                                  #
#######################################################################################

import sys
import numpy as np
from main import indv_run
from formatting import format_data_table, list_to_string
from itertools import product, filterfalse, chain
from helper import more_than_one_true, lists_from_bools
from data import read_agent_data, read_configs, get_sub_folders
import copy

np.set_printoptions(suppress=True)


def find_trained_solutions(train_exp_paths, winner_file_name):

    train_paths = []
    train_params = []

    for train_path in train_exp_paths:
        fitnesses, _, _, params, paths = read_agent_data(train_path,
                                                         winner_file_name)

        # If more than one solution has been read in, select solution with highest
        # fitness
        arg_fit_max = np.argmax(fitnesses)
        train_paths.append(paths[arg_fit_max])
        train_params.append(params[arg_fit_max].tolist())

    return train_paths, train_params


def test_solutions(train_paths, train_params, test_params, env_seed, render):

    rewards = []

    # Do individual runs on test parameters
    for train_info in zip(train_paths, train_params):
        print('Testing solution {} trained on {}'.format(train_info[0], train_info[1]))

        train_rewards = indv_run(train_info[0], test_params, env_seed, render=False)
        rewards.append(train_rewards)

    return np.array(rewards)


def add_train_to_test(test_params, train_params):

    for train_param in train_params:
        if (len(train_param) == 1) and (train_param[0] not in test_params):
            test_params.append(train_param[0])

    return test_params


# Parse and format experiment train directories from command line
def parse_train_dirs(argv, data_dir_path):

    # If -groups arg specified, groups of experiments have been declared
    if '--groups' in argv:
        groups_index = argv.index('--groups')
        # Read experiment group names
        group_dirs = argv[groups_index + 1]
        # Split csv group names
        group_dirs = group_dirs.split(',')
        # Append data directory path as prefix
        group_paths = map(lambda gd: data_dir_path + gd, group_dirs)

        # Get experiment directories
        exp_dirs = [get_sub_folders(group_path, recursive=False)
                    for group_path in group_paths]
        exp_dirs = chain.from_iterable(exp_dirs)

    else:
        exp_dirs = argv[1]
        # Trained solution directories should be comma seperated
        exp_dirs = exp_dirs.split(',')
        # Append data directory path
        exp_dirs = map(lambda ed: data_dir_path + ed, exp_dirs)

    return exp_dirs


# Argument should be a comma separated list of either: a single solution
# directory; an experiment directory of which the solution with the
# highest fitness will be selected; or a group of experiments directory
def train_test_table(argv, test_params, data_dir_path, winner_file_name, env_seed):

    # Either read in the trained models
    if len(argv) >= 2:
        # Parse command line for train directories
        train_exp_paths = list(parse_train_dirs(argv, data_dir_path))

        # Find trained solutions
        train_paths, train_params = find_trained_solutions(train_exp_paths,
                                                           winner_file_name)

    # Or train them
    elif len(argv) == 1:
        # TODO
        pass

    # Add train params to test params if they are not already there
    test_params = add_train_to_test(test_params, train_params)

    # Test solutions
    rewards = test_solutions(train_paths, train_params, test_params,
                             env_seed, render=False)

    # Get power set of test params so that we can calculate all appropriate means
    pow_set_bools = product([True, False], repeat=len(test_params))
    # Remove from power set all tuples with less than two Trues
    filtered_bools = filterfalse(lambda x: not more_than_one_true(x), pow_set_bools)
    # Get test parameters according to boolean tuples
    filtered_test_params = lists_from_bools(test_params, copy.deepcopy(filtered_bools))
    filtered_test_params.reverse()

    # Calculate relevant means
    pow_set_means = []
    for b in filtered_bools:
        pow_set_means.append(np.mean(rewards, axis=1, where=b))
    pow_set_means = np.fliplr(np.array(pow_set_means).T)
    rewards_w_means = np.concatenate((rewards, pow_set_means), axis=1)

    # Calculate the mean scores of each test environment
    test_means = np.mean(rewards_w_means, axis=0)
    rewards_w_means = np.row_stack((rewards_w_means, test_means))

    # Build axis strings
    test_params_str = list(map(str, test_params))
    # Convert lists to strings
    test_params_str += list(map(list_to_string, filtered_test_params))
    train_params_str = list(map(list_to_string, train_params))

    # Remove data directory path from data directory paths
    train_exp_dirs = list(map(lambda ep: ep.removeprefix(data_dir_path),
                              train_exp_paths))
    # Append trains dirs to train params string
    train_params_str = list(map(lambda td, tp: td + ':' + tp,
                                train_exp_dirs, train_params_str))
    train_params_str.append('test means')

    # Format results in table
    format_data_table(rewards_w_means, train_params_str, test_params_str,
                      row_axis='train', column_axis='test', show_all_columns=True,
                      dump_file_path=data_dir_path + 'train_test_table.csv')


if __name__ == "__main__":

    config = read_configs(sys.argv)[0]

    test_parameters = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

    train_test_table(sys.argv, test_parameters,
                     config['logging']['data_dir_path'],
                     config['logging']['winner_file_name'],
                     config['env'].get('seed', None))
