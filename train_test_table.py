#######################################################################################
# Create a train test table that tests a number of trained controllers on a selection #
# of test parameters                                                                  #
#######################################################################################

import sys
import numpy as np
from main import indv_run
from constants import ENV_NAME
from formatting import format_data_table, list_to_string
from itertools import product, filterfalse
from helper import more_than_one_true, lists_from_bools
from data import read_agent_data
import copy

np.set_printoptions(suppress=True)


def find_trained_solutions(train_dirs):

    # Trained solution directories should be comma seperated
    train_dirs = train_dirs.split(',')

    train_paths = []
    train_params = []

    for td in train_dirs:
        fitnesses, _, _, params, paths = read_agent_data(td)

        # If more than one solution has been read in, select solution with highest
        # fitness
        arg_fit_max = np.argmax(fitnesses)
        train_paths.append(paths[arg_fit_max])
        train_params.append(params[arg_fit_max].tolist())

    return train_paths, train_params


def test_solutions(train_paths, train_params, test_params, render):

    rewards = []

    # Do individual runs on test parameters
    for train_info in zip(train_paths, train_params):
        print('Testing solution trained on:', train_info[1])

        train_rewards = indv_run(train_info[0], ENV_NAME, test_params, render=False)
        rewards.append(train_rewards)

    return np.array(rewards)


def add_train_to_test(test_params, train_params):

    for train_param in train_params:
        if (len(train_param) == 1) and (train_param[0] not in test_params):
            test_params.append(train_param[0])

    return test_params


# Argument should be a comma separated list of either a single solution
# directory or an experiment directory of which the solution with the
# highest fitness will be selected
def train_test_table(argv, test_params):

    # Either read in the trained models
    if len(argv) == 2:
        train_dirs = argv[1]
        train_paths, train_params = find_trained_solutions(train_dirs)

    # Or train them
    elif len(argv) == 1:
        # TODO
        pass

    # Add train params to test params if they are not already there
    test_params = add_train_to_test(test_params, train_params)

    # Test solutions
    rewards = test_solutions(train_paths, train_params, test_params, render=False)

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
    test_params_str += list(map(list_to_string, filtered_test_params))
    train_params_str = list(map(list_to_string, train_params))
    train_params_str.append('test means')

    # Format results in table
    format_data_table(rewards_w_means, train_params_str, test_params_str,
                      row_axis='train', column_axis='test')


if __name__ == "__main__":

    test_parameters = [0.001000, 0.001001, 0.002]

    train_test_table(sys.argv, test_parameters)
