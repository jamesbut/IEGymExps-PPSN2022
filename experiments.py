#Some experiments that I might need to run frequently

import sys
import csv
import numpy as np
from data import read_data, get_sub_folders
from domain_params import get_env_kwargs
from main import indv_run
from constants import ENV_NAME
from formatting import *

np.set_printoptions(suppress=True)

def find_trained_solutions(train_dirs):

    #Trained solution directories should be comma seperated
    train_dirs = train_dirs.split(',')

    train_paths = []
    train_params = []

    for td in train_dirs:
        fitnesses, genos, phenos, params, paths = read_data(td)

        #If more than one solution has been read in, select solution with highest
        #fitness
        arg_fit_max = np.argmax(fitnesses)
        train_paths.append(paths[arg_fit_max])
        train_params.append(params[arg_fit_max].tolist())

    return train_paths, train_params


def test_solutions(train_paths, train_params, test_params, render):

    rewards = []

    #Do individual runs on test parameters
    for train_info in zip(train_paths, train_params):
        print('Testing solution trained on:', train_info[1])

        train_rewards = indv_run(train_info[0], ENV_NAME, test_params, render=False)
        rewards.append(train_rewards)

    return rewards


'''
Argument should be a comma separated list of either a single solution
directory or an experiment directory of which the solution with the
highest fitness will be selected
'''
def train_test_table(argv, test_params):

    #Either read in the trained models
    if len(argv) == 2:
        train_dirs = argv[1]
        train_paths, train_params = find_trained_solutions(train_dirs)

    #Or train them
    elif len(argv) == 1:
        pass

    #Test solutions
    rewards = test_solutions(train_paths, train_params, test_params, render=False)

    #Format results in table
    test_params_str = list(map(str, test_params))
    train_params_str = list(map(list_to_string, train_params))
    format_data_table(rewards, train_params_str, test_params_str)


if __name__ == "__main__":

    test_parameters = [0.001000, 0.001001]

    train_test_table(sys.argv, test_parameters)
