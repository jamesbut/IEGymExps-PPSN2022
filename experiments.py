#Some experiments that I might need to run frequently

import sys
import csv
import numpy as np
from data import read_data, get_sub_folders
from domain_params import get_env_kwargs
from main import env_name, indv_run

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
        train_params.append(params[arg_fit_max])

    return train_paths, train_params


def test_solutions(train_paths, train_params, test_params):

    #Do individual runs on test parameters
    for t in zip(train_paths, train_params):
        print('Testing solution trained on:', t[1])

        env_kwargs = get_env_kwargs(env_name, domain_params=test_params)
        reward = indv_run(t[0], env_kwargs)



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

    test_solutions(train_paths, train_params, test_params)

if __name__ == "__main__":

    test_parameters = [0.001000, 0.001001]

    train_test_table(sys.argv, test_parameters)
