import os
import sys
import csv
from glob import glob
import numpy as np
import json


# Read useful data about winning agents from experiment
def read_agent_data(exp_data_path, winner_file_name):

    # Get directories in data_dir_path
    run_folder_paths = get_sub_folders(exp_data_path)

    # If there are no directories in exp_data_path, then look for data in exp_data_path
    if not run_folder_paths:
        run_folder_paths = [exp_data_path]

    # Append winner file name to folder paths
    for i in range(len(run_folder_paths)):
        run_folder_paths[i] += '/' + winner_file_name

    # Check for old data format
    if check_for_old_format(run_folder_paths[0]):
        fitnesses, genos, phenos, domain_params = read_data_old_format(run_folder_paths)
    else:
        fitnesses, genos, phenos, domain_params = read_data_new_format(run_folder_paths)

    return fitnesses, genos, phenos, domain_params, run_folder_paths


# Read data from evolutionary runs
def read_evo_data(exp_data_path):

    # Get sub folders in data_dir (different run folders)
    run_folder_paths = get_sub_folders(exp_data_path)

    exp_mean_fitnesses = []
    exp_best_fitnesses = []
    for rfp in run_folder_paths:
        rfp_means = rfp + '/mean_fitnesses'
        rfp_bests = rfp + '/best_fitnesses'

        # Read data and convert to floats
        mean_fitnesses = list(map(float, read_list(rfp_means)))
        best_fitnesses = list(map(float, read_list(rfp_bests)))

        exp_mean_fitnesses.append(mean_fitnesses)
        exp_best_fitnesses.append(best_fitnesses)

    return np.array(exp_mean_fitnesses), np.array(exp_best_fitnesses)


# Check for old data format
def check_for_old_format(data_path):
    try:
        open(data_path)
        return True
    except FileNotFoundError:
        return False


# Read new data format that uses json files
def read_data_new_format(folder_paths):

    fitnesses = []
    genotypes = []
    phenotypes = []
    domain_params = []

    # Append .json to folder paths
    for i in range(len(folder_paths)):
        folder_paths[i] += '.json'

    for fp in folder_paths:
        try:
            with open(fp) as agent_file:
                agent = json.load(agent_file)

                fitnesses.append(agent['fitness'])
                genotypes.append(agent['genotype'])
                phenotypes.append(agent['network']['weights'])
                domain_params.append(agent['env']['domain_params'])

        except FileNotFoundError:
            sys.exit("Could not find file named: " + fp)

    return np.array(fitnesses), np.array(genotypes), np.array(phenotypes), \
           np.array(domain_params)


# Read old data format that was used by NeuroEvo C++ code
def read_data_old_format(folder_paths):

    fitnesses = []
    genotypes = []
    phenotypes = []
    domain_params = []

    for fp in folder_paths:
        try:
            with open(fp) as data_file:

                csv_reader = csv.reader(data_file, delimiter=',')

                for i, row in enumerate(csv_reader):

                    # Convert to floats
                    row = [float(i) for i in row]

                    # First row is fitness and genotype
                    if i == 0:
                        fitnesses.append(row[0])
                        genotypes.append(row[1:])

                    # Second row is parameters, if they are there
                    elif i == 1:
                        domain_params.append(row)

                    # If an IE was used the phenotype is on the third row
                    elif i == 2:
                        phenotypes.append(row)

        except FileNotFoundError:
            sys.exit("Could not find file named: " + fp)

    # If phenotypes are not in file, make phenotypes equal to the genotypes
    if not phenotypes:
        import copy
        phenotypes = copy.deepcopy(genotypes)

    return np.array(fitnesses), \
           np.array(genotypes), \
           np.array(phenotypes) if phenotypes else None, \
           np.array(domain_params) if domain_params else None


def get_sub_folders(dir_path):

    if os.path.isdir(dir_path):
        # Get all folder names
        folder_names = [x[0] for x in os.walk(dir_path)][1:]

    else:
        raise NotADirectoryError("{} is not a directory".format(dir_path))

    return folder_names


# Dump list into a file and check for directory existence
def dump_list(lst, dir_path, file_name):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + file_name

    with open(file_path, 'w') as f:
        for val in lst:
            f.write(str(val) + '\n')


# Read list from file
def read_list(file_path):

    with open(file_path, 'r') as f:
        lst = f.readlines()

    # Remove carriage returns
    lst = [val.strip() for val in lst]

    return lst


# Read json
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Dump dictionary as json
def dump_json(file_path, json_dict):
    with open(file_path, 'w') as file:
        json.dump(json_dict, file, indent=4)


# Calculates the name of the directory to store exp data in, it is looking for
# exp_*next integer*
def create_exp_dir_name(base_path):

    exp_full_dirs = glob(base_path + "/*")

    max_exp_num = 0

    if len(exp_full_dirs) == 0:
        max_exp_num = 0

    else:
        for ed in exp_full_dirs:

            # Get exp numbers in directory
            split_path = ed.split("/")
            exp_string = split_path[-1]
            try:
                exp_num = int(exp_string.split("_")[-1])
            except ValueError:
                continue

            # Find largest exp number
            if max_exp_num is None:
                max_exp_num = exp_num
            else:
                if exp_num > max_exp_num:
                    max_exp_num = exp_num

    return 'exp_' + str(max_exp_num + 1)


def create_synthetic_data(code_size, num_data_points=500):

    # return -10. + np.randn(500, code_size)

    means = np.zeros(500, 2)
    for i in range(means.size(0)):
        for j in range(means.size(1)):
            if j == 0:
                means[i][j] = 10.
            else:
                means[i][j] = -10.

    return means + np.randn(500, 2)
