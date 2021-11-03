import os
import sys
import csv
from glob import glob
import numpy as np


def get_sub_folders(folders_dir,
                    dir_path='../IndirectEncodingsExperiments/lib/NeuroEvo/data/'):

    dir_path += folders_dir

    if os.path.isdir(dir_path):
        # Get all folder names
        folder_names = [x[0] for x in os.walk(dir_path)][1:]

    else:
        raise NotADirectoryError("{} is not a directory".format(dir_path))

    return folder_names, dir_path


def read_data(data_dir, dir_path='../IndirectEncodingsExperiments/lib/NeuroEvo/data/'):

    # Get directories in data_dir
    try:
        folder_paths, dir_path = get_sub_folders(data_dir, dir_path)

        # If there are no directories in data_dir, then look for data in data_dir
        if not folder_paths:
            folder_paths = [dir_path]

    except NotADirectoryError as e:
        print(e)
        sys.exit(1)

    fitnesses = []
    genotypes = []
    params = []
    phenotypes = []

    for i in range(len(folder_paths)):
        folder_paths[i] += '/best_winner_so_far'

        try:
            with open(folder_paths[i]) as data_file:

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
                        params.append(row)

                    # If an IE was used the phenotype is on the third row
                    elif i == 2:
                        phenotypes.append(row)

        except FileNotFoundError:
            sys.exit("Could not find file named: " + folder_paths[i])

    # If phenotypes are not in file, make phenotypes equal to the genotypes
    if not phenotypes:
        import copy
        phenotypes = copy.deepcopy(genotypes)

    return np.array(fitnesses), \
           np.array(genotypes), \
           np.array(phenotypes) if phenotypes else None, \
           np.array(params) if params else None, \
           folder_paths


def dump_data(data, dir_path, file_name):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + file_name

    with open(file_path, 'w') as f:

        for d in data:
            f.write(str(d) + '\n')

        f.close()


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
