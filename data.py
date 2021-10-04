import os
import sys
import csv
import torch
from glob import glob

def get_train_folders(folders_dir, dir_path):

    dir_path += folders_dir

    #Get all folder names
    folder_names = [x[0] for x in os.walk(dir_path)][1:]

    return folder_names

def read_data(data_dir, as_torch_tensor=True,
              dir_path='../IndirectEncodingsExperiments/lib/NeuroEvo/data/'):

    folder_paths = get_train_folders(data_dir, dir_path)

    if len(folder_paths) == 0:
        print("No folders where you have specified read from")
        print("data_dir:", data_dir)
        print("dir_path:", dir_path)
        sys.exit(1)

    data = []

    for fp in folder_paths:
        fp += "/best_winner_so_far"

        try:
            with open(fp) as data_file:

                csv_reader = csv.reader(data_file, delimiter=',')

                #Get first row
                d = next(csv_reader)
                #Convert to floats
                d = [float(i) for i in d]
                #Remove fitness
                del d[0]
                data.append(d)

        except FileNotFoundError:

            sys.exit("Could not find file named: " + fp)

    if as_torch_tensor:
        return torch.Tensor(data)
    else:
        return data

def dump_data(data, dir_path, file_name):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + file_name

    with open(file_path, 'w') as f:

        for d in data:
            f.write(str(d) + '\n')

        f.close()


#Calculates the name of the directory to store exp data in, it is looking for
#exp_*next integer*
def create_exp_dir_name(base_path):

    exp_full_dirs = glob(base_path + "/*")

    #max_exp_num = None
    max_exp_num = 0

    if len(exp_full_dirs) == 0:
        max_exp_num = 0

    else:
        for ed in exp_full_dirs:

            #Get exp numbers in directory
            split_path = ed.split("/")
            exp_string = split_path[-1]
            try:
                exp_num = int(exp_string.split("_")[-1])
            except ValueError:
                continue

            #Find largest exp number
            if max_exp_num is None:
                max_exp_num = exp_num
            else:
                if exp_num > max_exp_num:
                    max_exp_num = exp_num

    return 'exp_' + str(max_exp_num+1)

def create_synthetic_data(code_size, num_data_points=500):

    #return -10. + torch.randn(500, code_size)

    means = torch.zeros(500, 2)
    for i in range(means.size(0)):
        for j in range(means.size(1)):
            if j == 0:
                means[i][j] = 10.
            else:
                means[i][j] = -10.

    return means + torch.randn(500, 2)
