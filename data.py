import os
import sys
import csv
import torch

def get_train_folders(folders_dir):

    dir_path = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/"
    dir_path += folders_dir

    #Get all folder names
    folder_names = [x[0] for x in os.walk(dir_path)][1:]

    return folder_names

def read_data(data_dir):

    folder_paths = get_train_folders(data_dir)

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

    return torch.Tensor(data)
