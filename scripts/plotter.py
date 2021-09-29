import sys
sys.path.append('..')

from data import get_train_folders
import matplotlib.pyplot as plt
import numpy as np
import csv

np.set_printoptions(suppress=True)

def read_data(data_dir, parent_dir=False):

    if parent_dir:
        dir_path = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/"
    else:
        dir_path = "../../IndirectEncodingsExperiments/lib/NeuroEvo/data/"

    folder_paths = get_train_folders(data_dir, dir_path)

    data = []
    params_included = False

    for fp in folder_paths:
        fp += "/best_winner_so_far"

        try:
            with open(fp) as data_file:

                csv_reader = csv.reader(data_file, delimiter=',')

                d = []
                for i, row in enumerate(csv_reader):
                    #Convert to floats
                    row = [float(i) for i in row]
                    #Delete fitness in first row
                    if i == 0:
                        del row[0]

                    d += row

                    if i == 1:
                        params_included = True

                data.append(d)

        except FileNotFoundError:
            sys.exit("Could not find file named: " + fp)

    return np.array(data), params_included


def plot_data(data, params_included, test_data=None):

    if params_included:
        plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap='plasma')
        cbar = plt.colorbar()
    else:
        plt.scatter(data[:,0], data[:,1])

    if test_data is not None:
        plt.scatter(test_data[:,0], test_data[:,1])

    plt.show()

def read_and_plot(train_data_dir, test_data=None, parent_dir=False):
    data, params_included = read_data(train_data_dir, parent_dir)
    plot_data(data, params_included, test_data)

if __name__ == '__main__':

    read_and_plot(sys.argv[1])
