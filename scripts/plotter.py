import sys
sys.path.append('..')

from data import get_train_folders
import matplotlib.pyplot as plt
import numpy as np
import csv

np.set_printoptions(suppress=True)

def read_data(data_dir):

    dir_path = "../../IndirectEncodingsExperiments/lib/NeuroEvo/data/"

    folder_paths = get_train_folders(data_dir, dir_path)

    data = []

    for fp in folder_paths:
        fp += "/best_winner_so_far"

        try:
            with open(fp) as data_file:

                csv_reader = csv.reader(data_file, delimiter=',')

                #Get first row
                genes = next(csv_reader)
                #Convert to floats
                genes = [float(i) for i in genes]
                #Remove fitness
                del genes[0]

                #Get second row
                params = next(csv_reader)
                params = [float(i) for i in params]

                data.append(genes + params)

        except FileNotFoundError:

            sys.exit("Could not find file named: " + fp)

    return np.array(data)


def plot_data(data):

    plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap='plasma')
    cbar = plt.colorbar()
    plt.show()


if __name__ == '__main__':


    data = read_data(sys.argv[1])

    print(data)

    plot_data(data)
