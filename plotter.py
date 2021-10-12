import sys
sys.path.append('..')

from data import get_train_folders
import matplotlib.pyplot as plt
import numpy as np
import csv

np.set_printoptions(suppress=True)

def __read_data(data_dir):

    dir_path = "../IndirectEncodingsExperiments/lib/NeuroEvo/data/"

    try:
        folder_paths = get_train_folders(data_dir, dir_path)
    except NotADirectoryError as e:
        print(e)
        sys.exit(1)

    fitnesses = []
    genotypes = []
    params = []

    for fp in folder_paths:
        fp += "/best_winner_so_far"

        try:
            with open(fp) as data_file:

                csv_reader = csv.reader(data_file, delimiter=',')

                for i, row in enumerate(csv_reader):

                    #Convert to floats
                    row = [float(i) for i in row]

                    #First row is fitness and genotype
                    if i == 0:
                        fitnesses.append(row[0])
                        genotypes.append(row[1:])

                    #Second row is parameters, if they are there
                    elif i == 1:
                        params.append(row)

                    #If an IE was used the phenotype is on the third row
                    elif i == 2:
                        del genotypes[-1]
                        genotypes.append(row)

        except FileNotFoundError:
            sys.exit("Could not find file named: " + fp)

    return np.array(fitnesses), np.array(genotypes), \
           np.array(params) if params else None, \
           folder_paths


def __plot_data(train_genotypes=None, params=None, test_genotypes=None):

    if train_genotypes is not None:
        if params is not None:
            plt.scatter(train_genotypes[:,0], train_genotypes[:,1],
                        c=params[:,0], cmap='plasma')
            cbar = plt.colorbar()
        else:
            plt.scatter(train_genotypes[:,0], train_genotypes[:,1])

    if test_genotypes is not None:
        plt.scatter(test_genotypes[:,0], test_genotypes[:,1])

    plt.show()


def __fitness_analysis(fitnesses, folder_paths):

    max_arg = np.argmax(fitnesses)
    max_fitness = fitnesses[max_arg]
    max_file = folder_paths[max_arg]

    print("Max fitness: {}              File: {}".format(max_fitness, max_file))

    min_arg = np.argmin(fitnesses)
    min_fitness = fitnesses[min_arg]
    min_file = folder_paths[min_arg]

    print("Min fitness: {}              File: {}".format(min_fitness, min_file))

    mean_fitness = np.mean(fitnesses)
    print("Mean fitness:", mean_fitness)


def read_and_plot(train_data_dir=None, test_genotypes=None):

    #Read training data
    if train_data_dir is not None:
        fitnesses, genotypes, params, folder_paths = __read_data(train_data_dir)

    print("Fitnesses:\n", fitnesses)
    print("Genotypes:\n", genotypes)
    print("Params:\n", params)

    #Provide fitness analysis
    __fitness_analysis(fitnesses, folder_paths)

    #Plot training and/or test data
    __plot_data(genotypes, params, test_genotypes)

if __name__ == '__main__':

    read_and_plot(sys.argv[1])
