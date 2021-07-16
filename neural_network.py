import numpy as np
import torch
import csv
import os
import shutil

class NeuralNetwork():

    def __init__(self, num_inputs, num_outputs,
                 num_hidden_layers=0, neurons_per_hidden_layer=0,
                 genotype=None):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer

        #Build neural net
        self._build_nn()

        #Set genotype as weights
        #If no genotype is given, torch generates random weights
        if genotype is not None:
            self.set_weights(genotype)


    def _build_nn(self):

        layers = []
        if self.num_hidden_layers == 0:
            layers.append(torch.nn.Linear(self.num_inputs, self.num_outputs))

        else:
            layers.append(torch.nn.Linear(self.num_inputs, self.neurons_per_hidden_layer))
            #Hidden layers have ReLU activation
            layers.append(torch.nn.ReLU())

            for i in range(self.num_hidden_layers-1):
                layers.append(torch.nn.Linear(self.neurons_per_hidden_layer,
                                              self.neurons_per_hidden_layer))
                layers.append(torch.ReLU())

            layers.append(torch.nn.Linear(self.neurons_per_hidden_layer,
                                          self.num_outputs))

        #Final layer goes through Sigmoid
        layers.append(torch.nn.Sigmoid())

        self.nn = torch.nn.Sequential(*layers).double()

    #Takes a list, passes through the network and returns a list
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        net_out = self.nn.forward(x)
        return net_out.tolist()

    #Returns the number of weights
    def get_num_weights(self):
        num_weights = 0
        for layer in self.nn:
            for params in layer.parameters():
                num_weights += params.numel()
        return num_weights

    def print_weights(self):
        for layer in self.nn:
            for params in layer.parameters():
                print(params)

    def _set_weights_err_msg(self, weights_len, num_weights_required):
        return "Trying to set {} weights to an NN that requires {} weights" \
            .format(weights_len, num_weights_required)

    #Sets a list of weights
    def set_weights(self, new_weights):

        #Check new weights is of correct size
        num_weights_required = self.get_num_weights()
        assert num_weights_required == len(new_weights), \
                                       self._set_weights_err_msg(len(new_weights), \
                                                                 num_weights_required)

        weight_index = 0
        for layer in self.nn:
            for params in layer.parameters():

                #Slice out new weights
                p_weights = new_weights[weight_index : weight_index + params.numel()]
                weight_index += params.numel()

                #Resize and set new weights
                params.data = torch.tensor(np.reshape(p_weights, params.size()), \
                                           dtype=torch.float64)

    #Return weights as a 1d list
    def get_weights(self):
        weights = []
        for layer in self.nn:
            for params in layer.parameters():
                weights += params.flatten().tolist()
        return weights

    def save_genotype(self, dir_path, file_name):
        #Save genotype as a csv - it is just a list
        file_path = dir_path + file_name

        #Create directory
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)

        with open(file_path, 'w') as outfile:
            csv_writer = csv.writer(outfile)
            #Added default fitness of 0 at the beginning because this is how it is
            #read in on the NeuroEvo side
            fitness_and_weights = [0.] + self.get_weights()
            csv_writer.writerow(fitness_and_weights)

    def read_genotype(self, genotype_filepath):
        with open(genotype_filepath, 'r') as genotype_file:
            reader = csv.reader(genotype_file)
            genotype = list(map(float, list(reader)[0]))
        return genotype
