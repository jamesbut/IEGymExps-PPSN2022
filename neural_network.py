import numpy as np
import torch
import csv
import os
import shutil
import json
from decoder import Decoder

class NeuralNetwork():

    def __init__(self, num_inputs=None, num_outputs=None,
                 num_hidden_layers=0, neurons_per_hidden_layer=0,
                 genotype=None, genotype_dir=None, decoder=False,
                 bias=True):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.bias = bias

        #Read genotype and metadata from files
        if genotype_dir is not None:
            genotype = self._read_genotype(genotype_dir)

            metadata = self._read_metadata(genotype_dir)

            self.num_inputs = metadata['num_inputs']
            self.num_outputs = metadata['num_outputs']
            self.num_hidden_layers = metadata['num_hidden_layers']
            self.neurons_per_hidden_layer = metadata['neurons_per_hidden_layer']
            self.bias = metadata['bias']

        #Build neural net
        self._build_nn(self.bias)

        self.decoder = None
        if decoder:
            self.decoder = Decoder("generator.pt")

        #Set genotype as weights
        #If no genotype is given, torch generates random weights
        if genotype is not None:
            #If decoder is true, read in and use decoder to set weights
            if decoder:
                weights = self.decoder.decode(genotype)
                self.set_weights(weights)
            else:
                self.set_weights(genotype)

            self.genotype = genotype

        else:
            self.genotype = self.get_weights()



    def _build_nn(self, bias=True):

        layers = []
        if self.num_hidden_layers == 0:
            layers.append(torch.nn.Linear(self.num_inputs, self.num_outputs,
                                          bias=bias))

        else:
            layers.append(torch.nn.Linear(self.num_inputs,
                                          self.neurons_per_hidden_layer,
                                          bias=bias))
            #Hidden layers have ReLU activation
            layers.append(torch.nn.ReLU())

            for i in range(self.num_hidden_layers-1):
                layers.append(torch.nn.Linear(self.neurons_per_hidden_layer,
                                              self.neurons_per_hidden_layer,
                                              bias=bias))
                layers.append(torch.nn.ReLU())

            layers.append(torch.nn.Linear(self.neurons_per_hidden_layer,
                                          self.num_outputs,
                                          bias=bias))

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

    #Returns size of genotype needed for this NN
    def get_genotype_size(self):
        if self.decoder is None:
            return self.get_num_weights()
        else:
            return self.decoder.get_num_inputs()

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

    #Set genotype - this uses a decoder if there is one as opposed to set_weights
    #which just sets the NN weights
    def set_genotype(self, genotype):

        self.genotype = genotype

        if self.decoder is not None:
            weights = self.decoder.decode(genotype)
            self.set_weights(weights)
        else:
            self.set_weights(genotype)

    #Return weights as a 1d list
    def get_weights(self):
        weights = []
        for layer in self.nn:
            for params in layer.parameters():
                weights += params.flatten().tolist()
        return weights

    def save_genotype(self, dir_path, file_name, fitness, domain_params=None):
        #Save genotype as a csv - it is just a list
        file_path = dir_path + file_name

        #Create directory
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        with open(file_path, 'w') as outfile:
            csv_writer = csv.writer(outfile)
            #Added fitness at the beginning because this is how it is
            #read in on the NeuroEvo side
            fitness_and_genotype = [fitness] + self.genotype
            csv_writer.writerow(fitness_and_genotype)

            #Add domain hyperparameters on next line for cGAN
            if domain_params is not None:
                csv_writer.writerow(domain_params)

        #Also save metadata
        self._save_metadata(file_path)

    def _save_metadata(self, genotype_filepath):
        metadata_filepath = genotype_filepath + '_metadata.json'

        metadata = {}

        metadata['num_inputs'] = self.num_inputs
        metadata['num_outputs'] = self.num_outputs
        metadata['num_hidden_layers'] = self.num_hidden_layers
        metadata['neurons_per_hidden_layer'] = self.neurons_per_hidden_layer
        metadata['bias'] = self.bias

        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f)

    def _read_genotype(self, genotype_filepath):
        with open(genotype_filepath, 'r') as genotype_file:
            reader = csv.reader(genotype_file)
            genotype = list(map(float, list(reader)[0]))

        #Remove fitness
        del genotype[0]

        return genotype

    def _read_metadata(self, genotype_filepath):
        metadata_filepath = genotype_filepath + '_metadata.json'

        with open(metadata_filepath, 'r') as f:
            metadata = json.load(f)

        return metadata

