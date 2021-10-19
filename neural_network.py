import numpy as np
import torch
import csv
import os
import shutil
import json

class NeuralNetwork():

    def __init__(self, num_inputs=None, num_outputs=None,
                 num_hidden_layers=0, neurons_per_hidden_layer=0,
                 genotype=None, genotype_path=None, decoder=None,
                 bias=True, w_lb=None, w_ub=None, enforce_wb=True):

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_hidden_layers = num_hidden_layers
        self._neurons_per_hidden_layer = neurons_per_hidden_layer
        self._bias = bias

        self._w_lb = w_lb
        self._w_ub = w_ub
        self._enforce_wb = enforce_wb

        self._decoder = decoder

        #Read genotype, metadata and decoder from files
        if genotype_path is not None:
            genotype = self._read_genotype(genotype_path)
            metadata = self._read_metadata(genotype_path)

            self._num_inputs = metadata['num_inputs']
            self._num_outputs = metadata['num_outputs']
            self._num_hidden_layers = metadata['num_hidden_layers']
            self._neurons_per_hidden_layer = metadata['neurons_per_hidden_layer']
            self._bias = metadata['bias']

            #Read decoder if there is one
            decoder = self._read_decoder(genotype_path)

        #Build neural net
        self._build_nn()

        #Set genotype as weights
        #If no genotype is given, torch generates random weights
        if genotype is not None:
            self.genotype = genotype
        else:
            self.genotype = self.weights


    def _build_nn(self):

        layers = []
        if self._num_hidden_layers == 0:
            layers.append(torch.nn.Linear(self._num_inputs, self._num_outputs,
                                          bias=self._bias))

        else:
            layers.append(torch.nn.Linear(self._num_inputs,
                                          self._neurons_per_hidden_layer,
                                          bias=self._bias))
            #Hidden layers have ReLU activation
            layers.append(torch.nn.ReLU())

            for i in range(self._num_hidden_layers-1):
                layers.append(torch.nn.Linear(self._neurons_per_hidden_layer,
                                              self._neurons_per_hidden_layer,
                                              bias=self._bias))
                layers.append(torch.nn.ReLU())

            layers.append(torch.nn.Linear(self._neurons_per_hidden_layer,
                                          self._num_outputs,
                                          bias=self._bias))

        #Final layer goes through Sigmoid
        layers.append(torch.nn.Sigmoid())

        self._nn = torch.nn.Sequential(*layers).double()


    #Takes a list, passes through the network and returns a list
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float64)
        net_out = self._nn.forward(x)
        return net_out.tolist()


    #Returns the number of weights
    @property
    def num_weights(self):
        total_weights = 0
        for layer in self._nn:
            for params in layer.parameters():
                total_weights += params.numel()
        return total_weights


    #Returns size of genotype needed for this NN
    @property
    def genotype_size(self):
        if self._decoder is None:
            return self.num_weights
        else:
            return self._decoder.l1.in_features


    def print_weights(self):
        for layer in self._nn:
            for params in layer.parameters():
                print(params)


    def _set_weights_err_msg(self, weights_len, num_weights_required):
        return "Trying to set {} weights to an NN that requires {} weights" \
            .format(weights_len, num_weights_required)


    #Return weights as a 1d list
    @property
    def weights(self):
        w = []
        for layer in self._nn:
            for params in layer.parameters():
                w += params.flatten().tolist()
        return w


    #Sets a list of weights
    #This also checks the new weights against a weight lower and upper bound
    @weights.setter
    def weights(self, new_weights):

        #Check new weights is of correct size
        num_weights_required = self.num_weights
        assert num_weights_required == len(new_weights), \
                                       self._set_weights_err_msg(len(new_weights), \
                                                                 num_weights_required)

        #Bound weights
        if ((self._w_lb is not None) or (self._w_ub is not None)) and self._enforce_wb:
            new_weights = self._bound_weights(new_weights, self._w_lb, self._w_ub)

        weight_index = 0
        for layer in self._nn:
            for params in layer.parameters():

                #Slice out new weights
                p_weights = new_weights[weight_index : weight_index + params.numel()]
                weight_index += params.numel()

                #Resize and set new weights
                params.data = torch.tensor(np.reshape(p_weights, params.size()), \
                                           dtype=torch.float64)


    @property
    def genotype(self):
        return self._genotype

    #Set genotype - this uses a decoder if there is one as opposed to set_weights
    #which just sets the NN weights
    @genotype.setter
    def genotype(self, genotype):

        self._genotype = genotype

        if self._decoder is not None:
            weights = self._decode(genotype)
            self.weights = weights
        else:
            self.weights = genotype


    #Decode genotype and apply appropriate type conversions
    def _decode(self, genotype):
        output = self._decoder.forward(torch.Tensor(genotype))
        return output.detach().numpy()


    #Bound weights between upper and lower bounds
    def _bound_weights(self, weights, w_lb, w_ub):

        self._check_bounds_size(weights, w_lb, w_ub)

        #If weight exceeds bounds, set weight to bound
        weights = np.maximum(weights, w_lb)
        weights = np.minimum(weights, w_ub)

        return weights


    #Returns True if ANY weight exceeds its bound
    def _bounds_exceeded(self, weights, w_lb, w_ub):

        #Return False if bounds are None
        if w_lb is None and w_ub is None:
            return False

        #I know this is very C, but I couldn't think of another way and it just makes
        #sense :/
        for i in range(len(weights)):
            if weights[i] <= w_lb[i]:
                return True
            if weights[i] >= w_ub[i]:
                return True
        return False


    def _check_bounds_size(self, weights, w_lb, w_ub):

        #Check bounds are the same size as weights
        if len(weights) is not len(w_lb):
            print("neural_network.py _check_bounds_size(): weights length is not the "
                  "same as weights lower bound length")
            sys.exit(1)
        if len(weights) is not len(w_ub):
            print("neural_network.py _check_bounds_size(): weights length is not the "
                  "same as weights upper bound length")
            sys.exit(1)


    #Return bool for success or fail
    def save(self, dir_path, file_name, fitness, domain_params=None,
             save_if_bounds_exceeded=False):

        #If bounds were exceeded do not save
        if not save_if_bounds_exceeded:
            b_exceeded = self._bounds_exceeded(self.get_weights(),
                                               self._w_lb, self._w_ub)
            if b_exceeded:
                return False

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
            fitness_and_genotype = [fitness] + self._genotype
            csv_writer.writerow(fitness_and_genotype)

            #Add domain hyperparameters on next line for cGAN
            if domain_params is not None:
                csv_writer.writerow(domain_params)

            #Also save phenotype if there is a decoder
            csv_writer.writerow(self.weights)

        #Save metadata
        self._save_metadata(file_path)

        #Save decoder
        self._save_decoder(file_path)

        return True


    def _save_metadata(self, genotype_filepath):
        metadata_filepath = genotype_filepath + '_metadata.json'

        metadata = {}

        metadata['num_inputs'] = self._num_inputs
        metadata['num_outputs'] = self._num_outputs
        metadata['num_hidden_layers'] = self._num_hidden_layers
        metadata['neurons_per_hidden_layer'] = self._neurons_per_hidden_layer
        metadata['bias'] = self._bias

        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f)


    def _save_decoder(self, genotype_filepath):
        decoder_filepath = genotype_filepath + '_decoder.pt'
        torch.save(self._decoder, decoder_filepath)


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


    def _read_decoder(self, genotype_filepath):
        decoder_path = genotype_filepath + '_decoder.pt'
        try:
            self._decoder = torch.load(decoder_path)
        except IOError:
            #No problem if there is not a decoder
            pass
