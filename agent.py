import csv
import os
import shutil
import json
from neural_network import NeuralNetwork


class Agent():

    def __init__(self, num_inputs=None, num_outputs=None,
                 num_hidden_layers=0, neurons_per_hidden_layer=0,
                 genotype=None, agent_path=None, decoder=None,
                 bias=True, w_lb=None, w_ub=None, enforce_wb=True):

        # If an agent path is given, read from file
        if agent_path:
            genotype, self._decoder, self._network = self._read(agent_path)

        # Otherwise build network from given arguments
        else:

            self._decoder = decoder

            self._fitness = None

            self._network = NeuralNetwork(num_inputs, num_outputs, num_hidden_layers,
                                          neurons_per_hidden_layer, bias, w_lb, w_ub,
                                          enforce_wb)
            self._genotype = self._network.weights

        # Setting the genotype pushes the genotype through the decoder (if there is one)
        # and sets the networks weights
        if genotype:
            self.genotype = genotype

    # Takes a list, passes through the network and returns a list
    def forward(self, state):
        return self._network.forward(state)

    # Returns size of genotype needed for this NN
    @property
    def genotype_size(self):
        if self._decoder is None:
            return self._network.num_weights
        else:
            return self._decoder.num_inputs

    @property
    def genotype(self):
        return self._genotype

    # Set genotype - this pushes genotype through decoder if there is one to get weights
    @genotype.setter
    def genotype(self, genotype):

        self._genotype = genotype

        if self._decoder is not None:
            self._network.weights = self._decoder.forward(genotype)
        else:
            self._network.weights = genotype

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

    # Return bool for success or fail
    def save(self, dir_path, file_name, trained_env_wrapper=None,
             save_if_bounds_exceeded=False):

        # If weight bounds were exceeded do not save
        if not save_if_bounds_exceeded:
            if self._network.bounds_exceeded():
                return False

        # Create directory
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        # Get dictionary representation for serialisation
        agent_data = self.to_dict()
        if trained_env_wrapper is not None:
            agent_data['env'] = trained_env_wrapper.to_dict()

        agent_file_path = dir_path + file_name
        with open(agent_file_path + '.json', 'w') as f:
            json.dump(agent_data, f, indent=4)

        # Save decoder
        if self._decoder:
            self._decoder.save(agent_file_path + '_decoder.pt')

        # Save controller network
        self._network.save(agent_file_path + '_network.pt')

        return True

    def _read_genotype(self, genotype_filepath):
        with open(genotype_filepath, 'r') as genotype_file:
            reader = csv.reader(genotype_file)
            genotype = list(map(float, list(reader)[0]))

        # Remove fitness
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
            # No problem if there is not a decoder
            pass

    def to_dict(self):

        agent_dict = {
            'fitness': self._fitness,
            'genotype': self._genotype,
            'network': self._network.to_dict()
        }

        return agent_dict
