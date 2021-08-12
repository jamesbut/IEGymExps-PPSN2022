#The decoder trained via the C++ code
#It is imported here and used as an indirect encoding

import torch

class Decoder:

    #def __init__(self, num_inputs, num_outputs, num_hidden_layers,
    #             neurons_per_hidden_layer, file_name):
    def __init__(self, file_name):

        '''
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        '''


        #Read decoder
        self.decoder = torch.load(file_name)
        #self.decoder = torch.jit.load(self.path)
        #self.decoder = torch.load(self.path)
        #self.decoder = self._build_nn()
        #print("Decoder: ", self.decoder)
        #self.decoder.load_state_dict(torch.jit.load(self.path))


    def decode(self, genotype):
        output = self.decoder.forward(torch.Tensor(genotype))
        return output.detach().numpy()

    def get_num_inputs(self):
        return self.decoder.l1.in_features

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
        #layers.append(torch.nn.Sigmoid())

        return torch.nn.Sequential(*layers).double()
