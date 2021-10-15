import torch

class Decoder:

    def __init__(self, file_name):

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
