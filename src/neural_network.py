import numpy as np
import torch
import sys
import copy
from pandas.api.types import is_list_like


class NeuralNetwork(torch.nn.Module):

    def __init__(self, num_inputs=None, num_outputs=None,
                 num_hidden_layers=0, neurons_per_hidden_layer=0,
                 hidden_activ_func='relu', final_activ_func='sigmoid',
                 bias=True, w_lb=None, w_ub=None, enforce_wb=True,
                 file_path=None):

        super(NeuralNetwork, self).__init__()

        if not file_path:

            self._num_inputs = num_inputs
            self._num_outputs = num_outputs
            self._num_hidden_layers = num_hidden_layers
            self._neurons_per_hidden_layer = neurons_per_hidden_layer
            self._bias = bias
            self._hidden_activ_func = hidden_activ_func
            self._final_activ_func = final_activ_func

            # Build neural net
            self._nn = self._build_nn()

        else:
            self._nn = self._read(file_path)

        # Set weight bounds
        self._w_lb = copy.copy(w_lb)
        self._w_ub = copy.copy(w_ub)
        # If the length of the weight bounds is one,
        # expand list to number of weights
        if self._w_lb and (len(self._w_lb) == 1):
            self._w_lb *= self.num_weights
        if self._w_ub and (len(self._w_ub) == 1):
            self._w_ub *= self.num_weights

        self._enforce_wb = enforce_wb

    # Build torch network from specification
    def _build_nn(self):

        layers = []
        if self._num_hidden_layers == 0:
            layers.append(torch.nn.Linear(self._num_inputs, self._num_outputs,
                                          bias=self._bias))

        else:
            layers.append(torch.nn.Linear(self._num_inputs,
                                          self._neurons_per_hidden_layer,
                                          bias=self._bias))
            # Hidden layers have ReLU activation
            layers.append(self._activ_func_from_string(self._hidden_activ_func))

            for i in range(self._num_hidden_layers - 1):
                layers.append(torch.nn.Linear(self._neurons_per_hidden_layer,
                                              self._neurons_per_hidden_layer,
                                              bias=self._bias))
                layers.append(self._activ_func_from_string(self._hidden_activ_func))

            layers.append(torch.nn.Linear(self._neurons_per_hidden_layer,
                                          self._num_outputs,
                                          bias=self._bias))

        # Final layer goes through Sigmoid
        if self._final_activ_func:
            layers.append(self._activ_func_from_string(self._final_activ_func))

        return torch.nn.Sequential(*layers)

    # Takes an int, float, list, numpy array or torch Tensor,
    # passes it through the network and returns a torch Tensor
    def forward(self, x):
        # Convert to list
        if not is_list_like(x):
            x = [x]
        # Convert to torch float array
        if not isinstance(x, torch.FloatTensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        net_out = self._nn.forward(x)
        return net_out

    @property
    def num_inputs(self):
        return self._nn[0].in_features

    # Returns the number of weights
    @property
    def num_weights(self):
        total_weights = 0
        for layer in self._nn:
            for params in layer.parameters():
                total_weights += params.numel()
        return total_weights

    # Return weights as a 1d list
    @property
    def weights(self):
        w = []
        for layer in self._nn:
            for params in layer.parameters():
                w += params.flatten().tolist()
        return w

    # Sets a list of weights
    # This also checks the new weights against a weight lower and upper bound
    @weights.setter
    def weights(self, new_weights):

        # Convert to numpy array if torch Tensor is given
        if isinstance(new_weights, torch.Tensor):
            new_weights = new_weights.numpy()

        # Check new weights is of correct size
        num_weights_required = self.num_weights
        assert num_weights_required == len(new_weights), \
                                       self._set_weights_err_msg(len(new_weights),
                                                                 num_weights_required)

        # Bound weights
        if ((self._w_lb is not None) or (self._w_ub is not None)) and self._enforce_wb:
            new_weights = self._bound_weights(new_weights, self._w_lb, self._w_ub)

        weight_index = 0
        for layer in self._nn:
            for params in layer.parameters():

                # Slice out new weights
                p_weights = new_weights[weight_index:weight_index + params.numel()]
                weight_index += params.numel()

                # Resize and set new weights
                params.data = torch.tensor(np.reshape(p_weights, params.size()),
                                           dtype=torch.float32)

    # Bound weights between upper and lower bounds
    def _bound_weights(self, weights, w_lb, w_ub):

        self._check_bounds_size(weights, w_lb, w_ub)

        # If weight exceeds bounds, set weight to bound
        weights = np.maximum(weights, w_lb)
        weights = np.minimum(weights, w_ub)

        return weights

    # Returns True if ANY weight exceeds its bound
    def bounds_exceeded(self):

        # Return False if bounds are None
        if self._w_lb is None and self._w_ub is None:
            return False

        # I know this is very C, but I couldn't think of another way and it just makes
        # sense :/
        w = copy.deepcopy(self.weights)
        for i in range(len(w)):
            if w[i] <= self._w_lb[i]:
                return True
            if w[i] >= self._w_ub[i]:
                return True
        return False

    def _check_bounds_size(self, weights, w_lb, w_ub):

        # Check bounds are the same size as weights
        if len(weights) is not len(w_lb):
            print("neural_network.py _check_bounds_size(): weights length is not the "
                  "same as weights lower bound length")
            print("Weights: {}, Lower bounds: {}, Weights length: {}, "
                  "Lower bounds length: {}".format(weights, w_lb,
                                                   len(weights), len(w_lb)))
            sys.exit(1)
        if len(weights) is not len(w_ub):
            print("neural_network.py _check_bounds_size(): weights length is not the "
                  "same as weights upper bound length")
            print("Weights: {}, Upper bounds: {}, Weights length: {}, "
                  "Upper bounds length: {}".format(weights, w_ub,
                                                   len(weights), len(w_ub)))
            sys.exit(1)

    def _set_weights_err_msg(self, weights_len, num_weights_required):
        return "Trying to set {} weights to an NN that requires {} weights" \
            .format(weights_len, num_weights_required)

    def save(self, file_path):
        torch.save(self._nn, file_path)

    def _read(self, file_path):
        return torch.load(file_path)

    def to_dict(self):

        network_dict = {
            'num_inputs': self._num_inputs,
            'num_outputs': self._num_outputs,
            'num_hidden_layers': self._num_hidden_layers,
            'neurons_per_hidden_layer': self._neurons_per_hidden_layer,
            'hidden_layer_activation_function': self._hidden_activ_func,
            'final_layer_activation_function': self._final_activ_func,
            'bias': self._bias,
            'weights': self.weights
        }

        return network_dict

    # Get activation function from specifying string
    def _activ_func_from_string(self, func_string):

        if func_string == 'relu':
            return torch.nn.ReLU()
        elif func_string == 'sigmoid':
            return torch.nn.Sigmoid()
        else:
            raise KeyError('Activation function: {} not known'.format(func_string))
