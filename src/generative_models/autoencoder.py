import torch
from generative_models.batch_utils import generate_batches
from generative_models.model_testing import code_in_range
from neural_network import NeuralNetwork


class Autoencoder(torch.nn.Module):

    def __init__(self, code_size=None, train_data_vec_size=None, num_hidden_layers=None,
                 neurons_per_hidden_layer=None, lr=1e-3, decoder_file_path=None):
        super().__init__()

        if decoder_file_path:
            self._decoder = NeuralNetwork(file_path=decoder_file_path)
        else:
            self._encoder = NeuralNetwork(
                num_inputs=train_data_vec_size,
                num_outputs=code_size,
                num_hidden_layers=num_hidden_layers,
                neurons_per_hidden_layer=neurons_per_hidden_layer,
                final_activ_func=None)
            self._decoder = NeuralNetwork(
                num_inputs=code_size,
                num_outputs=train_data_vec_size,
                num_hidden_layers=num_hidden_layers,
                neurons_per_hidden_layer=neurons_per_hidden_layer,
                final_activ_func=None)

        self._optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        self._lr = lr

    def train(self, train_data, num_epochs, batch_size):

        self._num_epochs = num_epochs
        self._batch_size = batch_size

        loss_fn = torch.nn.MSELoss()

        batches = generate_batches(train_data, batch_size, shuffle=True)

        for epoch in range(num_epochs):
            loss = 0
            for batch in batches:

                self._optimiser.zero_grad()
                outputs = self(batch)
                train_loss = loss_fn(outputs, batch)
                train_loss.backward()
                self._optimiser.step()
                loss += train_loss.item()

            loss = loss / len(batches)

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))

        self._final_loss = loss

    def forward(self, inputs, verbosity=False):

        output = self._encoder(inputs)
        code = torch.sigmoid(output)
        output = self._decoder(code)

        if verbosity:
            for i in range(len(inputs)):
                for j in range(inputs[i].size()[0]):
                    print(" {:<12.6f}".format(inputs[i][j].item()), end='')
                print(" | {:<12.6f}".format(code[i].item()), end='')
                print(" | ", end='')
                for j in range(output[i].size()[0]):
                    print(" {:<12.6f}".format(output[i][j].item()), end='')
                print('')

        return output

    def test_decoder(self):

        code_range = code_in_range(self._decoder.num_inputs, 0., 1., step_size=0.004)
        output = self._decoder(code_range)
        return output.detach().numpy()

    def test(self):
        self(self.training_data, verbosity=True)

    def to_dict(self, train_data_exp_dir_paths):

        return {
            'num_epochs': self._num_epochs,
            'batch_size': self._batch_size,
            'final_loss': self._final_loss,
            'learning_rate': self._lr,
            'train_data_exp_dir_paths': train_data_exp_dir_paths,
            'encoder': self._encoder.to_dict(),
            'decoder': self._decoder.to_dict()
        }

    def dump_config(self, config_path, train_data_exp_dir_paths):

        import json

        with open(config_path + '.json', 'w') as f:
            json.dump(self.to_dict(train_data_exp_dir_paths), f, indent=4)

    def dump_decoder(self, file_path):
        self._decoder.save(file_path)
