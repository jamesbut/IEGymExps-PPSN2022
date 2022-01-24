import torch
from generative_models.batch_utils import generate_batches
from neural_network import NeuralNetwork
from generative_models.model_testing import code_in_range
from plotter import read_and_plot_phenos


class VAE(torch.nn.Module):

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

            self._hidden2mu = torch.nn.Linear(code_size, code_size)
            self._hidden2log_var = torch.nn.Linear(code_size, code_size)

            self._decoder = NeuralNetwork(
                num_inputs=code_size,
                num_outputs=train_data_vec_size,
                num_hidden_layers=num_hidden_layers,
                neurons_per_hidden_layer=neurons_per_hidden_layer,
                final_activ_func=None)

        self._optimiser = torch.optim.Adam(self.parameters(), lr=lr)

    def train(self, train_data, num_epochs, batch_size):

        self._num_epochs = num_epochs
        self._batch_size = batch_size

        batches = generate_batches(train_data, batch_size, shuffle=True)

        for epoch in range(num_epochs):
            loss = 0
            for batch in batches:

                self._optimiser.zero_grad()

                mu, log_var = self._encode(batch)
                hidden = self._reparametrise(mu, log_var)
                outputs = self._decode(hidden)

                train_loss = self._loss_function(mu, log_var,
                                                 batch, outputs)
                train_loss.backward()
                self._optimiser.step()

                loss += train_loss.item()

            loss = loss / len(batches)

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))

        self._final_loss = loss

    def forward(self, inputs, verbosity=False):

        mu, log_var = self._encode(inputs)
        hidden = self._reparametrise(mu, log_var)
        output = self._decoder(hidden)

        if verbosity:
            for i in range(len(inputs)):
                for j in range(inputs[i].size()[0]):
                    print(" {:<12.6f}".format(inputs[i][j].item()), end='')
                print(" | ", end='')
                for j in range(mu[i].size()[0]):
                    print(" {:<12.6f}".format(mu[i][j].item()), end='')
                print(" | ", end='')
                for j in range(log_var[i].size()[0]):
                    print(" {:<12.6f}".format(log_var[i][j].item()), end='')
                print(" | ", end='')
                for j in range(hidden[i].size()[0]):
                    print(" {:<12.6f}".format(hidden[i].item()), end='')
                print(" | ", end='')
                for j in range(output[i].size()[0]):
                    print(" {:<12.6f}".format(output[i][j].item()), end='')
                print('')

        return output

    def _encode(self, x):

        hidden = self._encoder(x)
        mu = self._hidden2mu(hidden)
        log_var = self._hidden2log_var(hidden)
        return mu, log_var

    def _decode(self, x):
        return self._decoder(x)

    def _reparametrise(self, mu, log_var):

        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)
        return mu + sigma * z

    def _loss_function(self, mu, log_var, inputs, outputs):

        kl_loss = (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)) \
            .mean(dim=0)
        recon_loss_criterion = torch.nn.MSELoss()
        reconstruction_loss = recon_loss_criterion(inputs, outputs)

        return reconstruction_loss + kl_loss

    def test_decoder(self, plot=False, train_data_exp_path=None, winner_file_name=None):
        code_range = code_in_range(self._decoder.num_inputs, -4., 4., step_size=0.01)
        output = self._decoder(code_range)
        print(output)

        if plot:
            read_and_plot_phenos(train_data_exp_path, test_data=output.detach().numpy(),
                                 winner_file_name=winner_file_name)

    def test(self):
        self(self.training_data, verbosity=True)

    def to_dict(self, train_data_exp_dir_path):

        return {
            'num_epochs': self._num_epochs,
            'batch_size': self._batch_size,
            'final_loss': self._final_loss,
            'train_data_exp_dir_path': train_data_exp_dir_path,
            'encoder': self._encoder.to_dict(),
            'decoder': self._decoder.to_dict()
        }

    def dump_config(self, config_path, train_data_exp_dir_path):

        import json

        with open(config_path + '.json', 'w') as f:
            json.dump(self.to_dict(train_data_exp_dir_path), f, indent=4)

    def dump_decoder(self, file_path):
        self._decoder.save(file_path)
