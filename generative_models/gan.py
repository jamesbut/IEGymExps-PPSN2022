import torch
from generative_models.batch_utils import generate_batches
from generative_models.model_testing import code_in_range
from plotter import read_and_plot_phenos
from neural_network import NeuralNetwork


class GAN():

    def __init__(self, code_size, train_data_vec_size, num_hidden_layers,
                 neurons_per_hidden_layer, g_lr=2e-4, d_lr=5e-4, read_decoder=False):

        self._generator = NeuralNetwork(
            num_inputs=code_size,
            num_outputs=train_data_vec_size,
            num_hidden_layers=num_hidden_layers,
            neurons_per_hidden_layer=neurons_per_hidden_layer,
            final_activ_func=None)

        self._discriminator = NeuralNetwork(
            num_inputs=train_data_vec_size,
            num_outputs=1,
            num_hidden_layers=num_hidden_layers,
            neurons_per_hidden_layer=neurons_per_hidden_layer,
            final_activ_func='sigmoid')

        if read_decoder:
            self._generator = NeuralNetwork(file_path='decoder.pt')

        self._g_optimiser = torch.optim.RMSprop(self._generator.parameters(),
                                                lr=g_lr)
        self._d_optimiser = torch.optim.RMSprop(self._discriminator.parameters(),
                                                lr=d_lr)

        self._code_size = code_size

    def train(self, train_data, num_epochs, batch_size):

        self._num_epochs = num_epochs
        self._batch_size = batch_size

        loss = torch.nn.BCELoss()

        # Generate batches
        batches = generate_batches(train_data, batch_size, shuffle=True)

        for i in range(num_epochs):
            d_total_loss = 0
            g_total_loss = 0
            for batch in batches:

                self._discriminator.zero_grad()

                # Train discriminator on real data
                d_real_output = self._discriminator(batch)

                true_labels = torch.Tensor([[1.] * batch.size(0)]).T
                d_real_loss = loss(d_real_output, true_labels)
                d_real_loss.backward()

                # Generate fake data using generator
                noise = torch.randn(batch.size(0), self._code_size)
                fake_data = self._generator(noise)

                # Train discriminator on false data
                d_fake_output = self._discriminator(fake_data.detach())

                false_labels = torch.Tensor([[0.] * batch.size(0)]).T
                d_fake_loss = loss(d_fake_output, false_labels)
                d_fake_loss.backward()

                self._d_optimiser.step()

                # Train generator
                self._generator.zero_grad()
                d_output = self._discriminator(fake_data)
                g_loss = loss(d_output, true_labels)
                g_loss.backward()

                self._g_optimiser.step()

                d_total_loss += d_real_loss + d_fake_loss
                g_total_loss += g_loss

            # Print loss
            d_avg_loss = d_total_loss / (2 * len(batches))
            g_avg_loss = g_total_loss / len(batches)
            print("Step: {}    D loss: {}       G loss: {}".format(i,
                                                                   d_avg_loss,
                                                                   g_avg_loss))

        self._final_d_avg_loss = d_avg_loss.item()
        self._final_g_avg_loss = g_avg_loss.item()

    def test(self, rand_code=False, plot=False,
             train_data_exp_path=None, winner_file_name=None):

        # Generate fake data using generator
        if rand_code:
            noise = torch.randn(self._training_data.size(0), self._code_size)
        else:
            noise = code_in_range(self._code_size, -4., 4., step_size=0.01)
        fake_data = self._generator(noise)

        print('Testing, fake data:')
        print(fake_data)

        if plot:
            read_and_plot_phenos(train_data_exp_path, winner_file_name,
                                 fake_data.detach().numpy())

    def to_dict(self, train_data_exp_dir_path):

        return {
            'num_epochs': self._num_epochs,
            'batch_size': self._batch_size,
            'final_g_loss': self._final_g_avg_loss,
            'final_d_loss': self._final_d_avg_loss,
            'train_data_exp_dir_path': train_data_exp_dir_path,
            'generator': self._generator.to_dict(),
            'discriminator': self._discriminator.to_dict()
        }

    def dump_config(self, config_path, train_data_exp_dir_path):

        import json

        with open(config_path + '.json', 'w') as f:
            json.dump(self.to_dict(train_data_exp_dir_path), f, indent=4)

    def dump_decoder(self, file_path):
        self._generator.save(file_path)
