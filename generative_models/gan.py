import torch
import torch.nn as nn
import torch.nn.functional as F
from generative_models.batch_utils import generate_batches
from generative_models.model_testing import code_in_range
from plotter import read_and_plot


class Generator(nn.Module):

    def __init__(self, code_size, num_outputs, num_hidden_neurons=None):
        super(Generator, self).__init__()

        self._num_hidden_neurons = num_hidden_neurons

        if self._num_hidden_neurons is None:
            self.l1 = nn.Linear(code_size, num_outputs)

        else:
            self.l1 = nn.Linear(code_size, self._num_hidden_neurons)
            self.l2 = nn.Linear(self._num_hidden_neurons, num_outputs)

    def forward(self, x):

        if self._num_hidden_neurons is None:
            return self.l1(x)
        else:
            return self.l2(F.relu(self.l1(x)))


class Discriminator(nn.Module):

    def __init__(self, num_inputs, num_hidden_neurons=None):
        super(Discriminator, self).__init__()

        self.num_hidden_neurons = num_hidden_neurons

        if self.num_hidden_neurons is None:
            self.l1 = nn.Linear(num_inputs, 1)

        else:
            self.l1 = nn.Linear(num_inputs, self.num_hidden_neurons)
            self.l2 = nn.Linear(self.num_hidden_neurons, 1)

    def forward(self, x):

        if self.num_hidden_neurons is None:
            return F.sigmoid(self.l1(x))

        else:
            return F.sigmoid(self.l2(F.relu(self.l1(x))))


class GAN():

    def __init__(self, code_size, training_data_vec_size, num_hidden_layers,
                 neurons_per_hidden_layer, read_generator=False):

        self._generator = Generator(code_size, training_data_vec_size,
                                    neurons_per_hidden_layer)
        self._discriminator = Discriminator(training_data_vec_size,
                                            neurons_per_hidden_layer)

        if read_generator:
            self._generator = torch.load('generator.pt')

        self._g_optimiser = torch.optim.RMSprop(self._generator.parameters(), lr=2e-4)
        self._d_optimiser = torch.optim.RMSprop(self._discriminator.parameters(), lr=5e-4)

        self._code_size = code_size

    def train(self, train_data, num_epochs, batch_size):

        loss = nn.BCELoss()

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

    def test(self, rand_code=False, plot=False, train_data_dir=None):

        if rand_code:
            # Generate fake data using generator
            noise = torch.randn(self._training_data.size(0), self._code_size)
            fake_data = self._generator(noise)

        else:

            code_range = code_in_range(self._code_size, -4., 4., step_size=0.01)
            fake_data = self._generator(code_range)

        print(fake_data)

        if plot:
            read_and_plot(train_data_dir, test_data=fake_data.detach().numpy())

    def dump_decoder(self, file_path):
        torch.save(self._generator, file_path)
