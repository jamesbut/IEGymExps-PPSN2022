import torch
import torch.nn as nn
import torch.nn.functional as F
from generative_models.batch_utils import generate_batches
from generative_models.model_testing import code_in_range
from scripts.plotter import read_and_plot

class Generator(nn.Module):

    def __init__(self, code_size, num_outputs, num_hidden_neurons=None):
        super(Generator, self).__init__()

        self.num_hidden_neurons = num_hidden_neurons

        if self.num_hidden_neurons is None:
            self.l1 = nn.Linear(code_size, num_outputs)

        else:
            self.l1 = nn.Linear(code_size, self.num_hidden_neurons)
            self.l2 = nn.Linear(self.num_hidden_neurons, num_outputs)


    def forward(self, x):

        if self.num_hidden_neurons is None:
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

    def __init__(self, code_size, training_data, read_generator=False,
                 num_hidden_neurons=None):

        self.generator = Generator(code_size, training_data.size(1), num_hidden_neurons)
        self.discriminator = Discriminator(training_data.size(1), num_hidden_neurons)

        if read_generator:
            self.generator = torch.load('generator.pt')

        self.g_optimiser = torch.optim.RMSprop(self.generator.parameters(), lr=2e-4)
        self.d_optimiser = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-4)

        self.training_data = training_data

        self.code_size = code_size

    def train(self, training_steps, batch_size):

        loss = nn.BCELoss()

        #Generate batches
        batches = generate_batches(self.training_data, batch_size, shuffle=True)

        for i in range(training_steps):
            d_total_loss = 0
            g_total_loss = 0
            for batch in batches:

                self.discriminator.zero_grad()

                #Train discriminator on real data
                d_real_output = self.discriminator(batch)

                true_labels = torch.Tensor([[1.] * batch.size(0)]).T
                d_real_loss = loss(d_real_output, true_labels)
                d_real_loss.backward()

                #Generate fake data using generator
                noise = torch.randn(batch.size(0), self.code_size)
                fake_data = self.generator(noise)

                #Train discriminator on false data
                d_fake_output = self.discriminator(fake_data.detach())

                false_labels = torch.Tensor([[0.] * batch.size(0)]).T
                d_fake_loss = loss(d_fake_output, false_labels)
                d_fake_loss.backward()

                self.d_optimiser.step()

                #Train generator
                self.generator.zero_grad()
                d_output = self.discriminator(fake_data)
                g_loss = loss(d_output, true_labels)
                g_loss.backward()

                self.g_optimiser.step()

                d_total_loss += d_real_loss + d_fake_loss
                g_total_loss += g_loss


            #Print loss
            d_avg_loss = d_total_loss / (2 * len(batches))
            g_avg_loss = g_total_loss / len(batches)
            print("Step: {}    D loss: {}       G loss: {}".format(i,
                                                                   d_avg_loss,
                                                                   g_avg_loss))

    def test(self, rand_code=False, plot=False, train_data_dir=None):

        if rand_code:
            #Generate fake data using generator
            noise = torch.randn(self.training_data.size(0), self.code_size)
            fake_data = self.generator(noise)

        else:

            code_range = code_in_range(self.code_size, -4., 4., step_size=0.01)
            fake_data = self.generator(code_range)

        print(fake_data)

        if plot:
            read_and_plot(train_data_dir, test_genotypes=fake_data.detach().numpy())

    def dump_generator(self):
        torch.save(self.generator, "generator.pt")
