import torch
import torch.nn as nn
from generative_models.batch_utils import generate_batches

class Generator(nn.Module):

    def __init__(self, num_inputs, num_outputs):

        super(Generator, self).__init__()

        self.l1 = nn.Linear(num_inputs, num_outputs)


    def forward(self, x):
        return self.l1(x)


class Discriminator(nn.Module):

    def __init__(self, num_inputs):

        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(num_inputs, 1)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        return self.activation(self.l1(x))



class GAN():

    def __init__(self, code_size, training_data):

        self.generator = Generator(code_size, training_data.size(1))
        self.discriminator = Discriminator(training_data.size(1))

        self.g_optimiser = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.d_optimiser = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.training_data = training_data

        self.code_size = code_size

    def train(self, training_steps, batch_size=256):

        loss = nn.BCELoss()

        #Generate batches
        #batches = generate_batches(self.training_data, batch_size)

        for i in range(training_steps):

            self.discriminator.zero_grad()

            #Train discriminator on real data
            d_real_output = self.discriminator(self.training_data)

            true_labels = torch.Tensor([[1.] * self.training_data.size(0)]).T
            d_real_loss = loss(d_real_output, true_labels)
            d_real_loss.backward()

            #Generate fake data using generator
            noise = torch.randn(self.training_data.size(0), self.code_size)
            fake_data = self.generator(noise)

            #Train discriminator on false data
            d_fake_output = self.discriminator(fake_data.detach())

            false_labels = torch.Tensor([[0.] * self.training_data.size(0)]).T
            d_fake_loss = loss(d_fake_output, false_labels)
            d_fake_loss.backward()

            self.d_optimiser.step()

            #Train generator
            self.generator.zero_grad()
            d_output = self.discriminator(fake_data)
            g_loss = loss(d_output, true_labels)
            g_loss.backward()

            self.g_optimiser.step()


            #Print loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            print("Step: {}    D loss: {}       G loss: {}".format(i, d_loss, g_loss))


    def test(self):

        #Generate fake data using generator
        noise = torch.randn(self.training_data.size(0), self.code_size)
        fake_data = self.generator(noise)

        print("Testing...\nFake data:")
        print(fake_data)

    def dump_generator(self):
        torch.save(self.generator, "generator.pt")
