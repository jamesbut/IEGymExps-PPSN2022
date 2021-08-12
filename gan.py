import torch
import torch.nn as nn

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

    def __init__(self, code_size, input_size, training_data):

        self.generator = Generator(code_size, input_size)
        self.discriminator = Discriminator(input_size)

        self.g_optimiser = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.d_optimiser = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.training_data = training_data

        self.code_size = code_size

    def train(self, training_steps, batch_size=0):

        loss = nn.BCELoss()

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


def create_synthetic_data(code_size, num_data_points=500):

    #return -10. + torch.randn(500, code_size)

    means = torch.zeros(500, 2)
    for i in range(means.size(0)):
        for j in range(means.size(1)):
            if j == 0:
                means[i][j] = 10.
            else:
                means[i][j] = -10.

    return means + torch.randn(500, 2)
