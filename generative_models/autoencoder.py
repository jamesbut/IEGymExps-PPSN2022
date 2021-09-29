import torch
import torch.nn as nn
from generative_models.batch_utils import generate_batches
from generative_models.model_testing import code_in_range
from scripts.plotter import read_and_plot

class Encoder(nn.Module):

    def __init__(self, num_inputs, code_size):
        super(Encoder, self).__init__()

        self.l1 = nn.Linear(num_inputs, code_size)

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):

    def __init__(self, code_size, num_outputs):
        super(Decoder, self).__init__()

        self.l1 = nn.Linear(code_size, num_outputs)

    def forward(self, x):
        return self.l1(x)


class Autoencoder(nn.Module):

    def __init__(self, code_size, training_data, read_decoder=False):
        super().__init__()

        self.training_data = training_data

        self.encoder = Encoder(training_data.size(1), code_size)
        self.decoder = Decoder(code_size, training_data.size(1))

        if read_decoder:
            self.decoder = torch.load('generator.pt')

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.code_size = code_size

    def forward(self, inputs, verbosity=False):

        output = self.encoder(inputs)
        code = torch.sigmoid(output)
        output = self.decoder(code)

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

    def train(self, num_epochs, batch_size):

        loss_fn = nn.MSELoss()

        batches = generate_batches(self.training_data, batch_size, shuffle=True)

        for epoch in range(num_epochs):
            loss = 0
            for batch in batches:

                self.optimiser.zero_grad()

                outputs = self(batch)

                train_loss = loss_fn(outputs, batch)

                train_loss.backward()

                self.optimiser.step()

                loss += train_loss.item()

            loss = loss / len(batches)

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))

    def test_decoder(self, plot=False, train_data_dir=None):
        test_code = code_in_range(self.code_size, 0., 1., step_size=0.002)
        output = self.decoder(test_code)
        print(output)

        if plot:
            read_and_plot(train_data_dir, test_data=output.detach().numpy(),
                          parent_dir=True)

    def test(self):
        self(self.training_data, verbosity=True)

    def dump_decoder(self):
        torch.save(self.decoder, "generator.pt")
