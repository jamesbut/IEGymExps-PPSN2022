import torch
import torch.nn as nn

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

    def __init__(self, code_size, training_data):
        super().__init__()

        self.training_data = training_data

        self.encoder = Encoder(training_data.size(1), code_size)
        self.decoder = Decoder(code_size, training_data.size(1))

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

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

    def train(self, num_epochs, batch_size=0):

        loss = nn.MSELoss()

        for epoch in range(num_epochs):

            self.optimiser.zero_grad()

            outputs = self(self.training_data)

            train_loss = loss(outputs, self.training_data)

            train_loss.backward()

            self.optimiser.step()

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs,
                                                        train_loss))

    def test(self):
        self(self.training_data, verbosity=True)

    def dump_decoder(self):
        torch.save(self.decoder, "generator.pt")
