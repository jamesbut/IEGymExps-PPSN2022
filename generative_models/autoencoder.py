import torch
import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self, code_size, training_data):
        super().__init__()

        self.training_data = training_data

        self.encoder_ol = nn.Linear(training_data.size(1), code_size)
        self.decoder_ol = nn.Linear(code_size, training_data.size(1))

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, inputs, verbosity=False):

        output = self.encoder_ol(inputs)
        code = torch.sigmoid(output)
        output = self.decoder_ol(code)

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
        #Figure out how to dump just the decoder
        pass
