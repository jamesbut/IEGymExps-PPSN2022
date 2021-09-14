import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, num_inputs, code_size):
        super(Encoder, self).__init__()

        self.l1 = nn.Linear(num_inputs, code_size)

    def forward(self, x):
        return self.l1(F.relu(x))

class Decoder(nn.Module):

    def __init__(self, code_size, num_outputs):
        super(Decoder, self).__init__()

        self.l1 = nn.Linear(code_size, num_outputs)

    def forward(self, x):
        return self.l1(x)


class VAE(nn.Module):

    def __init__(self, code_size, training_data):
        super().__init__()

        self.training_data = training_data

        self.encoder = Encoder(training_data.size(1), code_size)
        self.hidden2mu = nn.Linear(code_size, code_size)
        self.hidden2log_var = nn.Linear(code_size, code_size)
        self.decoder = Decoder(code_size, training_data.size(1))

        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, inputs, verbosity=False):

        mu, log_var = self.encode(inputs)
        hidden = self.reparametrise(mu, log_var)
        output = self.decoder(hidden)

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

    def encode(self, x):

        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):

        return self.decoder(x)

    def reparametrise(self, mu, log_var):

        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size = (mu.size(0), mu.size(1)))
        z = z.type_as(mu)
        return mu + sigma * z

    def loss_function(self, mu, log_var, inputs, outputs):

        kl_loss = (-0.5*(1+log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        reconstruction_loss = recon_loss_criterion(inputs, outputs)

        return reconstruction_loss + kl_loss

    def train(self, num_epochs, batch_size=0):

        for epoch in range(num_epochs):

            self.optimiser.zero_grad()

            mu, log_var = self.encode(self.training_data)
            hidden = self.reparametrise(mu, log_var)
            outputs = self.decode(hidden)

            train_loss = self.loss_function(mu, log_var,
                                            self.training_data, outputs)
            train_loss.backward()
            self.optimiser.step()

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs,
                                                        train_loss))

    def test(self):
        self(self.training_data, verbosity=True)

    def dump_decoder(self):
        torch.save(self.decoder, "generator.pt")
