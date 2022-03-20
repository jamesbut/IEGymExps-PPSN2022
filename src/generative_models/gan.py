import torch
from generative_models.batch_utils import generate_batches
from generative_models.model_testing import code_in_range
from neural_network import NeuralNetwork


class GAN():

    def __init__(self, code_size=None, train_data_vec_size=None, num_hidden_layers=None,
                 neurons_per_hidden_layer=None, g_lr=2e-4, d_lr=5e-4,
                 decoder_file_path=None):

        if decoder_file_path:
            self._generator = NeuralNetwork(file_path=decoder_file_path)
        else:
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

            self._g_optimiser = torch.optim.RMSprop(self._generator.parameters(),
                                                    lr=g_lr)
            self._d_optimiser = torch.optim.RMSprop(self._discriminator.parameters(),
                                                    lr=d_lr)
            self._g_lr = g_lr
            self._d_lr = d_lr

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
                noise = torch.randn(batch.size(0), self._generator.num_inputs)
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

    def test_decoder(self):

        code_range = code_in_range(self._generator.num_inputs, -3., 3., step_size=0.05)
        output = self._generator(code_range)
        return output.detach().numpy()

    def to_dict(self, train_data_exp_dir_path):

        return {
            'num_epochs': self._num_epochs,
            'batch_size': self._batch_size,
            'final_g_loss': self._final_g_avg_loss,
            'final_d_loss': self._final_d_avg_loss,
            'generator_learning_rate': self._g_lr,
            'discriminator_learning_rate': self._d_lr,
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
