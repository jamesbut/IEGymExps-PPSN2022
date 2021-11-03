from generative_models.autoencoder import Autoencoder
from generative_models.vae import VAE
from generative_models.gan import GAN
from data import read_data
from torch import Tensor


def train_generative_model(gen_model_type, code_size, num_hidden_layers,
                           neurons_per_hidden_layer, num_epochs, batch_size,
                           train_data_path, dump_file_path, data_dir_path,
                           winner_file_name):

    # Read training data
    _, _, phenotypes, _, _ = read_data(train_data_path, data_dir_path, winner_file_name)
    train_data = Tensor(phenotypes)

    # Build model
    gen_model = _build_generative_model(gen_model_type, code_size, train_data.size(1),
                                        num_hidden_layers, neurons_per_hidden_layer)

    # Train model
    gen_model.train(train_data, num_epochs, batch_size)
    gen_model.dump_decoder(dump_file_path)

    # Test model
    gen_model.test(plot=True, train_data_dir=train_data_path,
                   data_dir_path=data_dir_path, winner_file_name=winner_file_name)


def _build_generative_model(gen_model_type, code_size, data_size,
                            num_hidden_layers, neurons_per_hidden_layer):

    if gen_model_type == 'ae':
        return Autoencoder(code_size, data_size,
                           num_hidden_layers, neurons_per_hidden_layer)
    elif gen_model_type == 'vae':
        return VAE(code_size, data_size,
                   num_hidden_layers, neurons_per_hidden_layer)
    elif gen_model_type == 'gan':
        return GAN(code_size, data_size,
                   num_hidden_layers, neurons_per_hidden_layer)
    else:
        raise ValueError('{} is not a valid generative model type'
                         .format(gen_model_type))
