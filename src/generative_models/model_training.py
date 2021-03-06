from generative_models.autoencoder import Autoencoder
from generative_models.vae import VAE
from generative_models.gan import GAN
from data import read_agent_data, get_sub_folders
from torch import Tensor
import numpy as np


def train_generative_model(gen_model_type, code_size, num_hidden_layers,
                           neurons_per_hidden_layer, num_epochs, batch_size,
                           train_data_path, exp_group, dump_model_dir,
                           winner_file_name, optimiser_json):

    train_data_exp_dirs = [train_data_path]

    # If the training data of a group of experiments is being parsed
    if exp_group:
        train_data_exp_dirs = get_sub_folders(train_data_path, recursive=False)

    # Read training data from experiment directiories
    train_data = None
    for exp_dir in train_data_exp_dirs:
        _, _, phenotypes, _, _ = read_agent_data(exp_dir, winner_file_name)
        if train_data is None:
            train_data = phenotypes
        else:
            np.concatenate((train_data, phenotypes))

    # Convert numpy array to tensor
    train_data = Tensor(phenotypes)

    # Build model
    gen_model = build_generative_model(gen_model_type, code_size, train_data.size(1),
                                       num_hidden_layers, neurons_per_hidden_layer,
                                       optimiser_json)

    # Train model
    gen_model.train(train_data, num_epochs, batch_size)

    # Dump model
    model_path = get_model_path(gen_model_type, dump_model_dir)
    gen_model.dump_decoder(model_path + '.pt')
    gen_model.dump_config(model_path, train_data_exp_dirs)

    # Test model
    # gen_model.test(plot=True, train_data_path=train_data_exp_path)


def build_generative_model(gen_model_type, code_size, data_size,
                           num_hidden_layers, neurons_per_hidden_layer,
                           optimiser_json):

    if gen_model_type == 'ae':
        return Autoencoder(code_size, data_size,
                           num_hidden_layers, neurons_per_hidden_layer,
                           optimiser_json['lr'])
    elif gen_model_type == 'vae':
        return VAE(code_size, data_size,
                   num_hidden_layers, neurons_per_hidden_layer,
                   optimiser_json['lr'])
    elif gen_model_type == 'gan':
        return GAN(code_size, data_size,
                   num_hidden_layers, neurons_per_hidden_layer,
                   optimiser_json['g_lr'], optimiser_json['d_lr'])
    else:
        raise ValueError('{} is not a valid generative model type'
                         .format(gen_model_type))


# Get model path to dump to
def get_model_path(gen_model_type, dump_model_dir):

    if gen_model_type == 'ae':
        model_file_name = '/ae_' + str(get_model_num('ae', dump_model_dir))
    elif gen_model_type == 'vae':
        model_file_name = '/vae_' + str(get_model_num('vae', dump_model_dir))
    elif gen_model_type == 'gan':
        model_file_name = '/gan_' + str(get_model_num('gan', dump_model_dir))
    else:
        raise ValueError('{} is not a valid generative model type'
                         .format(gen_model_type))

    return dump_model_dir + '/' + model_file_name


# Increments model number so that other models are not overwritten
def get_model_num(gen_model_type, dump_model_dir):

    import glob

    # Read previous model files
    prev_model_files = glob.glob(dump_model_dir + '/' + gen_model_type + '_*.pt')

    # If no models are present, set model num to 0
    if not prev_model_files:
        return 0

    # Parse model numbers
    prev_model_nums = [int(mf.split('/')[-1].split('_')[-1].split('.')[0])
                       for mf in prev_model_files]

    # Find max model num and increment
    return max(prev_model_nums) + 1
