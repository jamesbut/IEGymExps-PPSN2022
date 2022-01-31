import numpy as np
import torch


def test_decoder(dump_model_dir, gen_model_type, decoder_file_num, train_data_path,
                 winner_file_name, train_g_lb=None, train_g_ub=None):

    # Read decoder
    decoder_file_path = dump_model_dir + '/' + gen_model_type + '_' \
                        + str(decoder_file_num) + '.pt'

    try:
        gen_model = build_decoder(gen_model_type, decoder_file_path)
    except IOError:
        print("Could not find requested decoder for testing:", decoder_file_path)
        return

    print('Testing decoder:', decoder_file_path)

    # Test decoder
    gen_model.test_decoder(plot=True, train_data_exp_path=train_data_path,
                           winner_file_name=winner_file_name,
                           train_g_lb=train_g_lb, train_g_ub=train_g_ub)


def build_decoder(gen_model_type, decoder_file_path):

    if gen_model_type == 'ae':
        from generative_models.autoencoder import Autoencoder
        return Autoencoder(decoder_file_path=decoder_file_path)
    elif gen_model_type == 'vae':
        from generative_models.vae import VAE
        return VAE(decoder_file_path=decoder_file_path)
    if gen_model_type == 'gan':
        from generative_models.gan import GAN
        return GAN(decoder_file_path=decoder_file_path)


def code_in_range(code_size, lb, ub, step_size=0.001):

    if code_size == 1:
        code = np.arange(lb, ub, step_size)
        code = code.reshape(len(code), code_size)

    elif code_size == 2:
        code = []
        code_range = np.arange(lb, ub, step_size)
        for i in range(len(code_range)):
            for j in range(len(code_range)):
                code.append([code_range[i], code_range[j]])
        code = np.array(code)

    return torch.from_numpy(code).float()
