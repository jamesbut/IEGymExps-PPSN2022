import numpy as np
import torch
from plotter import read_and_plot_phenos
from typing import Optional, List


def test_decoder(dump_model_dir, gen_model_type, decoder_file_num, train_data_path,
                 winner_file_name, plot_axis_lb=None, plot_axis_ub=None,
                 colour_params=False, print_numpy_arrays=False,
                 train_data_exp_group: bool = False,
                 data_lb: Optional[float] = None, data_ub: Optional[float] = None,
                 plot_evo_starts: Optional[List[str]] = None):

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
    decoder_output = gen_model.test_decoder()
    if print_numpy_arrays:
        import sys
        np.set_printoptions(threshold=sys.maxsize)
    print(decoder_output)

    ub_filter = np.full((decoder_output.shape[0]), True)
    lb_filter = np.full((decoder_output.shape[0]), True)
    if data_ub is not None:
        # Filter out phenotypes if any elements are greater than data_ub
        ub_filter = ~np.any(decoder_output > data_ub, axis=1)
    if data_lb is not None:
        # Filter out phenotypes if any elements are greater than data_ub
        lb_filter = ~np.any(decoder_output < data_lb, axis=1)

    # Apply filters
    decoder_output = decoder_output[ub_filter & lb_filter]

    # Plot decoder output and training data
    read_and_plot_phenos(train_data_path, test_data=decoder_output,
                         winner_file_name=winner_file_name,
                         plot_axis_lb=plot_axis_lb, plot_axis_ub=plot_axis_ub,
                         colour_params=colour_params, group=train_data_exp_group,
                         plot_evo_starts=plot_evo_starts)


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
