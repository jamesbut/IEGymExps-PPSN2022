####################################
# Can generate a number of configs #
####################################

import json
from data import read_configs, get_sub_folders, dump_json
from helper import modify_dict, print_dict
import itertools
import copy
import os


# Generate new config file using a base config and setting modifications
def generate_new_config(base_config, settings):
    new_config = copy.deepcopy(base_config)
    for setting in settings:
        new_config = modify_dict(setting[0], setting[1], new_config)
    return new_config


# Create list of setting tuples from list of hyperparameter values
def create_settings(hyper_params):

    # Separate keys and values
    hyper_param_keys = [hp[0] for hp in hyper_params]
    hyper_param_vals = [hp[1] for hp in hyper_params]
    # Find cross product of values
    prod = itertools.product(*hyper_param_vals)

    # Build settings from each of the results of the cross product
    settings = []
    for p in prod:
        setting = []
        for kv in zip(hyper_param_keys, p):
            setting.append(kv)
        settings.append(setting)

    return settings


# Dump configs into directory
def dump_configs(configs):

    # Check what group number we are up to and calculate next group number
    prev_group_dirs = get_sub_folders('configs', recursive=False, append_dir_path=False)
    prev_group_nums = [gd.replace('g', '') for gd in prev_group_dirs]
    new_group_num = int(prev_group_nums[-1]) + 1
    new_group_name = 'g' + str(new_group_num)
    new_group_path = 'configs/' + new_group_name

    # Create group directory
    os.mkdir(new_group_path)

    # Dump configs
    for i, config in enumerate(configs):
        config_file_path = new_group_path + '/' + new_group_name + '_exp_' \
                           + str(i) + '.json'
        dump_json(config_file_path, config)


def main():
    base_config = read_configs(None)[0]

    hyper_params = [
        (["env", "domain_params"], [[0.0008, 0.0012, 0.0016], [0.0008],
                                    [0.0012], [0.0016]]),
        (["controller", "neurons_per_hidden_layer"], [8, 16, 32, 64]),
        (["env", "domain_params_input"], [True]),
    ]

    # Generate settings and then configs from those settings
    settings = create_settings(hyper_params)
    configs = [generate_new_config(base_config, s) for s in settings]

    # Print configs
    for c in configs:
        print_dict(c)

    dump_configs(configs)


if __name__ == '__main__':
    main()
