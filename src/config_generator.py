####################################
# Can generate a number of configs #
####################################

from data import read_configs, get_sub_folders, dump_json
from helper import modify_dict, print_dict
import itertools
import copy
import os
import re
from boltons import iterutils
from typing import Tuple


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


def _calculate_new_group_path(configs) -> Tuple[str, str]:

    # Get all directories in configs dir
    config_dirs = get_sub_folders('configs', recursive=False, append_dir_path=False)
    # Filter to just get group directories
    prev_group_dirs = [cd for cd in config_dirs if re.search("^g[0-9]+", cd) is not None]

    # Check what group number we are up to and calculate next group number
    prev_group_nums = list(map(int, [gd.replace('g', '') for gd in prev_group_dirs]))
    new_group_num = 1
    if prev_group_nums:
        new_group_num = max(prev_group_nums) + 1

    new_group_name = 'g' + str(new_group_num)
    new_group_path = 'configs/' + new_group_name

    return new_group_path, new_group_name


# Dump configs into directory
def dump_configs(configs):

    new_group_path, new_group_name = _calculate_new_group_path(configs)

    # Create group directory
    os.mkdir(new_group_path)

    # Dump configs
    for i, config in enumerate(configs):
        config_file_path = new_group_path + '/'

        # If sub groups are used
        if isinstance(configs[0], list):

            # Create subgroup directories
            sub_group_dir_path = new_group_path + '/sg' + str(i + 1)
            os.mkdir(sub_group_dir_path)

            # Build config names
            for j, c in enumerate(config):
                config_file_name = new_group_name + '_exp_' + \
                    str((i * len(config)) + j) + '.json'
                config_file_path = sub_group_dir_path + '/' + config_file_name

                dump_json(config_file_path, c)

        # Otherwise just dump configs in group directory
        else:

            config_file_path = new_group_path + '/' + new_group_name + '_exp_' \
                           + str(i) + '.json'
            dump_json(config_file_path, config)


def main():
    base_config = read_configs(None)[0]

    # Get centroid directories for universal controllers
    # centroid_dirs = get_sub_folders('data/g3', final_sub_dirs_only=True,
    #                               num_top_dirs_removed=1)

    hyper_params = [
        #(["env", "domain_params"], [[2.0], [3.0], [4.0], [5.0], [6.0]]),
        (["env", "domain_params"], [[2.0, 4.0, 6.0]]),
        # (["optimiser", "cmaes", "centroid"], centroid_dirs)
        # (["ie", "decoder_file_num"], [0, 1, 2, 3, 4])
    ]

    if len(hyper_params) > 2:
        raise ValueError('Sub group code does not account for hyper_params length > 2')

    # Generate settings and then configs from those settings
    settings = create_settings(hyper_params)
    configs = [generate_new_config(base_config, s) for s in settings]

    # Sort configs into subgroups
    if len(hyper_params) == 2:
        configs = iterutils.chunked(configs, len(hyper_params[1][1]))

    # Print configs
    for c in configs:
        print_dict(c)

    dump_configs(configs)


if __name__ == '__main__':
    main()
