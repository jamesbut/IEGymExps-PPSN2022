####################################
# Can generate a number of configs #
####################################

from data import read_configs, get_sub_folders, dump_json
from helper import modify_dict, print_dict
import itertools
import copy
import os
import re


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

    # Create group directory
    os.mkdir(new_group_path)

    # Dump configs
    for i, config in enumerate(configs):
        config_file_path = new_group_path + '/' + new_group_name + '_exp_' \
                           + str(i) + '.json'
        dump_json(config_file_path, config)


def main():
    base_config = read_configs(None)[0]

    # Get centroid directories for universal controllers
    centroid_dirs = get_sub_folders('data/g6', final_sub_dirs_only=True,
                                   num_top_dirs_removed=1)

    hyper_params = [
        (["env", "domain_params"], [[0.0008], [0.0010], [0.0012], [0.0014], [0.0016]]),
        # (["env" "domain_params"], [[0.0008, 0.0012, 0.0016]])
        # (["optimiser", "cmaes", "centroid"], centroid_dirs)
        (["ie", "decoder_file_num"], [0, 1, 2, 3, 4])
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
