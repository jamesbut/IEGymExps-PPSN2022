import os
from typing import List, Tuple, Optional
from data import create_exp_dir_name, read_json


# Return flag arguments, where --flag *args*
# Returns all arguments after flag or None if flag is not in args
def retrieve_flag_args(flag: str, argv: List[str]) -> Optional[List[str]]:

    if flag not in argv:
        return None

    flag_pos: int = argv.index(flag)
    return argv[flag_pos + 1:]


# Read plot axis limits from command line
def parse_axis_limits(argv: List[str]) -> Tuple[Optional[float], Optional[float]]:

    if '--fixed-axis' in argv:
        arg_pos = argv.index('--fixed-axis')

        try:
            plot_axis_lb = float(argv[arg_pos + 1])
            plot_axis_ub = float(argv[arg_pos + 2])
            return plot_axis_lb, plot_axis_ub
        except IndexError:
            return None, None

    return None, None


# Parse test decoder from command line
def parse_test_decoder(argv) -> Tuple[str, int]:

    try:
        decoder_type = argv[2]
        decoder_num = int(argv[3])
    except IndexError:
        print('argv:', argv)
        print('Example call: python main.py -test_decoder *decoder_type* '
              '*decoder_num*')
        raise

    return decoder_type, decoder_num


def read_configs(argv):

    working_dir_path = os.getcwd()

    # Read in group of config files
    if argv is not None and '--configs' in argv:

        # Get config group directory from command line
        config_index = argv.index('--configs')
        config_dir = argv[config_index + 1]

        # Recursively get all config files in directory
        config_files = []
        for config_walk in os.walk(working_dir_path + '/configs/' + config_dir):
            if config_walk[2]:
                config_files += [config_walk[0] + '/' + config_file_name
                                 for config_file_name in config_walk[2]]

    # Read in single config file
    elif argv is not None and '--config' in argv:

        config_dir = argv[argv.index('--config') + 1]
        config_files = [working_dir_path + '/configs/' + config_dir]

    else:
        # Use default config file
        config_files = [working_dir_path + '/configs/default.json']

    # Read config files
    configs = list(map(read_json, config_files))

    # Add experiment directory path to configs
    for conf in zip(configs, config_files):
        conf[0]['logging']['exp_dir_name'] = create_exp_dir_name(conf[0], conf[1])
        conf[0]['logging']['config_file_path'] = conf[1]

    return configs
