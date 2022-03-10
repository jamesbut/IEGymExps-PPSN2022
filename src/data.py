import os
import sys
from glob import glob
import numpy as np
import json
from helper import remove_dirs_from_path


# Read useful data about winning agents from experiment
def read_agent_data(exp_data_path, winner_file_name):

    # Get directories in data_dir_path
    run_folder_paths = get_sub_folders(exp_data_path)

    # If there are no directories in exp_data_path, then look for data in exp_data_path
    if not run_folder_paths:
        run_folder_paths = [exp_data_path]

    # Append winner file name to folder paths
    for i in range(len(run_folder_paths)):
        run_folder_paths[i] += '/' + winner_file_name + '.json'

    # Read data
    fitnesses, genos, phenos, domain_params = read_data(run_folder_paths)

    return fitnesses, genos, phenos, domain_params, run_folder_paths


# Read data from evolutionary runs
def read_evo_data(exp_data_path):

    # Get sub folders in data_dir (different run folders)
    run_folder_paths = get_sub_folders(exp_data_path)
    if not run_folder_paths:
        run_folder_paths = [exp_data_path]

    exp_mean_fitnesses = []
    exp_best_fitnesses = []
    for rfp in run_folder_paths:
        rfp_means = rfp + '/mean_fitnesses'
        rfp_bests = rfp + '/best_fitnesses'

        # Read data and convert to floats
        mean_fitnesses = list(map(float, read_list(rfp_means)))
        best_fitnesses = list(map(float, read_list(rfp_bests)))

        exp_mean_fitnesses.append(mean_fitnesses)
        exp_best_fitnesses.append(best_fitnesses)

    return np.array(exp_mean_fitnesses), np.array(exp_best_fitnesses)


# Read new data format that uses json files
def read_data(folder_paths):

    fitnesses = []
    genotypes = []
    phenotypes = []
    domain_params = []

    for fp in folder_paths:
        try:
            with open(fp) as agent_file:
                agent = json.load(agent_file)

                fitnesses.append(agent['fitness'])
                genotypes.append(agent['genotype'])
                phenotypes.append(agent['network']['weights'])
                domain_params.append(agent['env']['domain_params'])

        except FileNotFoundError:
            sys.exit("Could not find file named: " + fp)

    return np.array(fitnesses), np.array(genotypes), np.array(phenotypes), \
           np.array(domain_params)


def get_sub_folders(dir_path, recursive=True, append_dir_path=True,
                    num_top_dirs_appended=0, final_sub_dirs_only=False,
                    num_top_dirs_removed=0, sort_by_suffix_num=False):

    # If appended top directories is given, auto turn off append full directory path
    if num_top_dirs_appended:
        append_dir_path = False

    if os.path.isdir(dir_path):
        # Walk directory
        walk = list(os.walk(dir_path))

        # If final subdirectories are needed only
        if final_sub_dirs_only:
            folder_names = [x[0] for x in walk if not x[1]]
        elif recursive and append_dir_path:
            folder_names = [x[0] for x in walk][1:]
        elif not recursive:
            folder_names = walk[0][1]
            if append_dir_path:
                folder_names = list(map(lambda fn: dir_path + '/' + fn, folder_names))

        # Sort results
        folder_names = sorted(folder_names)
        if sort_by_suffix_num:
            folder_names = _sort_by_suffix_num(folder_names)

        # Remove top directories
        folder_names = [remove_dirs_from_path(x, num_top_dirs_removed)
                        for x in folder_names]

        # Append top directories
        if num_top_dirs_appended > 0:
            top_dirs = '/'.join(dir_path.split('/')[-num_top_dirs_appended:])
            folder_names = [top_dirs + '/' + fn for fn in folder_names]

    else:
        raise NotADirectoryError("{} is not a directory".format(dir_path))

    return folder_names


# Sort directories by the number at the end of the file
def _sort_by_suffix_num(directories):

    def parse_suffix(directory):
        return int(directory.split('_')[-1])

    directories.sort(key=parse_suffix)
    return directories


# Dump list into a file and check for directory existence
def dump_list(lst, dir_path, file_name):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + file_name

    with open(file_path, 'w') as f:
        for val in lst:
            f.write(str(val) + '\n')


# Read list from file
def read_list(file_path):

    with open(file_path, 'r') as f:
        lst = f.readlines()

    # Remove carriage returns
    lst = [val.strip() for val in lst]

    return lst


# Read json
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Dump dictionary as json
def dump_json(file_path, json_dict):
    try:
        with open(file_path, 'w') as file:
            json.dump(json_dict, file, indent=4)
    # If directory is not created, do not do anything
    except FileNotFoundError:
        pass


# Calculates the name of the directory to store exp data in
def create_exp_dir_name(config, config_file_name):

    # If the default config is used, search in the data directory for the next
    # exp_ directory number
    if config_file_name.endswith('configs/default.json'):

        exp_full_dirs = glob(config['logging']['data_dir_path'] + "/*")

        max_exp_num = 0

        if len(exp_full_dirs) == 0:
            max_exp_num = 0

        else:
            for ed in exp_full_dirs:

                # Get exp numbers in directory
                split_path = ed.split("/")
                exp_string = split_path[-1]
                try:
                    exp_num = int(exp_string.split("_")[-1])
                except ValueError:
                    continue

                # Find largest exp number
                if max_exp_num is None:
                    max_exp_num = exp_num
                else:
                    if exp_num > max_exp_num:
                        max_exp_num = exp_num

        exp_dir_name = 'exp_' + str(max_exp_num + 1)

    # If the default config is not used, set the experiment directory name as a
    # modified config file name
    else:

        # Remove */configs/
        config_file_name = config_file_name.split('configs/')[1]
        # Remove .json
        exp_dir_name = config_file_name.removesuffix('.json')

    return exp_dir_name


def read_configs(argv):

    working_dir_path = os.getcwd()

    # Read in group of config files
    if argv is not None and '-configs' in argv:

        # Get config group directory from command line
        config_index = argv.index('-configs')
        config_dir = argv[config_index + 1]

        # Recursively get all config files in directory
        config_files = []
        for config_walk in os.walk(working_dir_path + '/configs/' + config_dir):
            if config_walk[2]:
                config_files += [config_walk[0] + '/' + config_file_name
                                 for config_file_name in config_walk[2]]

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


def create_synthetic_data(code_size, num_data_points=500):

    # return -10. + np.randn(500, code_size)

    means = np.zeros(500, 2)
    for i in range(means.size(0)):
        for j in range(means.size(1)):
            if j == 0:
                means[i][j] = 10.
            else:
                means[i][j] = -10.

    return means + np.randn(500, 2)


# Function to clean data according to some filter
# (according to numerical bounds for now)
def clean_data(exp_data_dir, config, l_gb, u_gb):

    # Create experiment directory path
    exp_data_path = config['logging']['data_dir_path'] + exp_data_dir
    print('exp_data_path:', exp_data_path)

    # Read genotypes
    _, genos, _, _, folder_paths = \
        read_agent_data(exp_data_path, config['logging']['winner_file_name'])
    print(genos)

    # Calculate genotypes that are out of bounds
    print('\nOut of bounds genos:')

    num_out_of_bounds = 0
    out_of_bound_geno_paths = []
    for i, geno in enumerate(genos):
        if any((g > u_gb or g < l_gb) for g in geno):
            num_out_of_bounds += 1
            out_of_bound_geno_paths.append(folder_paths[i])
            print(geno)

    print('Num genos out of bounds: {}/{}'.format(num_out_of_bounds, len(genos)))

    # Get directories to delete and print them
    print('Files to be deleted:')
    out_of_bound_geno_paths = ['/'.join(p.split('/')[:-1])
                               for p in out_of_bound_geno_paths]
    for p in out_of_bound_geno_paths:
        print(p)

    # Ask for files to be deleted
    delete_q = input('Do you want to delete these dirs (y/n)? ')
    if delete_q != 'y':
        return

    # Delete directories of out of bounds genotypes
    import shutil
    for p in out_of_bound_geno_paths:
        shutil.rmtree(p)

    print('Files successfully deleted')


if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    np.set_printoptions(threshold=sys.maxsize)

    config = read_configs(None)[0]

    L_GB = -100.
    U_GB = 100.

    clean_data(sys.argv[1], config, L_GB, U_GB)
