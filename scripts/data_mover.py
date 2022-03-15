# Small script to move data around

import sys
from glob import glob
import random
import os
import shutil


def main(args):

    # Take from and to directory from command line
    from_dir = args[1]
    to_dir = args[2]

    print('From dir:', from_dir)
    print('To dir:', to_dir)

    # Read all directories in from_dir
    if not os.path.isdir(from_dir):
        raise IOError('{} does not exist'.format(from_dir))
    from_data_dirs = glob(from_dir + '/*/')

    # Sample n random directories in the from_dir
    NUM_REQUIRED_DIRS = 333
    NUM_DIRS_TO_REMOVE = len(from_data_dirs) - NUM_REQUIRED_DIRS
    selected_dirs = random.sample(from_data_dirs, NUM_DIRS_TO_REMOVE)
    for sd in selected_dirs:
        print(sd)

    # Prompt to move directories
    move_y_n = input('Move the {} above directories (y/n) '.format(len(selected_dirs)))

    # Move selected directories into to_dir
    if move_y_n == 'y':
        for sd in selected_dirs:
            shutil.move(sd, to_dir)


if __name__ == '__main__':
    main(sys.argv)
