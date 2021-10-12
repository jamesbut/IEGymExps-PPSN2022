#Some experiments that I might need to run frequently

import sys

def train_test_table(argv):

    print(argv)

    #Either read in the trained models
    if len(argv) == 2:
        train_dirs = argv[1]
        print("Read")
        print("Train dirs:", train_dirs)

    #Or train them
    elif len(argv) == 1:
        print("Train")

if __name__ == "__main__":

    train_test_table(sys.argv)
