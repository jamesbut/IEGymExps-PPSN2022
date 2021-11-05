##################################################
# A way to run a number of experiments in serial #
##################################################

from main import main
import sys
import constants as consts


if __name__ == '__main__':

    settings = [
        'consts.NUM_GENS = 10',
        'consts.NUM_GENS = 20'
    ]

    for s in settings:
        exec(s)
        main(sys.argv)
