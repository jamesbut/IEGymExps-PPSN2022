##################################################
# A way to run a number of experiments in serial #
##################################################

from main import main
import sys
import constants as consts


if __name__ == '__main__':

    consts.LAMBDA = 10
    consts.NUM_GENS = 10
    consts.NUM_EVO_RUNS = 1

    main(sys.argv)

    consts.NUM_GENS = 20

    main(sys.argv)
