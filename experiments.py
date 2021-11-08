##################################################
# A way to run a number of experiments in serial #
##################################################

from main import main
import sys
import constants as consts


if __name__ == '__main__':

    settings = [
        'consts.INIT_SIGMA = 1.0;
         consts.DOMAIN_PARAMS_INPUT = False',
        'consts.INIT_SIGMA = 0.5',
        'consts.INIT_SIGMA = 0.1'
    ]

    for s in settings:
        exec(s)
        main(sys.argv)
