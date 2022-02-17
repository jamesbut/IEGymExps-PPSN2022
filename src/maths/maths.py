###############################
# Some mathematical functions #
###############################

import numpy as np


def normalise(values, val_max, val_min, norm_max=1., norm_min=0.):

    # Convert lists to numpy arrays
    if isinstance(values, list):
        values = np.asarray(values)
    if isinstance(val_max, list):
        val_max = np.asarray(val_max)
    if isinstance(val_min, list):
        val_min = np.asarray(val_min)

    # Normalise between 0 and 1
    norm_0_1 = (values - val_min) / (val_max - val_min)

    # TODO: normalise between norm_max and norm_min

    return norm_0_1
