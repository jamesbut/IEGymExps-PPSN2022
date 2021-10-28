###############################
# Some mathematical functions #
###############################


def normalise(values, val_max, val_min, norm_max=1., norm_min=0.):
    #Normalise between 0 and 1
    norm_0_1 = (values - val_min) / (val_max - val_min)
    return norm_0_1

    #TODO: normalise between norm_max and norm_min

