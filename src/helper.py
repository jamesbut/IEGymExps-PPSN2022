###################################################################
# Some arbitrary helper functions for which I do not have a place #
###################################################################

import json


# Modify dictionary with list of keys (for recursive dictionaries)
def modify_dict(keys, value, dictionary):
    if isinstance(dictionary[keys[0]], dict):
        dictionary[keys[0]] = modify_dict(keys[1:], value, dictionary[keys[0]])
    else:
        dictionary[keys[0]] = value
    return dictionary


# Pretty print dictionary
def print_dict(config):
    print(json.dumps(config, indent=4))


# Returns whether there is strictly more than one True in a list of booleans
def more_than_one_true(bools):
    num_trues = 0
    for b in bools:
        num_trues += b
        if num_trues > 1:
            return True
    return False


# Takes in a list, l, and a list of boolean tuples, b.
# The size of the individual boolean tuples should be the same size as l.
# This function returns a list of lists where each element is a list containing
# elements of the original list, l, iff True is at the same index for each of the b
def lists_from_bools(lst, bl):

    list_of_lists = []

    for b in bl:
        filtered_list = list_from_bools(lst, b)
        list_of_lists.append(filtered_list)

    return list_of_lists


# Takes a list and a bool tuple and returns a filtered list that contains the original
# list elements iff the bool of bools at the same index is True
def list_from_bools(lst, bools):

    filtered_list = []

    # Check bools tuple and list are the same length
    if len(bools) is not len(lst):
        raise ValueError('Length of bools: {} is not equal to length of lst{}'
                         .format(len(bools), len(lst)))

    for v in zip(bools, lst):
        if v[0]:
            filtered_list.append(v[1])

    return filtered_list


# Removes a number of directories from a path string
def remove_dirs_from_path(path, num_dirs_removed=1):
    return '/'.join(path.split('/')[num_dirs_removed:])
