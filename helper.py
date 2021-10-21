###################################################################
# Some arbitrary helper functions for which I do not have a place #
###################################################################


#Returns whether there is strictly more than one True in a list of booleans
def more_than_one_true(bools):
    num_trues = 0
    for b in bools:
        num_trues += b
        if num_trues > 1:
            return True
    return False


#Takes in a list, l, and a list of boolean tuples, b.
#The size of the individual boolean tuples should be the same size as l.
#This function returns a list of lists where each element is a list containing
#elements of the original list, l, iff True is at the same index for each of the
#b
def lists_from_bools(l, bl):

    list_of_lists = []

    for b in bl:
        filtered_list = list_from_bools(l, b)
        list_of_lists.append(filtered_list)

    return list_of_lists


#Takes a list and a bool tuple and returns a filtered list that contains the original
#list elements iff the bool of bools at the same index is True
def list_from_bools(l, bools):

    filtered_list = []

    #Check bools tuple and list are the same length
    if len(bools) is not len(l):
        raise ValueError('Length of bools: {} is not equal to length of l{}' \
            .format(len(bools), len(l)))

    for v in zip(bools, l):
        if v[0]:
            filtered_list.append(v[1])

    return filtered_list
