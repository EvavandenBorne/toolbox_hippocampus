#

import numpy as np

# Activation function (transfer (f/I) curve)
def activation_function(current, parameter):
    #return parameter * np.log(1 + np.exp(current / parameter)) #
    return 1/(1 + np.exp(-current))

#
def closest_location_index(location_range, target_location):
    index = np.argmin(np.absolute(location_range - target_location))
    return index

#
def permute_matrix(M, permutation):
    X = M[:, permutation]
    Y = X[permutation,:]
    return Y

