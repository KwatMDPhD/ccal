from numpy import log


def compute_1d_array_entropy(_1d_array):

    probability = _1d_array / _1d_array.sum()

    return -(probability * log(probability)).sum()
