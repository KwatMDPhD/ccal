from numpy import absolute, where
from scipy.spatial.distance import correlation


def compute_correlation_distance_between_2_1d_arrays(_1d_array_0, _1d_array_1):

    correlation_distance = correlation(_1d_array_0, _1d_array_1)

    return where(absolute(correlation_distance) < 1e-8, 0, correlation_distance)
