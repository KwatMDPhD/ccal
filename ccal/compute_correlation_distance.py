from numpy import absolute, where
from scipy.spatial.distance import correlation


def compute_correlation_distance(x, y):

    correlation_distance = correlation(x, y)

    return where(absolute(correlation_distance) < 1e-8, 0, correlation_distance)
