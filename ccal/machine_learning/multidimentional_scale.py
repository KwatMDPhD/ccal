"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from pandas import DataFrame
from sklearn.manifold import MDS

from .. import RANDOM_SEED
from .normalize import normalize_dataframe_or_series
from .score import compute_similarity_matrix


def mds(matrix, distance_function=None, mds_seed=RANDOM_SEED, n_init=1000, max_iter=1000, standardize=True):
    """
    Multidimentional-scale rows of matrix from <n_cols>D into 2D.
    :param matrix: DataFrame; (n_points, n_dimentions)
    :param distance_function: function; capable of computing the distance between 2 vectors
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param n_init: int;
    :param max_iter: int;
    :param standardize: bool;
    :return: DataFrame; (n_points, 2 ('x', 'y'))
    """

    if distance_function:  # Use precomputed distances
        mds_obj = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(
            compute_similarity_matrix(matrix, matrix, distance_function, is_distance=True, axis=1))

    else:  # Use Euclidean distances
        mds_obj = MDS(random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(matrix)

    # Convert to DataFrame
    coordinates = DataFrame(coordinates, index=matrix.index, columns=['x', 'y'])

    if standardize:  # Rescale coordinates between 0 and 1
        coordinates = normalize_dataframe_or_series(coordinates, method='0-1', axis=0)

    return coordinates
