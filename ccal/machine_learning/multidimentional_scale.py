"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from sklearn.manifold import MDS

from .. import RANDOM_SEED
from .score import compute_similarity_matrix


# TODO: set better default parameters: lower eps and n_init
def mds(matrix,
        n_components=2,
        dissimilarity='euclidean',
        metric=True,
        n_init=1000,
        max_iter=1000,
        verbose=0,
        eps=1e-3,
        n_jobs=1,
        random_state=RANDOM_SEED):
    """
    Multidimensional-scale rows of matrix from <n_dimensions>D into <n_components>D.
    :param matrix: DataFrame; (n_points, n_dimensions)
    :param matrix:
    :param n_components:
    :param dissimilarity: str or function; given metric or capable of computing the distance between 2 array-likes
    :param metric:
    :param n_init: int;
    :param max_iter: int;
    :param verbose:
    :param eps:
    :param n_jobs:
    :param random_state: int; random seed for the initial coordinates
    :return: ndarray; (n_points, n_components)
    """

    if isinstance(dissimilarity, str):
        mds_obj = MDS(n_components=n_components,
                      dissimilarity=dissimilarity,
                      metric=metric,
                      n_init=n_init,
                      max_iter=max_iter,
                      verbose=verbose,
                      eps=eps,
                      n_jobs=n_jobs,
                      random_state=random_state)
        coordinates = mds_obj.fit_transform(matrix)

    else:  # Compute distances using dissimilarity, a function
        mds_obj = MDS(n_components=n_components,
                      dissimilarity='precomputed',
                      metric=metric,
                      n_init=n_init,
                      max_iter=max_iter,
                      verbose=verbose,
                      eps=eps,
                      n_jobs=n_jobs,
                      random_state=random_state)
        coordinates = mds_obj.fit_transform(
            compute_similarity_matrix(
                matrix, matrix, dissimilarity, is_distance=True, axis=1))

    return coordinates
