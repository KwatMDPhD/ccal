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

from numpy import array, asarray, empty, log2, zeros
from numpy.random import seed, shuffle
from pandas import DataFrame

from .. import RANDOM_SEED
from ..mathematics.information import information_coefficient
from ..support.log import print_log


def compute_association_and_pvalue(x,
                                   y,
                                   function=information_coefficient,
                                   n_permutations=1000,
                                   random_seed=RANDOM_SEED):
    """
    Compute function(x, y) and p-value using permutation test.
    :param x: array-like;
    :param y: array-like;
    :param function: function;
    :param n_permutations: int; number of permutations for the p-value permutation test
    :param random_seed: int or array-like;
    :return: float and float; score and p-value
    """

    # Compute score
    score = function(x, y)

    # Compute scores against permuted target
    permutation_scores = empty(n_permutations)
    shuffled_target = array(y)
    seed(random_seed)
    for p in range(n_permutations):
        shuffle(shuffled_target)
        permutation_scores[p] = function(x, shuffled_target)

    # Compute p-value
    if 0 <= score:
        p_val = (permutation_scores >= score).sum() / n_permutations
    else:
        p_val = (permutation_scores <= score).sum() / n_permutations

    if p_val == 0:
        p_val = 1 / n_permutations

    return score, p_val


def compute_similarity_matrix(matrix1,
                              matrix2,
                              function,
                              axis=0,
                              is_distance=False):
    """
    Make association or distance matrix of matrix1 and matrix2 by row (axis=1) or by column (axis=0).
    :param matrix1: pandas DataFrame;
    :param matrix2: pandas DataFrame;
    :param function: function; function used to compute association or dissociation
    :param axis: int; 0 for row-wise and 1 column-wise comparison
    :param is_distance: bool; True for distance and False for association
    :return: pandas DataFrame; (n, n); association or distance matrix
    """

    # Rotate matrices to make the comparison by row
    if axis == 1:
        matrix1 = matrix1.copy()
        matrix2 = matrix2.copy()
    else:
        matrix1 = matrix1.T
        matrix2 = matrix2.T

    # Work with array
    m1 = asarray(matrix1)
    m2 = asarray(matrix2)

    # Number of comparables
    n_1 = m1.shape[0]
    n_2 = m2.shape[0]

    # Compare
    compared_matrix = empty((n_1, n_2))
    for i_1 in range(n_1):
        print_log(
            'Computing associations (axis={}) between matrices ({}/{}) ...'.
            format(axis, i_1 + 1, n_1))
        for i_2 in range(n_2):
            compared_matrix[i_1, i_2] = function(m1[i_1, :], m2[i_2, :])

    if is_distance:  # Convert association to distance
        print_log('Converting association to distance (1 - association) ...')
        compared_matrix = 1 - compared_matrix

    return DataFrame(
        compared_matrix, index=matrix1.index, columns=matrix2.index)


def compute_sliding_mean(vector, window_size=1):
    """
    Return a vector of means for each window_size in vector.
    :param vector:
    :param window_size:
    :return:
    """

    m = zeros(len(vector))
    for i in range(len(vector)):
        m[i] = (vector[max(0, i - window_size):min(
            len(vector), i + window_size + 1)]
                ).sum() / float(window_size * 2 + 1)
    return m


def compute_geometric_mean(vector):
    """
    Return the geometric mean (the n-th root of the product of n terms) of an vector.
    :param vector:
    :return:
    """

    product = vector[0]
    for n in vector[1:]:
        product *= n
    return product**(1 / len(vector))


def compute_fold_change(dataframe, column_name_before, column_name_after):
    """

    :param dataframe: DataFrame;
    :param column_name_before: str or index;
    :param column_name_after: str or index;
    :return: DataFrame;
    """

    tmp1 = dataframe.ix[dataframe.ix[:, column_name_before] != 0,
                        column_name_before]
    tmp2 = dataframe.ix[dataframe.ix[:, column_name_after] != 0,
                        column_name_after]

    min_before = tmp1.min()
    min_after = tmp2.min()

    # Compute normalization factor
    norm_factor = 1 / (tmp2.sum() / tmp1.sum())

    print('min_before={}\nmin_after={}\nnorm_factor={}'.format(
        min_before, min_after, norm_factor))
    return dataframe.apply(
        compute_fold_change_for_a_feature,
        axis=1,
        args=(column_name_before, column_name_after, min_before, min_after,
              norm_factor))


def compute_fold_change_for_a_feature(series, before_col_name, after_col_name,
                                      before_min, after_min, norm_factor):
    """

    :param series:
    :param before_col_name:
    :param after_col_name:
    :param before_min:
    :param after_min:
    :param norm_factor:
    :return:
    """

    fpkm1 = max(series.ix[before_col_name], before_min)
    fpkm2 = max(series.ix[after_col_name], after_min)

    return log2(norm_factor * fpkm2 / fpkm1)
