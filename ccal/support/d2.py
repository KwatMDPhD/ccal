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

from numpy import array, ones, isnan
from numpy.random import seed, shuffle
from pandas import DataFrame, concat, Series
from scipy.cluster.hierarchy import linkage, dendrogram

from .. import RANDOM_SEED
from ..support.d1 import drop_na_1d, normalize_1d
from ..support.log import print_log


def drop_na_2d(df, axis='both', how='all'):
    """

    :param df:
    :param axis:
    :param how:
    :return:
    """

    if axis in ('both', 1):
        df = drop_na_1d(df, axis=1, how=how)

    if axis in ('both', 0):
        df = drop_na_1d(df, axis=0, how=how)

    return df


def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all arrays.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """

    # Keep all column indices
    not_nan_filter = ones(len(arrays[0]), dtype=bool)

    # Keep column indices without missing value in all arrays
    for a in arrays:
        not_nan_filter &= ~isnan(a)

    return [a[not_nan_filter] for a in arrays]


def get_top_and_bottom_indices(df, column_name, threshold, max_n=None):
    """

    :param df: DataFrame;
    :param column_name: str;
    :param threshold: number; quantile if < 1; ranking number if >= 1
    :param max_n: int; maximum number of rows
    :return: list; list of indices
    """

    if threshold < 1:
        column = df.ix[:, column_name]

        is_top = column >= column.quantile(threshold)
        is_bottom = column <= column.quantile(1 - threshold)

        top_and_bottom = df.index[is_top | is_bottom].tolist()

        if max_n and max_n < len(top_and_bottom):
            threshold = max_n // 2

    if 1 <= threshold:
        if 2 * threshold <= df.shape[0]:
            top_and_bottom = df.index[:threshold].tolist() + df.index[-threshold:].tolist()
        else:
            top_and_bottom = df.index

    return top_and_bottom


def get_dendrogram_leaf_indices(matrix):
    """

    :param matrix:
    :return:
    """

    row_leaves = dendrogram(linkage(matrix), no_plot=True)['leaves']
    col_leaves = dendrogram(linkage(matrix.T), no_plot=True)['leaves']
    return row_leaves, col_leaves


def split_slices(df, index, splitter, ax=0):
    """

    :param df:
    :param index:
    :param splitter:
    :param ax:
    :return:
    """

    splits = []

    if ax == 0:  # Split columns
        df = df.T

    for s_i, s in df.iterrows():

        old = s.ix[index]

        for new in old.split(splitter):
            splits.append(s.replace(old, new))

    # Concatenate
    if ax == 0:
        return concat(splits, axis=1)
    elif ax == 1:
        return concat(splits, axis=1).T


def drop_uniform_slice_from_dataframe(df, value, axis=0):
    """
    Drop slice that contains only value from df.
    :param df: DataFrame;
    :param value: obj; if a slice contains only obj, the slice will be dropped
    :param axis: int; 0 for dropping column; and 1 for dropping row
    :return: DataFrame; DataFrame without any slice that contains only value
    """

    if axis == 0:
        dropped = (df == value).all(axis=0)
        if any(dropped):
            print_log('Removed {} column index(ices) whose values are all {}.'.format(dropped.sum(), value))
        return df.ix[:, ~dropped]

    elif axis == 1:
        dropped = (df == value).all(axis=1)
        if any(dropped):
            print_log('Removed {} row index(ices) whose values are all {}.'.format(dropped.sum(), value))
        return df.ix[~dropped, :]


def shuffle_matrix(matrix, axis=0, random_seed=RANDOM_SEED):
    """

    :param matrix: DataFrame;
    :param axis: int; {0, 1}
    :param random_seed: int or array-like;
    :return: 2D array or DataFrame;
    """

    seed(random_seed)

    if isinstance(matrix, DataFrame):  # Work with 2D array (copy)
        a = array(matrix)
    else:
        a = matrix.copy()

    if axis == 0:  # Shuffle each column
        for i in range(a.shape[1]):
            shuffle(a[:, i])
    elif axis == 1:  # Shuffle each row
        for i in range(a.shape[0]):
            shuffle(a[i, :])
    else:
        ValueError('Unknown axis {}; choose from {0, 1}.')

    if isinstance(matrix, DataFrame):  # Return DataFrame
        return DataFrame(a, index=matrix.index, columns=matrix.columns)
    else:  # Return 2D array
        return a


def split_dataframe(df, n_split, axis=0):
    """
    Split df into n_split blocks (by row).
    :param df: DataFrame;
    :param n_split: int; 0 < n_split <= n_rows
    :param axis: int; {0, 1}
    :return: list; list of dataframes
    """

    # TODO: implement axis logic

    if df.shape[0] < n_split:
        raise ValueError('n_split ({}) can\'t be greater than the number of rows ({}).'.format(n_split, df.shape[0]))
    elif n_split <= 0:
        raise ValueError('n_split ({}) can\'t be less than 0.'.format(n_split))

    n = df.shape[0] // n_split

    splits = []

    for i in range(n_split):
        start_i = i * n
        end_i = (i + 1) * n
        splits.append(df.iloc[start_i: end_i, :])

    i = n * n_split
    if i < df.shape[0]:
        splits.append(df.ix[i:])

    return splits


def normalize_2d_or_1d(dataframe, method, axis=None, n_ranks=10000,
                       normalizing_mean=None, normalizing_std=None,
                       normalizing_min=None, normalizing_max=None,
                       normalizing_size=None):
    """
    Normalize a DataFrame or Series.
    :param dataframe: DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :param axis: int; None for global, 0 for by-column, and 1 for by-row normalization
    :param normalizing_mean:
    :param normalizing_std:
    :param normalizing_min:
    :param normalizing_max:
    :param normalizing_size:
    :return: DataFrame or Series; normalized DataFrame or Series
    """

    if isinstance(dataframe, Series):  # Series
        return normalize_1d(dataframe, method, n_ranks=n_ranks,
                            normalizing_mean=normalizing_mean, normalizing_std=normalizing_std,
                            normalizing_min=normalizing_min, normalizing_max=normalizing_max,
                            normalizing_size=normalizing_size)

    elif isinstance(dataframe, DataFrame):

        if axis == 0 or axis == 1:  # Normalize Series by axis
            return dataframe.apply(normalize_1d, **{'method': method, 'n_ranks': n_ranks,
                                                    'normalizing_mean': normalizing_mean,
                                                    'normalizing_std': normalizing_std,
                                                    'normalizing_min': normalizing_min,
                                                    'normalizing_max': normalizing_max,
                                                    'normalizing_size': normalizing_size}, axis=axis)

        else:  # Normalize globally

            # Get size
            if normalizing_size is not None:
                size = normalizing_size
            else:
                size = dataframe.values.size

            if method == '-0-':

                # Get mean
                if normalizing_mean is not None:
                    mean = normalizing_mean
                else:
                    mean = dataframe.values.mean()

                # Get STD
                if normalizing_std is not None:
                    std = normalizing_std
                else:
                    std = dataframe.values.std()

                # Normalize
                if std == 0:
                    print('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing ...')
                    return dataframe / size
                else:
                    return (dataframe - mean) / std

            elif method == '0-1':

                # Get min
                if normalizing_min is not None:
                    min_ = normalizing_min
                else:
                    min_ = dataframe.values.min()

                # Get max
                if normalizing_max is not None:
                    max_ = normalizing_max
                else:
                    max_ = dataframe.values.max()

                # Normalize
                if max_ - min_ == 0:
                    print('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing ...')
                    return dataframe / size
                else:
                    return (dataframe - min_) / (max_ - min_)

            elif method == 'rank':
                raise ValueError('Normalizing combination of \'rank\' & axis=\'all\' has not been implemented yet.')
