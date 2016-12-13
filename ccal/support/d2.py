"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from numpy import ones, sum, isnan
from numpy.random import seed, shuffle
from pandas import concat

from .. import RANDOM_SEED


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


def get_top_and_bottom_indices(dataframe, column_name, threshold, max_n=None):
    """

    :param dataframe: DataFrame;
    :param column_name: str;
    :param threshold: number; quantile if < 1; ranking number if >= 1
    :param max_n: int; maximum number of rows
    :return: list; list of indices
    """

    if threshold < 1:
        column = dataframe.ix[:, column_name]

        is_top = column >= column.quantile(threshold)
        is_bottom = column <= column.quantile(1 - threshold)

        top_and_bottom = dataframe.index[is_top | is_bottom].tolist()

        if max_n and max_n < len(top_and_bottom):
            threshold = max_n // 2

    if 1 <= threshold:
        if 2 * threshold <= dataframe.shape[0]:
            top_and_bottom = dataframe.index[:threshold].tolist() + dataframe.index[-threshold:].tolist()
        else:
            top_and_bottom = dataframe.index

    return top_and_bottom


def split_slices(dataframe, index, splitter, ax=0):
    """

    :param dataframe:
    :param index:
    :param splitter:
    :param ax:
    :return:
    """

    splits = []

    if ax == 0:  # Split columns
        dataframe = dataframe.T

    for s_i, s in dataframe.iterrows():

        old = s.ix[index]

        for new in old.split(splitter):
            splits.append(s.replace(old, new))

    # Concatenate
    if ax == 0:
        return concat(splits, axis=1)
    elif ax == 1:
        return concat(splits, axis=1).T


def drop_uniform_slice_from_dataframe(dataframe, value, axis=0):
    """

    :param dataframe:
    :param value:
    :param axis:
    :return:
    """

    if axis == 0:
        dropped = (dataframe == value).all(axis=0)
        if any(dropped):
            print('Removed {} column index(ices) whoes values are all {}.'.format(sum(dropped), value))
        return dataframe.ix[:, ~dropped]

    elif axis == 1:
        dropped = (dataframe == value).all(axis=1)
        if any(dropped):
            print('Removed {} row index(ices) whoes values are all {}.'.format(sum(dropped), value))
        return dataframe.ix[~dropped, :]


def shuffle_dataframe(dataframe, axis=0, random_seed=RANDOM_SEED):
    """

    :param dataframe: DataFrame;
    :param axis: int;
    :param random_seed: int or array-like;
    :return: DataFrame;
    """

    seed(random_seed)

    df = dataframe.copy()

    if axis == 0:
        for c_i, col in df.iteritems():
            # Shuffle in place
            shuffle(col)

    elif axis == 1:
        for r_i, row in df.iterrows():
            # Shuffle in place
            shuffle(row)

    return df
