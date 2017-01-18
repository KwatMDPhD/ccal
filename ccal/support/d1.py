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

from numpy import array, asarray, empty_like
from pandas import DataFrame, Series
from .log import print_log


def make_series(iterable, index=None, name=None):
    """

    :param iterable:
    :param index:
    :param name:
    :return:
    """

    return Series(iterable, index=index, name=name)


def drop_na_1d(df, axis=0, how='all'):
    """

    :param df:
    :param axis: int;
    :param how:
    :return:
    """

    if axis == 0:
        axis_name = 'column'
    else:
        axis_name = 'row'

    if how == 'any':
        nas = df.isnull().any(axis=axis)
    elif how == 'all':
        nas = df.isnull().all(axis=axis)
    else:
        raise ValueError('Unknown \'how\' \'{}\'; pick from (\'any\', \'all\').'.format(how))

    if any(nas):
        df = df.ix[~nas, :]
        print_log('Dropped {} {}(s) without any value: {}'.format(nas.sum(), axis_name, nas.index[nas].tolist()))

    return df


def quantize(array_, precision_factor):
    """
    Return a copy of vector that is scaled by precision_factor and then rounded to the nearest integer.
    To re-scale, simply divide by precision_factor.
    Note that because of rounding, an open interval from (x, y) will give rise to up to
    (x - y) * precision_factor + 1 bins.
    :param array_:
    :param precision_factor:
    :return:
    """

    return (asarray(array_) * precision_factor).round(0)


def discretize_categories(iterable):
    """

    :param iterable:
    :return:
    """

    uniques = sorted(set(iterable))

    discretize = False
    for v in uniques:
        if isinstance(v, str):
            discretize = True

    if discretize:  # Discretize and return an array
        str_to_int_map = {}
        for i, v in enumerate(uniques):
            str_to_int_map[v] = i

        ints = empty_like(iterable, dtype=int)
        for i, v in enumerate(iterable):
            ints[i] = str_to_int_map[v]

        return ints

    else:  # Do nothing and return as an array
        return array(iterable)


def flatten_nested_iterable(nested_iterable, list_type=(list, tuple)):
    """
    Flatten an arbitrarily-deep nested_list.
    :param nested_iterable: a list to flatten_nested_iterables
    :param list_type: valid variable types to flatten_nested_iterables
    :return: list; a flattened list
    """

    nested_iterable = list(nested_iterable)
    i = 0
    while i < len(nested_iterable):
        while isinstance(nested_iterable[i], list_type):
            if not nested_iterable[i]:
                nested_iterable.pop(i)
                i -= 1
                break
            else:
                nested_iterable[i:i + 1] = nested_iterable[i]
        i += 1
    return nested_iterable


def group_iterable(iterable, n, partial_final_item=False):
    """
    Given iterable, return sub-lists made of n items.
    :param iterable:
    :param n:
    :param partial_final_item:
    :return:
    """

    accumulator = []
    for item in iterable:
        accumulator.append(item)
        if len(accumulator) == n:
            yield accumulator
            accumulator = []
    if len(accumulator) != 0 and (len(accumulator) == n or partial_final_item):
        yield accumulator


def get_unique_in_order(iterable):
    """
    Get unique elements in order or appearance in iterable.
    :param iterable: iterable;
    :return: list;
    """

    unique_in_order = []
    for x in iterable:
        if x not in unique_in_order:
            unique_in_order.append(x)
    return unique_in_order


def explode(series):
    """
    Make a label-x-sample binary matrix from a Series.
    :param series: Series;
    :return: DataFrame; (n_labels, n_samples)
    """

    # Make an empty DataFrame (n_unique_labels, n_samples)
    label_x_sample = DataFrame(index=sorted(set(series)), columns=series.index)

    # Binarize each unique label
    for i in label_x_sample.index:
        label_x_sample.ix[i, :] = (series == i).astype(int)

    return label_x_sample
