"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from numpy import ones, isnan
from numpy.random import seed, shuffle
from pandas import concat

from .. import RANDOM_SEED
from ..support.system import get_random_state


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

    :param df:
    :param value:
    :param axis:
    :return:
    """

    if axis == 0:
        dropped = (df == value).all(axis=0)
        if any(dropped):
            print('Removed {} column index(ices) whoes values are all {}.'.format(dropped.sum(), value))
        return df.ix[:, ~dropped]

    elif axis == 1:
        dropped = (df == value).all(axis=1)
        if any(dropped):
            print('Removed {} row index(ices) whoes values are all {}.'.format(dropped.sum(), value))
        return df.ix[~dropped, :]


def shuffle_dataframe(df, axis=0, random_seed=RANDOM_SEED):
    """

    :param df: DataFrame;
    :param axis: int;
    :param random_seed: int or array-like;
    :return: DataFrame;
    """

    df = df.copy()

    seed(random_seed)
    if axis == 0:
        for c_i, col in df.iteritems():
            # Shuffle in place
            shuffle(col)

    elif axis == 1:
        for r_i, row in df.iterrows():
            # Shuffle in place
            shuffle(row)

    return df


def split_dataframe_for_random(df, n_jobs, random_seed, skipper, for_skipper):
    """
    Split df into n_jobs blocks (by row). Assign random states for the blocks' 1st rows, so that the assigned random
    state is the random state that would have assigned to them if a random operation, skipper, operates on each row of
    non-split data. Leftovers become its own block (the last block).
    :param df: DataFrame;
    :param n_jobs: int;
    :param random_seed: int;
    :param skipper: str;
    :param for_skipper: object;
    :return: list; list of tuples [{split_df1, random_state1}, {split_df2, random_state2} ...]
    """

    # Get number of rows per job
    n_per_job = df.shape[0] // n_jobs

    # List of functional args in each job
    args = []

    # Leftover rows
    leftovers = list(df.index)

    # Set the initial random state
    seed(random_seed)
    random_state = get_random_state('Before skipping')

    last_i = 0
    shuffled_for_skipping_random_states = list(range(df.shape[1]))
    for i in range(n_jobs):

        # Indeces for this job
        start_i = i * n_per_job
        end_i = (i + 1) * n_per_job
        split_df = df.iloc[start_i: end_i, :]

        # Skip random states (number of indeces for the previous job times)
        for r_i in range(last_i, start_i):
            exec(skipper)
            random_state = get_random_state('{}: skipping index {}'.format(i, r_i))

        last_i = start_i

        # Update functional args for this job
        args.append((split_df, random_state))

        # Remove included indeces
        for included_i in split_df.index:
            leftovers.remove(included_i)

    if leftovers:
        print('Leftovers: {}'.format(leftovers))

        # Skip random states
        start_i = n_jobs * n_per_job
        for r_i in range(last_i, start_i):
            exec(skipper)
            random_state = get_random_state('Skipping index {}'.format(r_i))

        # Update functional args for this job
        args.append((df.ix[leftovers, :], random_state))

    return args
