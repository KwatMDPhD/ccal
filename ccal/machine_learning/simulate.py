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

from numpy.random import random_integers, random_sample, seed
from pandas import DataFrame

from .. import RANDOM_SEED


def simulate_dataframe_or_series(n_rows,
                                 n_cols,
                                 n_categories=None,
                                 random_seed=RANDOM_SEED):
    """
    Simulate DataFrame (2D) or Series (1D).
    :param n_rows: int;
    :param n_cols: int;
    :param n_categories: None or int; continuous if None and categorical if int
    :param random_seed: int or array-like;
    :return: pandas DataFrame or Series; (n_rows, n_cols) or (1, n_cols)
    """

    # Set up indices and column names
    indices = ['Feature {}'.format(i) for i in range(n_rows)]
    columns = ['Sample {}'.format(i) for i in range(n_cols)]

    # Set up data type: continuous, categorical, or binary
    seed(random_seed)
    if n_categories:
        features = DataFrame(
            random_integers(0, n_categories - 1, (n_rows, n_cols)),
            index=indices,
            columns=columns)
    else:
        features = DataFrame(
            random_sample((n_rows, n_cols)), index=indices, columns=columns)

    if n_rows == 1:  # Return as series if there is only 1 row
        return features.iloc[0, :]
    else:
        return features
