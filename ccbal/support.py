"""
Computational Cancer Biology Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Biology, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Biology, UCSD Cancer Center


Description:
TODO
"""
import datetime

import numpy as np
import pandas as pd

VERBOSE = True


# ======================================================================================================================
# Utilities
# ======================================================================================================================
def verbose_print(string):
    """
    Print `string` with the current time.
    :param string: str, message to be printed
    :return: None
    """
    global VERBOSE
    if VERBOSE:
        print('<{}> {}'.format(datetime.datetime.now().time(), string))


# ======================================================================================================================
# File operations
# ======================================================================================================================
def read_gct(filename, fill_na=None):
    """
    Read .gct `filename` and convert it into a pandas DataFrame.
    :param filename: str, path to a .gct
    :param fill_na: value to replace NaN in the dataframe generated from a `filename`
    :return: pandas DataFrame, (n_samples, 2 + n_features)
    """
    dataframe = pd.read_csv(filename, skiprows=2, sep='\t')
    if fill_na:
        dataframe.fillna(fill_na, inplace=True)
    column1, column2 = dataframe.columns[:2]
    assert column1 == 'Name', 'Column 1 != "Name"'
    assert column2 == 'Description', 'Column 2 != "Description"'

    dataframe.set_index('Name', inplace=True)
    dataframe.index.name = None

    return dataframe


def write_gct(dataframe, filename, description=None, index_column=None):
    """
    Write a `dataframe` to a `filename` as a .gct.
    :param dataframe: pandas DataFrame (n_samples, n_features),
    :param filename: str, path
    :param description: array-like (n_samples), description column for the .gct
    :param index_column: str, column to be used as the .gct index
    """
    # Set output filename
    if not filename.endswith('.gct'):
        filename += '.gct'

    # Set index (Name)
    if index_column:
        dataframe.set_index(index_column, inplace=True)
    dataframe.index.name = 'Name'

    n_rows, n_cols = dataframe.shape[0], dataframe.shape[1]

    # Set Description
    if description:
        assert len(description) == n_rows, 'Description\' length doesn\'t match the dataframe\'s'
    else:
        description = dataframe.index
    dataframe.insert(0, 'Description', description)

    with open(filename, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(n_rows, n_cols))
        dataframe.to_csv(f, sep='\t')


# ======================================================================================================================
# Simulate
# ======================================================================================================================
def simulate_x_y(n, rho, threshold=3):
    """
    Generate 2 normal random vectors with correlation `rho` of length `n`.
    :param n: int, size of the output arrays
    :param rho: rho, correlation
    :param threshold: float, max absolute value in the data
    :return: 2 array-like, 2 normal random vectors with correlation `rho` of length `n`
    """
    means = [0, 1]
    stds = [0.5, 0.5]
    covs = [[stds[0] ** 2, stds[0] * stds[1] * rho], [stds[0] * stds[1] * rho, stds[1] ** 2]]

    m = np.random.multivariate_normal(means, covs, n).T
    x = (m[0] - np.mean(m[0])) / np.std(m[0])
    y = (m[1] - np.mean(m[1])) / np.std(m[1])

    if threshold:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        for i in range(n):
            if x[i] > threshold:
                x[i] = threshold
            elif x[i] < -threshold:
                x[i] = -threshold
            if y[i] > threshold:
                y[i] = threshold
            elif y[i] < -threshold:
                y[i] = -threshold
    return x, y
