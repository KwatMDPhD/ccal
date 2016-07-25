"""
Computational Cancer Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov


Description:
Supporting module for CCAL.
"""
import os
import datetime
import time

import numpy as np
import pandas as pd

VERBOSE = True


# ======================================================================================================================
# Utilities
# ======================================================================================================================
def _print(string):
    """
    Print `string` with the current time.
    :param string: str, message to be printed
    :return: None
    """
    global VERBOSE
    if VERBOSE:
        print('<{}> {}'.format(datetime.datetime.now().time(), string))


def runtime(function, n_range):
    """
    For i in n_range, get runtimes of function(x, y) where x and y are random vectors of size (i + 1) * 10.
    :param function: function,
    :param n_range: int,
    :param plot:
    :return:
    """
    ns = []
    runtimes = []
    for i in n_range:
        n = (i + 1) * 10
        _print('Getting runtime with vectors (x, y) with size {} ...'.format(n))
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        t0 = time.time()

        function(x, y)

        t = time.time() - t0
        ns.append(n)
        runtimes.append(t)

    return ns, runtimes


# ======================================================================================================================
# File operations
# ======================================================================================================================
def establish_path(path):
    """
    Make 'path' if it doesn't already exist.
    :param path:
    :return: None
    """
    if not (os.path.isdir(path) or os.path.isfile(path) or os.path.islink(path)):
        _print('Path {} doesn\'t exist, creating it ...'.format(path))
        path_dirs = []
        p, q = os.path.split(path)
        while q != '':
            path_dirs.append(q)
            p, q = os.path.split(p)
        path_dirs.append(p)
        partial_path = ''
        for path_element in path_dirs[::-1]:
            partial_path = os.path.join(partial_path, path_element)
            if not (os.path.isdir(partial_path) or os.path.isfile(partial_path) or os.path.islink(partial_path)):
                os.mkdir(partial_path)


def read_gct(filename, fill_na=None, drop_description=True):
    """
    Read .gct `filename` and convert it into a pandas DataFrame.
    :param filename: str, path to a .gct
    :param fill_na: value to replace NaN in the dataframe generated from a `filename`
    :param drop_description: bool, drop the .gct's Description column (#2) or not
    :return: pandas DataFrame, (n_samples, 2 + n_features)
    """
    dataframe = pd.read_csv(filename, skiprows=2, sep='\t')
    if fill_na:
        dataframe.fillna(fill_na, inplace=True)
    column1, column2 = dataframe.columns[:2]
    assert column1 == 'Name', 'Column 1 != \'Name\''
    assert column2 == 'Description', 'Column 2 != \'Description\''

    dataframe.set_index('Name', inplace=True)
    if drop_description:
        dataframe.drop('Description', axis=1, inplace=True)
    dataframe.index.name = None

    return dataframe


def write_gct(pandas_object, filename, index_column=None, description=None):
    """
    Write a `pandas_object` to a `filename` as a .gct.
    :param pandas_object: pandas DataFrame or Serires (n_samples, m_features),
    :param filename: str, path to save the .gct
    :param index_column: str, column to be used as the .gct index
    :param description: array-like (n_samples), description column for the .gct
    """
    pandas_object = pandas_object.copy()

    # Convert Series to DataFrame
    if isinstance(pandas_object, pd.Series):
        pandas_object = pd.DataFrame(pandas_object).T

    # Set output filename
    if not filename.endswith('.gct'):
        filename += '.gct'

    # Set index (Name)
    if index_column:
        pandas_object.set_index(index_column, inplace=True)
    pandas_object.index.name = 'Name'

    n_rows, n_cols = pandas_object.shape[0], pandas_object.shape[1]

    # Set Description
    if description:
        assert len(description) is n_rows, 'Description\' length doesn\'t match the dataframe\'s'
    else:
        description = pandas_object.index
    pandas_object.insert(0, 'Description', description)

    with open(filename, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(n_rows, n_cols))
        pandas_object.to_csv(f, sep='\t')


# ======================================================================================================================
# Simulate
# ======================================================================================================================
def make_random_features(nrow, ncol, n_category=None):
    """
    Make simulation features DataFrame (1D or 2D).
    :param nrow: int, number of rows
    :param ncol: int, number of columns
    :param n_category: None or int, if None, use continuous; if int, use  categorical
    :return: pandas DataFrame, features (`nrow`, `ncol`)
    """
    shape = (nrow, ncol)
    indices = ['Feature {}'.format(i) for i in range(nrow)]
    columns = ['Element {}'.format(i) for i in range(ncol)]
    if n_category:
        features = pd.DataFrame(np.random.random_integers(0, n_category, shape),
                                index=indices,
                                columns=columns)
    else:
        features = pd.DataFrame(np.random.random_sample(shape),
                                index=indices,
                                columns=columns)
    if nrow == 1:
        # Return series
        return features.iloc[0, :]
    else:
        return features


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


def drop_nan_columns(vectors):
    """
    Keep only column positions that are not nan in all vectors.
    :param vectors: list of numpy array, must have the same length (avoid [v1, ..., vn])
    :return: list of numpy arrays,
    """
    for v in vectors[1:]:
        if len(v) != len(vectors[0]):
            raise ValueError('Input arrays have different lengths: {} & {}.'.format(len(v), len(vectors[0])))

    not_nan_filter = np.ones(len(vectors[0]), dtype=bool)
    for v in vectors:
        not_nan_filter &= ~np.isnan(v)

    only_not_nan = []
    for i in range(len(vectors)):
        only_not_nan.append(vectors[i][not_nan_filter])
    return only_not_nan


def add_jitter(vectors, jitter=1E-10):
    """
    Add jitter to vectors.
    :param vectors: numpy array,
    :return: list of numpy arrays
    """
    jittered_vectors = []
    for i in range(len(vectors)):
        jittered_vectors.append(vectors[i] + np.random.random_sample(vectors[i].size) * jitter)
    return jittered_vectors
