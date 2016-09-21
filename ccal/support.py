"""
Computational Cancer Analysis Library v0.1

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov
"""
from numpy import finfo, ones, isnan
from pandas import DataFrame, Series, read_csv

# ======================================================================================================================
# Set up global parameters
# ======================================================================================================================
VERBOSE = True
SEED = 20121020
EPS = finfo(float).eps


# ======================================================================================================================
# Set up system
# ======================================================================================================================
def install_libraries(libraries_needed):
    """
    Check if `libraries_needed` are installed; if not then install using pip.
    :param libraries_needed: iterable; library names
    :return: None
    """
    from pip import get_installed_distributions, main

    print_log('Checking library dependencies ...')
    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]
    # Install missing libraries from `libraries_needed`
    for lib in libraries_needed:
        if lib not in libraries_installed:
            print_log('{} not found! Installing {} using pip ...'.format(lib, lib))
            main(['install', lib])
    # Print versions of `libraries_needed`
    print_log('Using the following libraries (plus libraries from the Anaconda distribution):')
    for lib in get_installed_distributions():
        if lib.key in libraries_needed:
            print_log('\t{} (v{})'.format(lib.key, lib.version))


# TODO: seed globally
def plant_seed(a_seed=SEED):
    """
    Set random seed.
    :param a_seed: int;
    :return: None
    """
    from random import seed

    seed(a_seed)
    print_log('Planted a random seed {}.'.format(SEED))


# ======================================================================================================================
# Log
# ======================================================================================================================
# TODO: use logging
def print_log(string):
    """
    Print `string` together with logging information.
    :param string: str; message to printed
    :return: None
    """
    global VERBOSE
    if VERBOSE:
        print('<{}> {}'.format(timestamp(time_only=True), string))


def timestamp(time_only=False):
    """
    Get the current time stamp.
    :param time_only: bool;
    :return: str;
    """
    from datetime import datetime

    if time_only:
        formatter = '%H%M%S'
    else:
        formatter = '%Y%m%d-%H%M%S'
    return datetime.now().strftime(formatter)


# ======================================================================================================================
# Operate on files
# ======================================================================================================================
def establish_path(filepath):
    """
    Make directories up to `fullpath` if they don't already exist.
    :param filepath: str;
    :return: None
    """
    from os import path, mkdir

    prefix, suffix = path.split(filepath)
    if not (path.isdir(prefix) or path.isfile(prefix) or path.islink(prefix)):
        print_log('Directory {} doesn\'t exist, creating it ...'.format(prefix))
        dirs = []
        prefix, suffix = path.split(filepath)
        dirs.append(prefix)
        while prefix != '' and suffix != '':
            prefix, suffix = path.split(prefix)
            if prefix:
                dirs.append(prefix)
        for d in reversed(dirs):
            if not (path.isdir(d) or path.isfile(d) or path.islink(d)):
                mkdir(d)
                print_log('Created directory {}.'.format(d))


def write_dictionary(dictionary, filepath, key_name, value_name):
    """
    Write a dictionary as a tab-separated file.
    :param dictionary: dict;
    :param filepath: str;
    :param key_name; str;
    :param value_name; str;
    :return: None
    """
    with open(filepath, 'w') as f:
        f.write('{}\t{}\n'.format(key_name, value_name))
        for k, v in sorted(dictionary.items()):
            f.writelines('{}\t{}\n'.format(k, v))


def read_gct(filepath, fill_na=None, drop_description=True):
    """
    Read a .gct (`filepath`) and convert it into a pandas DataFrame.
    :param filepath: str;
    :param fill_na: *; value to replace NaN in the DataFrame
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :return: pandas DataFrame; [n_samples, n_features (or n_features + 1 if not dropping the Description column)]
    """
    df = read_csv(filepath, skiprows=2, sep='\t')
    if fill_na:
        df.fillna(fill_na, inplace=True)
    c1, c2 = df.columns[:2]
    if c1 != 'Name':
        if c1.strip() != 'Name':
            raise ValueError('Column 1 != \'Name\'.')
        else:
            raise ValueError('Column 1 has more than 1 extra space around \'Name\'. Please strip it.')
    if c2 != 'Description':
        if c2.strip() != 'Description':
            raise ValueError('Column 2 != \'Description\'')
        else:
            raise ValueError('Column 2 has more than 1 extra space around \'Description\'. Please strip it.')
    df.set_index('Name', inplace=True)
    df.index.name = None
    if drop_description:
        df.drop('Description', axis=1, inplace=True)
    return df


def write_gct(pandas_object, filepath, descriptions=None):
    """
    Write a `pandas_object` to a `filepath` as a .gct.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of `pandas_object`); description column for the .gct
    :return: None
    """
    obj = pandas_object.copy()

    # Convert Series to DataFrame
    if isinstance(obj, Series):
        obj = DataFrame(obj).T

    obj.index.name = 'Name'
    if descriptions:
        obj.insert(0, 'Description', descriptions)
    else:
        obj.insert(0, 'Description', obj.index)
    if not filepath.endswith('.gct'):
        filepath += '.gct'
    with open(filepath, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(obj.shape[0], obj.shape[1] - 1))
        obj.to_csv(f, sep='\t')


def read_gmt(filepath):
    """
    Read a .gmt file.
    :param filepath:
    :return:
    """
    return read_csv(filepath, sep='\t', index_col=0)


def write_gmt(pandas_object, filepath, descriptions=None):
    """
    Write a `pandas_object` to a `filepath` as a .gmt.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of `pandas_object`); description column for the .gmt
    :return: None
    """
    obj = pandas_object.copy()
    obj.index.name = 'Name'
    if descriptions:
        obj.insert(0, 'Description', descriptions)
    else:
        obj.insert(0, 'Description', obj.index)
    if not filepath.endswith('.gmt'):
        filepath += '.gmt'
    obj.to_csv(filepath, sep='\t')


def save_nmf_results(nmf_results, filepath_prefix):
    """
    Save `nmf_results` dictionary.
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param filepath_prefix: str; `filepath_prefix`_nmf_k{k}_{w, h}.gct and  will be saved
    :return: None
    """
    establish_path(filepath_prefix)
    for k, v in nmf_results.items():
        write_gct(v['W'], filepath_prefix + '_nmf_k{}_w.gct'.format(k))
        write_gct(v['H'], filepath_prefix + '_nmf_k{}_h.gct'.format(k))


# ======================================================================================================================
# Simulate
# ======================================================================================================================
def make_random_features(n_rows, n_cols, n_categories=None):
    """
    Simulate DataFrame (2D) or Series (1D).
    :param n_rows: int;
    :param n_cols: int;
    :param n_categories: None or int; continuous if None and categorical if int
    :return: pandas DataFrame or Series; (`n_rows`, `n_cols`) or (1, `n_cols`)
    """
    from numpy.random import random_integers, random_sample

    indices = ['Feature {}'.format(i) for i in range(n_rows)]
    columns = ['Element {}'.format(i) for i in range(n_cols)]
    if n_categories:
        features = DataFrame(random_integers(0, n_categories - 1, (n_rows, n_cols)), index=indices, columns=columns)
    else:
        features = DataFrame(random_sample((n_rows, n_cols)), index=indices, columns=columns)
    if n_rows == 1:
        # Return series
        return features.iloc[0, :]
    else:
        return features


# ======================================================================================================================#
# Help
# ======================================================================================================================#
def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all `arrays`.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """
    not_nan_filter = ones(len(arrays[0]), dtype=bool)
    for v in arrays:
        not_nan_filter &= ~isnan(v)
    return [v[not_nan_filter] for v in arrays]


def get_unique_in_order(iterable):
    """
    Get unique elements in order or appearance in `iterable`.
    :param iterable: iterable;
    :return: list;
    """
    unique_in_order = []
    for x in iterable:
        if x not in unique_in_order:
            unique_in_order.append(x)
    return unique_in_order


def explode(series, filepath=None):
    """
    Make a label-x-sample binary matrix from a Series.
    :param series: pandas Series;
    :param filepath: str;
    :return: pandas DataFrame;
    """
    label_x_sample = DataFrame(index=sorted(set(series)), columns=series.index)
    for i in label_x_sample.index:
        label_x_sample.ix[i, :] = (series == i).astype(int)
    if filepath:
        establish_path(filepath)
        write_gct(label_x_sample, filepath)

    return label_x_sample
