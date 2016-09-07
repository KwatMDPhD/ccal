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
from numpy import array, asarray, zeros, ones, isnan, exp
from numpy.random import random_integers, random_sample
from pandas import DataFrame, Series, read_csv
from scipy.optimize import curve_fit
from sklearn.manifold import MDS

# ======================================================================================================================
# Parameters
# ======================================================================================================================
VERBOSE = True
SEED = 20121020


# ======================================================================================================================
# Utilities
# ======================================================================================================================
def print_log(string):
    """
    Print `string` together with logging information.
    :param string: str; message to printed
    :return: None
    """
    from datetime import datetime

    global VERBOSE
    if VERBOSE:
        print('<{}> {}'.format(datetime.now().strftime('%H:%M:%S'), string))


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
    print_log('Using the following libraries (in addition to the Anaconda libraries):')
    for lib in get_installed_distributions():
        if lib.key in libraries_needed:
            print_log('\t{} (v{})'.format(lib.key, lib.version))


def plant_seed(a_seed=SEED):
    """
    Set random seed.
    :param a_seed: int;
    :return: None
    """
    from random import seed
    seed(a_seed)
    print_log('Planted a random seed {}.'.format(SEED))


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
        raise ValueError('Column 1 != \'Name\'')
    if c2 != 'Description':
        raise ValueError('Column 2 != \'Description\'')
    df.set_index('Name', inplace=True)
    df.index.name = None
    if drop_description:
        df.drop('Description', axis=1, inplace=True)
    return df


def write_gct(pandas_object, filepath, index_column_name=None, descriptions=None):
    """
    Write a `pandas_object` to a `filepath` as a .gct.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param index_column_name: str; column to be used as the index for the .gct
    :param descriptions: iterable; (n_rows of `pandas_object`); description column for the .gct
    :return: None
    """
    obj = pandas_object.copy()

    # Convert Series to DataFrame
    if isinstance(obj, Series):
        obj = DataFrame(obj).T

    # Set index (Name)
    if index_column_name:
        obj.set_index(index_column_name, inplace=True)
    obj.index.name = 'Name'

    # Set Description
    if descriptions:
        obj.insert(0, 'Description', descriptions)
    else:
        obj.insert(0, 'Description', obj.index)

    # Set output filename suffix
    if not filepath.endswith('.gct'):
        filepath += '.gct'

    with open(filepath, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(*obj.shape))
        obj.to_csv(f, sep='\t')


# ======================================================================================================================#
# Data analysis
# ======================================================================================================================#
def exponential_function(x, a, k, c):
    """
    Apply exponential function on `x`.
    :param x:
    :param a:
    :param k:
    :param c:
    :return:
    """
    return a * exp(k * x) + c


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


def make_random_features(n_rows, n_cols, n_categories=None):
    """
    Simulate DataFrame (2D) or Series (1D).
    :param n_rows: int;
    :param n_cols: int;
    :param n_categories: None or int; continuous if None and categorical if int
    :return: pandas DataFrame or Series; (`n_rows`, `n_cols`) or (1, `n_cols`)
    """
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


def normalize_pandas_object(pandas_object, method='-0-', axis='all'):
    """
    Normalize a pandas object.
    :param pandas_object: pandas DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1'}
    :param axis: str or int; 'all' for global, 0 for by-column, and 1 for by-row normalization
    :return: pandas DataFrame or Series;
    """
    obj = pandas_object.copy()
    print_log('\'{}\' normalizing pandas object with axis={} ...'.format(method, axis))
    if isinstance(obj, DataFrame):
        if method == '-0-':
            if axis == 'all':
                obj_mean = obj.values.mean()
                obj_std = obj.values.std()
                if obj_std == 0:
                    print_log('Warning: tried to \'-0-\' normalize but the standard deviation is 0.')
                    obj = obj / obj.size
                else:
                    obj = obj.applymap(lambda v: (v - obj_mean) / obj_std)
            else:
                obj = obj.apply(lambda r: (r - r.mean()) / r.std(), axis=axis)
        elif method == '0-1':
            if axis == 'all':
                obj_min = obj.values.min()
                obj_max = obj.values.max()
                obj_range = obj_max - obj_min
                if obj_range == 0:
                    print_log('Warning: tried to \'0-1\' normalize but the range is 0.')
                    obj = obj / obj.size
                else:
                    obj = obj.applymap(lambda v: (v - obj_min) / obj_range)
            else:
                obj = obj.apply(lambda r: (r - r.min()) / (r.max() - r.min()), axis=axis)
    elif isinstance(obj, Series):
        obj = normalize_series(obj, method=method)
    return obj


def normalize_series(series, method='-0-'):
    """
     Normalize a pandas `series`.
    :param series: pandas Series;
    :param method: str; normalization type; {'-0-', '0-1'}
    :return: pandas Series;
    """
    if method == '-0-':
        mean = series.mean()
        std = series.std()
        if std == 0:
            print_log('Warning: tried to \'-0-\' normalize but the standard deviation is 0.')
            return series / series.size
        else:
            return (series - mean) / std
    elif method == '0-1':
        smin = series.min()
        smax = series.max()
        if smax - smin == 0:
            print_log('Warning: tried to \'0-1\' normalize but the range is 0.')
            return series / series.size
        else:
            return (series - smin) / (smax - smin)


def compare_matrices(matrix1, matrix2, function, axis=0, is_distance=False):
    """
    Make association or distance matrix of `matrix1` and `matrix2` by row or column.
    :param matrix1: pandas DataFrame;
    :param matrix2: pandas DataFrame;
    :param function: function; function used to compute association or dissociation
    :param axis: int; 0 for by-row and 1 for by-column
    :param is_distance: bool; True for distance and False for association
    :return: pandas DataFrame;
    """
    if axis == 1:
        m1 = matrix1.copy()
        m2 = matrix2.copy()
    else:
        m1 = matrix1.T
        m2 = matrix2.T

    compared_matrix = DataFrame(index=m1.index, columns=m2.index, dtype=float)
    for i, (i1, r1) in enumerate(m1.iterrows()):
        for i2, r2 in m2.iterrows():
            compared_matrix.ix[i1, i2] = function(r1, r2)

    if is_distance:
        print_log('Converting association to distance (1 - association) ...')
        compared_matrix = 1 - compared_matrix

    return compared_matrix


def consensus_cluster(clustering_labels):
    """
    Consenssu cluster `clustering_labels`, a distance matrix.
    :param clustering_labels: numpy array;
    :return: numpy array;
    """
    n_rows, n_cols = clustering_labels.shape
    consensus_clusterings = zeros((n_cols, n_cols))
    for i in range(n_cols):
        if i % 30 == 0:
            print_log('Consensus clustering ({}/{}) ...'.format(i, n_cols))
        for j in range(n_cols)[i:]:
            for r in range(n_rows):
                if clustering_labels[r, i] == clustering_labels[r, j]:
                    consensus_clusterings[i, j] += 1
    # Return normalized consensus clustering
    return consensus_clusterings / n_rows


def mds(dataframe, informational_mds=True, mds_seed=SEED, n_init=1000, max_iter=1000, standardize=True):
    """
    Multidimentional scale rows of `pandas_object` from <n_cols>D into 2D.
    :param dataframe: pandas DataFrame; (n_points, n_dimentions)
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param n_init: int;
    :param max_iter: int;
    :param standardize: bool;
    :return: pandas DataFrame; (n_points, [x, y])
    """
    if informational_mds:
        from .information import information_coefficient
        mds_obj = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(
            compare_matrices(dataframe, dataframe, information_coefficient, is_distance=True, axis=1))
    else:
        mds_obj = MDS(random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(dataframe)
    coordinates = DataFrame(coordinates, index=dataframe.index, columns=['x', 'y'])

    if standardize:
        coordinates = normalize_pandas_object(coordinates, method='0-1', axis=0)

    return coordinates


def fit_columns(dataframe, function_to_fit=exponential_function, maxfev=1000):
    """
    Fit columsn of `dataframe` to `function_to_fit`.
    :param dataframe: pandas DataFrame;
    :param function_to_fit: function;
    :param maxfev: int;
    :return: list; fit parameters
    """
    x = array(range(dataframe.shape[0]))
    y = asarray(dataframe.apply(sorted).apply(sum, axis=1)) / dataframe.shape[1]
    fit_parameters = curve_fit(function_to_fit, x, y, maxfev=maxfev)[0]
    return fit_parameters


def get_sample_coordinates_via_pulling(component_x_coordinates, component_x_samples,
                                       n_influencing_components='all', component_pulling_power=1):
    """
    Compute sample coordinates based on component coordinates, which pull samples.
    :param component_x_coordinates: pandas DataFrame; (n_points, [x, y])
    :param component_x_samples: pandas DataFrame; (n_points, n_samples)
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_pulling_power: str or number; power to raise components' influence on each sample
    :return: pandas DataFrame; (n_samples, [x, y])
    """
    sample_coordinates = DataFrame(index=component_x_samples.columns, columns=['x', 'y'])
    for sample in sample_coordinates.index:
        c = component_x_samples.ix[:, sample]
        if n_influencing_components == 'all':
            n_influencing_components = component_x_samples.shape[0]
        c = c.mask(c < c.sort_values()[-n_influencing_components], other=0)
        x = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'x']) / sum(c ** component_pulling_power)
        y = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'y']) / sum(c ** component_pulling_power)
        sample_coordinates.ix[sample, ['x', 'y']] = x, y
    return sample_coordinates
