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

from pip import get_installed_distributions, main
from os import mkdir
from os.path import abspath, split, isdir, isfile, islink
from datetime import datetime
import math
from random import seed
from multiprocessing import Pool, cpu_count

from numpy import finfo, array, asarray, empty, zeros, ones, sign, sum, sqrt, exp, log, dot, isnan, argmax
from numpy.linalg import pinv
from numpy.random import random_sample, random_integers, shuffle, choice
from pandas import Series, DataFrame, concat, merge, read_csv
from scipy.stats import pearsonr, norm
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.optimize import curve_fit, nnls
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import bwr, Paired
from matplotlib.backends.backend_pdf import PdfPages
from seaborn import light_palette, heatmap, clustermap, pointplot

from . import VERBOSE, SEED

EPS = finfo(float).eps

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


# ======================================================================================================================
# System
# ======================================================================================================================
def install_libraries(libraries_needed):
    """
    Check if `libraries_needed` are installed; if not, then install using pip.
    :param libraries_needed: iterable; library names
    :return: None
    """

    print_log('Checking library dependencies ...')

    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]

    # If any of the `libraries_needed` is not in the currently installed libraries, then install it using pip
    for lib in libraries_needed:
        if lib not in libraries_installed:
            print_log('{} not found; installing it using pip ...'.format(lib))
            main(['install', lib])


def plant_seed(a_seed):
    """
    Set random seed.
    :param a_seed: int;
    :return: None
    """

    seed(a_seed)
    print_log('Planted a random seed {}.'.format(a_seed))


# ======================================================================================================================
# Parallel computing
# ======================================================================================================================
def parallelize(function, list_of_args, n_jobs=None):
    """
    Apply function on args with parallel computing using `n_jobs` jobs.
    :param function: function;
    :param list_of_args: list-like; function's args
    :param n_jobs: int; if not specified, parallelize to all CPUs
    :return: list; list of outputs returned by all jobs
    """

    # Use almost all available CPUs
    if not n_jobs:
        n_jobs = max(cpu_count() - 1, 1)

    # Parallelize
    with Pool(n_jobs) as p:
        return p.map(function, list_of_args)


# ======================================================================================================================
# File
# ======================================================================================================================
def establish_path(filepath):
    """
    If the path up to the deepest directory in `filepath` doesn't exist, make the path up to the deepest directory.
    :param filepath: str;
    :return: None
    """

    prefix, suffix = split(filepath)
    prefix = abspath(prefix)

    # Get missing directories
    missing_directories = []
    while not (isdir(prefix) or isfile(prefix) or islink(prefix)):
        missing_directories.append(prefix)
        prefix, suffix = split(prefix)

    # Make missing directories
    for d in reversed(missing_directories):
        mkdir(d)
        print_log('Created directory {}.'.format(d))


def read_gct(filepath, fill_na=None, drop_description=True, row_name=None, column_name=None):
    """
    Read a .gct (`filepath`) and convert it into a pandas DataFrame.
    :param filepath: str;
    :param fill_na: *; value to replace NaN in the DataFrame
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :param row_name: str;
    :param column_name: str;
    :return: pandas DataFrame; [n_samples, n_features (or n_features + 1 if not dropping the Description column)]
    """

    # Read .gct
    df = read_csv(filepath, skiprows=2, sep='\t')

    # Fix missing values
    if fill_na:
        df.fillna(fill_na, inplace=True)

    # Get 'Name' and 'Description' columns
    c1, c2 = df.columns[:2]

    # Check if the 1st column is 'Name'; if so set it as the index
    if c1 != 'Name':
        if c1.strip() != 'Name':
            raise ValueError('Column 1 != \'Name\'.')
        else:
            raise ValueError('Column 1 has more than 1 extra space around \'Name\'. Please strip it.')
    df.set_index('Name', inplace=True)

    # Check if the 2nd column is 'Description'; is so drop it as necessary
    if c2 != 'Description':
        if c2.strip() != 'Description':
            raise ValueError('Column 2 != \'Description\'')
        else:
            raise ValueError('Column 2 has more than 1 extra space around \'Description\'. Please strip it.')
    if drop_description:
        df.drop('Description', axis=1, inplace=True)

    # Set row and column name
    df.index.name = row_name
    df.columns.name = column_name

    return df


def write_gct(pandas_object, filepath, descriptions=None):
    """
    Write a `pandas_object` to a `filepath` as a .gct.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of `pandas_object`); description column for the .gct
    :return: None
    """

    # Copy
    obj = pandas_object.copy()

    # Work with only DataFrame
    if isinstance(obj, Series):
        obj = DataFrame(obj).T

    # Add description column if missing
    if obj.columns[0] != 'Description':
        if descriptions:
            obj.insert(0, 'Description', descriptions)
        else:
            obj.insert(0, 'Description', obj.index)

    # Set row and column name
    obj.index.name = 'Name'
    obj.columns.name = None

    # Save as .gct
    if not filepath.endswith('.gct'):
        filepath += '.gct'
    with open(filepath, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(obj.shape[0], obj.shape[1] - 1))
        obj.to_csv(f, sep='\t')


def read_gmt(filepath, drop_description=True):
    """
    Read a .gmt file.
    :param filepath:
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :return: pandas DataFrame; (n_gene_sets, n_genes_in_the_largest_gene_set)
    """

    # Read .gct
    df = read_csv(filepath, sep='\t')

    # Get 'Name' and 'Description' columns
    c1, c2 = df.columns[:2]

    # Check if the 1st column is 'Name'; if so set it as the index
    if c1 != 'Name':
        if c1.strip() != 'Name':
            raise ValueError('Column 1 != \'Name\'.')
        else:
            raise ValueError('Column 1 has more than 1 extra space around \'Name\'. Please strip it.')
    df.set_index('Name', inplace=True)

    # Check if the 2nd column is 'Description'; is so drop it as necessary
    if c2 != 'Description':
        if c2.strip() != 'Description':
            raise ValueError('Column 2 != \'Description\'')
        else:
            raise ValueError('Column 2 has more than 1 extra space around \'Description\'. Please strip it.')
    if drop_description:
        df.drop('Description', axis=1, inplace=True)

    # Set row name (column name is None when read)
    df.index.name = 'Gene Set'

    return df


def write_gmt(pandas_object, filepath, descriptions=None):
    """
    Write a `pandas_object` to a `filepath` as a .gmt.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of `pandas_object`); description column for the .gmt
    :return: None
    """

    obj = pandas_object.copy()

    # Add description column if missing
    if obj.columns[0] != 'Description':
        if descriptions:
            obj.insert(0, 'Description', descriptions)
        else:
            obj.insert(0, 'Description', obj.index)

    # Set row and column name
    obj.index.name = 'Name'
    obj.columns.name = None

    # Save as .gmt
    if not filepath.endswith('.gmt'):
        filepath += '.gmt'
    obj.to_csv(filepath, sep='\t')


def write_dictionary(dictionary, filepath, key_name, value_name):
    """
    Write a dictionary as a 2-column-tab-separated file.
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


# ======================================================================================================================
# Log
# ======================================================================================================================
# TODO: use logging (https://docs.python.org/3.5/howto/logging.html)
def print_log(string):
    """
    Print `string` together with logging information.
    :param string: str; message to printed
    :return: None
    """

    if VERBOSE:
        print('<{}> {}'.format(timestamp(time_only=True), string))


def timestamp(time_only=False):
    """
    Get the current time.
    :param time_only: bool; exclude year, month, and date or not
    :return: str; the current time
    """

    if time_only:
        formatter = '%H%M%S'
    else:
        formatter = '%Y%m%d-%H%M%S'
    return datetime.now().strftime(formatter)


# ======================================================================================================================
# String
# ======================================================================================================================
def title_string(string):
    """
    Title a string.
    :param string: str;
    :return: str;
    """

    string = string.title().replace('_', ' ').replace('\n', '')
    for article in ['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'of']:
        string = string.replace(' ' + article.title() + ' ', ' ' + article + ' ')
    return string


def untitle_string(string):
    """
    Untitle a string.
    :param string: str;
    :return: str;
    """

    return string.lower().replace(' ', '_')


# ======================================================================================================================
# Equation
# ======================================================================================================================
def exponential_function(x, a, k, c):
    """
    Apply exponential function on `_nmf_and_score`.
    :param x: array-like; independent variables
    :param a: number; parameter a
    :param k: number; parameter k
    :param c: number; parameter c
    :return: numpy array; (n_independent_variables)
    """

    return a * exp(k * x) + c


# ======================================================================================================================
# Array-like
# ======================================================================================================================
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


def explode(series):
    """
    Make a label-_nmf_and_score-sample binary matrix from a Series.
    :param series: pandas Series;
    :return: pandas DataFrame; (n_labels, n_samples)
    """

    # Make an empty DataFrame (n_unique_labels, n_samples)
    label_x_sample = DataFrame(index=sorted(set(series)), columns=series.index)

    # Binarize each unique label
    for i in label_x_sample.index:
        label_x_sample.ix[i, :] = (series == i).astype(int)

    return label_x_sample


# ======================================================================================================================
# Matrix-like
# ======================================================================================================================
def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all `arrays`.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """

    # Keep all column indices
    not_nan_filter = ones(len(arrays[0]), dtype=bool)

    # Keep column indices without missing value in all arrays
    for a in arrays:
        not_nan_filter &= ~isnan(a)

    return [a[not_nan_filter] for a in arrays]


def count_coclusterings(sample_x_clustering):
    """
    Count number of co-clusterings.
    :param sample_x_clustering: pandas DataFrame; (n_samples, n_clusterings)
    :return: pandas DataFrame; (n_samples, n_samples)
    """
    sample_x_clustering_array = asarray(sample_x_clustering)

    n_samples, n_clusterings = sample_x_clustering_array.shape

    # Make sample _nmf_and_score sample matrix
    coclusterings = zeros((n_samples, n_samples))

    # Count the number of co-clusterings
    # TODO: don't compare same pair twice
    for i in range(n_samples):
        for j in range(n_samples):
            for c_i in range(n_clusterings):
                v1 = sample_x_clustering_array[i, c_i]
                v2 = sample_x_clustering_array[j, c_i]
                if v1 and v2 and (v1 == v2):
                    coclusterings[i, j] += 1

    # Normalize by the number of clusterings and return
    coclusterings /= n_clusterings
    return DataFrame(coclusterings, index=sample_x_clustering.index, columns=sample_x_clustering.index)


def mds(dataframe, distance_function=None, mds_seed=SEED, n_init=1000, max_iter=1000, standardize=True):
    """
    Multidimentional scale rows of `pandas_object` from <n_cols>D into 2D.
    :param dataframe: pandas DataFrame; (n_points, n_dimentions)
    :param distance_function: function; capable of computing the distance between 2 vectors
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param n_init: int;
    :param max_iter: int;
    :param standardize: bool;
    :return: pandas DataFrame; (n_points, 2 ('_nmf_and_score', 'y'))
    """

    if distance_function:  # Use precomputed distances
        mds_obj = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(compare_matrices(dataframe, dataframe, distance_function,
                                                             is_distance=True, axis=1))
    else:  # Use Euclidean distances
        mds_obj = MDS(random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(dataframe)

    # Convert to DataFrame
    coordinates = DataFrame(coordinates, index=dataframe.index, columns=['_nmf_and_score', 'y'])

    if standardize:  # Rescale coordinates between 0 and 1
        coordinates = normalize_pandas_object(coordinates, method='0-1', axis=0)

    return coordinates


# ======================================================================================================================
# Normalization
# ======================================================================================================================
# TODO: make sure the normalization when size == 0 or range == 0 is correct
def normalize_pandas_object(pandas_object, method, axis=None, n_ranks=10000):
    """
    Normalize a pandas object.
    :param pandas_object: pandas DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :param axis: int; None for global, 0 for by-column, and 1 for by-row normalization
    :return: pandas DataFrame or Series; normalized DataFrame or Series
    """

    print_log('\'{}\' normalizing pandas object on axis={} ...'.format(method, axis))

    if isinstance(pandas_object, Series):  # Series
        return normalize_series(pandas_object, method=method, n_ranks=n_ranks)

    elif isinstance(pandas_object, DataFrame):  # DataFrame
        if axis == 0 or axis == 1:  # Normalize by axis (Series)
            return pandas_object.apply(normalize_series, **{'method': method, 'n_ranks': n_ranks}, axis=axis)

        else:  # Normalize globally
            if method == '-0-':
                obj_mean = pandas_object.values.mean()
                obj_std = pandas_object.values.std()
                if obj_std == 0:
                    print_log('Not \'-0-\' normalizing (standard deviation is 0), but \'/ size\' normalizing.')
                    return pandas_object / pandas_object.size
                else:
                    return (pandas_object - obj_mean) / obj_std

            elif method == '0-1':
                obj_min = pandas_object.values.min()
                obj_max = pandas_object.values.max()
                if obj_max - obj_min == 0:
                    print_log('Not \'0-1\' normalizing (data range is 0), but \'/ size\' normalizing.')
                    return pandas_object / pandas_object.size
                else:
                    return (pandas_object - obj_min) / (obj_max - obj_min)

            elif method == 'rank':
                # TODO: implement global rank normalization
                raise ValueError('Normalizing combination of \'rank\' & axis=\'all\' has not been implemented yet.')


def normalize_series(series, method='-0-', n_ranks=10000):
    """
    Normalize a pandas `series`.
    :param series: pandas Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :return: pandas Series; normalized Series
    """

    if method == '-0-':
        mean = series.mean()
        std = series.std()
        if std == 0:
            print_log('Not \'-0-\' normalizing (standard deviation is 0), but \'/ size\' normalizing.')
            return series / series.size
        else:
            return (series - mean) / std
    elif method == '0-1':
        series_min = series.min()
        series_max = series.max()
        if series_max - series_min == 0:
            print_log('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing.')
            return series / series.size
        else:
            return (series - series_min) / (series_max - series_min)
    elif method == 'rank':
        return series.rank() / series.size * n_ranks


# ======================================================================================================================
# Information theory
# ======================================================================================================================
def information_coefficient(x, y, n_grids=25, jitter=1E-10):
    """
    Compute the information coefficient between `_nmf_and_score` and `y`, which can be either continuous, categorical, or binary
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grid lines in a dimention when estimating bandwidths
    :param jitter: number;
    :return: float;
    """

    # Can't work with missing any value
    # not_nan_filter = ~isnan(_nmf_and_score)
    # not_nan_filter &= ~isnan(y)
    # _nmf_and_score = _nmf_and_score[not_nan_filter]
    # y = y[not_nan_filter]
    x, y = drop_nan_columns([x, y])

    # Need at least 3 values to compute bandwidth
    if len(x) < 3 or len(y) < 3:
        return 0

    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)

    # Add jitter
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Compute bandwidths
    cor, p = pearsonr(x, y)
    bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Compute P(_nmf_and_score, y), P(_nmf_and_score), P(y)
    fxy = asarray(kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + EPS
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Compute mutual information;
    mi = sum(pxy * log(pxy / (asarray([px] * n_grids).T * asarray([py] * n_grids)))) * dx * dy

    # # Get H(_nmf_and_score, y), H(_nmf_and_score), and H(y)
    # hxy = - sum(pxy * log(pxy)) * dx * dy
    # hx = -sum(px * log(px)) * dx
    # hy = -sum(py * log(py)) * dy
    # mi = hx + hy - hxy

    # Compute information coefficient
    ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic


# ======================================================================================================================
# Association
# ======================================================================================================================
def compute_score_and_pvalue(x, y, function=information_coefficient, n_permutations=100):
    """
    Compute `function`(`_nmf_and_score`, `y`) and p-value using permutation test.
    :param x: array-like;
    :param y: array-like;
    :param function: function;
    :param n_permutations: int; number of permutations for the p-value permutation test
    :return: float and float; score and p-value
    """

    # Compute score
    score = function(x, y)

    # Compute scores against permuted target
    # TODO: decide which of _nmf_and_score and y is the target
    permutation_scores = empty(n_permutations)
    shuffled_target = array(y)
    for p in range(n_permutations):
        shuffle(shuffled_target)
        permutation_scores[p] = function(x, shuffled_target)

    # Compute p-value
    p_val = sum(permutation_scores > score) / n_permutations
    return score, p_val


def compare_matrices(dataframe1, dataframe2, function, axis=0, is_distance=False):
    """
    Make association or distance matrix of `dataframe1` and `dataframe2` by row (`axis=1`) or by column (`axis=0`).
    :param dataframe1: pandas DataFrame;
    :param dataframe2: pandas DataFrame;
    :param function: function; function used to compute association or dissociation
    :param axis: int; 0 for row-wise and 1 column-wise comparison
    :param is_distance: bool; True for distance and False for association
    :return: pandas DataFrame; (n, n); association or distance matrix
    """

    # Copy and rotate matrices to make the comparison by row
    if axis == 1:
        m1 = dataframe1.copy()
        m2 = dataframe2.copy()
    else:
        m1 = dataframe1.T
        m2 = dataframe2.T

    # Compare
    compared_matrix = DataFrame(index=m1.index, columns=m2.index, dtype=float)
    n = m1.shape[0]
    for i, (i1, r1) in enumerate(m1.iterrows()):
        if i % 100 == 0:
            print_log('Comparing {} ({}/{}) ...'.format(i1, i, n))
        for i2, r2 in m2.iterrows():
            compared_matrix.ix[i1, i2] = function(r1, r2)

    if is_distance:  # Convert associations to distances
        print_log('Converting association to distance (1 - association) ...')
        compared_matrix = 1 - compared_matrix

    return compared_matrix


def fit_matrix(matrix, function_to_fit, axis=0, maxfev=1000):
    """
    Fit rows or columns of `matrix` to `function_to_fit`.
    :param matrix: pandas DataFrame;
    :param function_to_fit: function;
    :param axis: int;
    :param maxfev: int;
    :return: list; fit parameters
    """

    if axis == 1:
        matrix = matrix.T

    x = array(range(matrix.shape[0]))
    y = asarray(matrix.apply(sorted).apply(sum, axis=1)) / matrix.shape[1]
    fit_parameters = curve_fit(function_to_fit, x, y, maxfev=maxfev)[0]
    return fit_parameters


def compute_against_target(features, target, function=information_coefficient, n_features=0.95, ascending=False,
                           n_jobs=1, min_n_per_job=100, n_samplings=30, confidence=0.95, n_permutations=30):
    """
    Compute: ith _score_dataframe_against_series = function(ith `feature`, `target`).
    Compute confidence interval (CI) for `n_features` features. And compute p-val and FDR (BH) for all features.
    :param features: pandas DataFrame; (n_features, n_samples); must have row and column indices
    :param target: pandas Series; (n_samples); must have name and indices, which must match `features`'s column index
    :param function: function; scoring function
    :param n_features: int or float; number of features to compute confidence interval and plot;
                        number threshold if >= 1, percentile threshold if < 1, and don't compute if None
    :param ascending: bool; True if score increase from top to bottom, and False otherwise
    :param n_jobs: int; number of jobs to parallelize
    :param min_n_per_job: int; minimum number of n per job for parallel computing
    :param n_samplings: int; number of bootstrap samplings to build distribution to get CI; must be > 2 to compute CI
    :param confidence: float; fraction compute confidence interval
    :param n_permutations: int; number of permutations for permutation test to compute P-val and FDR
    :return: pandas DataFrame; (n_features, 7 ('Score', 'MoE', 'P-value', 'FDR (forward)', 'FDR (reverse)', and 'FDR'))
    """

    #
    # Compute scores: scores[i] = `features`[i] vs. `target`
    #
    if n_jobs == 1:  # Non-parallel computing
        print_log('Scoring (without parallelizing) ...')
        scores = _score_dataframe_against_series((features, target, function))

    else:  # Parallel computing
        print_log('Scoring across {} parallelized jobs ...'.format(n_jobs))

        # Compute n for a job
        n_per_job = features.shape[0] // n_jobs

        if n_per_job < min_n_per_job:  # n is not enough for parallel computing
            print_log('\tNot parallelizing because n_per_job < {}.'.format(min_n_per_job))
            scores = _score_dataframe_against_series((features, target, function))

        else:  # n is enough for parallel computing
            # Group
            args = []
            leftovers = list(features.index)
            for i in range(n_jobs):
                split_features = features.iloc[i * n_per_job: (i + 1) * n_per_job, :]
                args.append((split_features, target, function))

                # Remove scored features
                for feature in split_features.index:
                    leftovers.remove(feature)

            # Parallelize
            scores = concat(parallelize(_score_dataframe_against_series, args, n_jobs=n_jobs))

            # Score leftovers
            if leftovers:
                print_log('Scoring leftovers: {} ...'.format(leftovers))
                scores = concat(
                    [scores, _score_dataframe_against_series((features.ix[leftovers, :], target, function))])
    scores.sort_values('Score', inplace=True)

    #
    #  Compute confidence interval using bootstrapped distribution
    #
    if not (isinstance(n_features, int) or isinstance(n_features, float)):
        print_log('Not computing confidence interval.')

    else:
        print_log('Computing {} CI using distributions built by {} bootstraps ...'.format(confidence, n_samplings))

        n_samples = math.ceil(0.632 * features.shape[1])
        if n_samples < 3:  # Can't bootstrap only if there is less than 3 samples in 63% of the samples
            print_log('Can\'t bootstrap because 0.632 * n_samples < 3.')

        else:  # Compute confidence interval for limited features
            if n_features < 1:  # Limit using percentile
                above_quantile = scores.ix[:, 'Score'] >= scores.ix[:, 'Score'].quantile(n_features)
                print_log('Bootstrapping {} features (> {} percentile) ...'.format(sum(above_quantile), n_features))
                below_quantile = scores.ix[:, 'Score'] <= scores.ix[:, 'Score'].quantile(1 - n_features)
                print_log('Bootstrapping {} features (< {} percentile) ...'.format(sum(below_quantile), 1 - n_features))
                indices_to_bootstrap = scores.index[above_quantile | below_quantile].tolist()
            else:  # Limit using numbers
                if 2 * n_features >= scores.shape[0]:
                    indices_to_bootstrap = scores.index
                    print_log('Bootstrapping all {} features ...'.format(scores.shape[0]))
                else:
                    indices_to_bootstrap = scores.index[:n_features].tolist() + scores.index[-n_features:].tolist()
                    print_log('Bootstrapping top & bottom {} features ...'.format(n_features))

            # Bootstrap: for `n_sampling` times, randomly choose 63% of the samples, _score_dataframe_against_series, and build _score_dataframe_against_series distribution
            sampled_scores = DataFrame(index=indices_to_bootstrap, columns=range(n_samplings))
            for c_i in sampled_scores:
                # Randomize
                ramdom_samples = choice(features.columns.tolist(), int(n_samples)).tolist()
                sampled_features = features.ix[indices_to_bootstrap, ramdom_samples]
                sampled_target = target.ix[ramdom_samples]
                # Score
                sampled_scores.ix[:, c_i] = sampled_features.apply(lambda r: function(r, sampled_target), axis=1)

            # Compute confidence interval for _score_dataframe_against_series using bootstrapped _score_dataframe_against_series distribution
            # TODO: improve confidence interval calculation
            z_critical = norm.ppf(q=confidence)
            confidence_intervals = sampled_scores.apply(lambda r: z_critical * (r.std() / math.sqrt(n_samplings)),
                                                        axis=1)
            confidence_intervals = DataFrame(confidence_intervals,
                                             index=indices_to_bootstrap, columns=['{} MoE'.format(confidence)])

            # Merge
            scores = merge(scores, confidence_intervals, how='outer', left_index=True, right_index='True')

    #
    # Compute P-values and FDRs by sores against permuted target
    #
    p_values_and_fdrs = DataFrame(index=features.index,
                                  columns=['P-value', 'FDR (forward)', 'FDR (reverse)', 'FDR'])
    print_log('Computing P-value and FDR using {} permutation test ...'.format(n_permutations))

    if n_jobs == 1:  # Non-parallel computing
        print_log('Scoring against permuted target (without parallelizing) ...')
        permutation_scores = _score_dataframe_against_permuted_series((features, target, function, n_permutations))

    else:  # Parallel computing
        print_log('Scoring against permuted target across {} parallelized jobs ...'.format(n_jobs))

        # Compute n for a job
        n_per_job = features.shape[0] // n_jobs

        if n_per_job < min_n_per_job:  # n is not enough for parallel computing
            print_log('\tNot parallelizing because n_per_job < {}.'.format(min_n_per_job))
            permutation_scores = _score_dataframe_against_permuted_series((features, target, function, n_permutations))

        else:  # n is enough for parallel computing
            # Group
            args = []
            leftovers = list(features.index)
            for i in range(n_jobs):
                split_features = features.iloc[i * n_per_job: (i + 1) * n_per_job, :]
                args.append((split_features, target, function, n_permutations))

                # Remove scored features
                for feature in split_features.index:
                    leftovers.remove(feature)

            # Parallelize
            permutation_scores = concat(parallelize(_score_dataframe_against_permuted_series, args, n_jobs=n_jobs))

            # Handle leftovers
            if leftovers:
                print_log('Scoring against permuted target using leftovers: {} ...'.format(leftovers))
                permutation_scores = concat(
                    [permutation_scores,
                     _score_dataframe_against_permuted_series(
                         (features.ix[leftovers, :], target, function, n_permutations))])

    # Compute local and global P-values
    print_log('Computing P-value and FDR ...')
    all_permutation_scores = permutation_scores.values.flatten()
    for i, (r_i, r) in enumerate(scores.iterrows()):
        # Compute global p-value
        p_value = float(sum(all_permutation_scores > float(r.ix['Score'])) / (n_permutations * features.shape[0]))
        if not p_value:
            p_value = float(1 / (n_permutations * features.shape[0]))
        p_values_and_fdrs.ix[r_i, 'P-value'] = p_value

    # Compute global permutation FDRs
    p_values_and_fdrs.ix[:, 'FDR (forward)'] = multipletests(p_values_and_fdrs.ix[:, 'P-value'], method='fdr_bh')[1]
    p_values_and_fdrs.ix[:, 'FDR (reverse)'] = multipletests(1 - p_values_and_fdrs.ix[:, 'P-value'], method='fdr_bh')[1]
    p_values_and_fdrs.ix[:, 'FDR'] = p_values_and_fdrs.ix[:, ['FDR (forward)', 'FDR (reverse)']].min(axis=1)

    # Merge
    scores = merge(scores, p_values_and_fdrs, left_index=True, right_index=True)

    return scores.sort_values('Score', ascending=ascending)


def _score_dataframe_against_series(args):
    """
    Compute: ith _score_dataframe_against_series = function(ith `feature`, `target`).
    :param args: list-like;
        (DataFrame (n_features, m_samples); features, Series (m_samples); target, function)
    :return: pandas DataFrame; (n_features, 1 ('Score'))
    """

    df, s, func = args
    return DataFrame(df.apply(lambda r: func(s, r), axis=1), index=df.index, columns=['Score'])


def _score_dataframe_against_permuted_series(args):
    """
    Compute: ith _score_dataframe_against_series = function(ith `feature`, permuted `target`) for n_permutations times.
    :param args: list-like;
        (DataFrame (n_features, m_samples); features, Series (m_samples); target, function, int; n_permutations)
    :return: pandas DataFrame; (n_features, n_permutations)
    """

    df, s, func, n_perms = args

    scores = DataFrame(index=df.index, columns=range(n_perms))
    shuffled_target = array(s)
    for p in range(n_perms):
        print_log('\tScoring against permuted target ({}/{}) ...'.format(p, n_perms))
        shuffle(shuffled_target)
        scores.iloc[:, p] = df.apply(lambda r: func(r, shuffled_target), axis=1)
    return scores


# ======================================================================================================================
# Cluster
# ======================================================================================================================
def consensus_cluster(matrix, ks, max_std=3, n_clusterings=50):
    """
    Consensus cluster `matrix`'s columns into k clusters.
    :param matrix: pandas DataFrame; (n_features, m_samples)
    :param ks: iterable; list of ks used for clustering
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consensus clustering
    :return: pandas DataFrame and Series; assignment matrix (n_ks, n_samples) and the cophenetic correlations (n_ks)
    """

    # '-0-' normalize by features and clip values `max_std` standard deviation away; then '0-1' normalize by features
    clipped_matrix = normalize_pandas_object(matrix, method='-0-', axis=1).clip(-max_std, max_std)
    normalized_matrix = normalize_pandas_object(clipped_matrix, method='0-1', axis=1)

    # Make sample-distance matrix
    print_log('Computing distances between samples ...')
    distance_matrix = compare_matrices(normalized_matrix, normalized_matrix, information_coefficient, is_distance=True)

    # Consensus cluster distance matrix
    print_log('Consensus clustering with {} clusterings ...'.format(n_clusterings))
    consensus_clustering_labels = DataFrame(index=ks, columns=list(matrix.columns))
    consensus_clustering_labels.index.name = 'k'
    cophenetic_correlations = {}

    if isinstance(ks, int):
        ks = [ks]
    for k in ks:
        print_log('k={} ...'.format(k))

        # For `n_clusterings` times, permute distance matrix with repeat, and cluster

        # Make sample _nmf_and_score clustering matrix
        sample_x_clustering = DataFrame(index=matrix.columns, columns=range(n_clusterings), dtype=int)
        for i in range(n_clusterings):
            if i % 10 == 0:
                print_log('\tPermuting distance matrix with repeat and clustering ({}/{}) ...'.format(i, n_clusterings))

            # Randomize samples with repeat
            random_indices = random_integers(0, distance_matrix.shape[0] - 1, distance_matrix.shape[0])

            # Cluster random samples
            ward = AgglomerativeClustering(n_clusters=k)
            ward.fit(distance_matrix.iloc[random_indices, random_indices])

            # Assign cluster labels to the random samples
            sample_x_clustering.iloc[random_indices, i] = ward.labels_

        # Make co-clustering matrix using labels created by clusterings of randomized distance matrix
        print_log('\tCounting co-clusterings of {} randomized distance matrix ...'.format(n_clusterings))
        coclusterings = count_coclusterings(sample_x_clustering)

        # Convert co-clustering matrix into distance matrix
        distances = 1 - coclusterings

        # Cluster distance matrix to assign the final label
        ward = linkage(distances, method='ward')
        consensus_clustering_labels.ix[k, :] = fcluster(ward, k, criterion='maxclust')

        # Compute clustering scores, the correlation between cophenetic and Euclidean distances
        cophenetic_correlations[k] = cophenet(ward, pdist(distances))[0]
        print_log('Computed cophenetic correlations.')

    return consensus_clustering_labels, cophenetic_correlations


# ======================================================================================================================
# NMF
# ======================================================================================================================
def nmf_and_score(matrix, ks, method='cophenetic_correlation', n_clusterings=30,
                  init='random', solver='cd', tol=1e-6, max_iter=1000, random_state=SEED, alpha=0, l1_ratio=0,
                  shuffle_=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
    """
    Perform NMF with k from `ks` and _score_dataframe_against_series each NMF decomposition.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param method: str; {'cophenetic_correlation'}
    :param n_clusterings:
    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_state:
    :param alpha:
    :param l1_ratio:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:
    :return: dict and dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}} and {k: cophenetic score}
    """

    if isinstance(ks, int):
        ks = [ks]
    else:
        ks = list(set(ks))

    nmf_results = {}
    nmf_scores = {}

    if method == 'cophenetic_correlation':
        print_log('Scoring NMF with cophenetic correlation from consensus-clustering ({} clusterings) ...'.format(
            n_clusterings))

        if len(ks) > 1:
            print_log('Parallelizing ...')
            args = [[matrix, k, n_clusterings, init, solver, tol, max_iter, random_state, alpha, l1_ratio, shuffle_,
                     nls_max_iter, sparseness, beta, eta] for k in ks]

            for nmf_result, nmf_score in parallelize(_nmf_and_score, args):
                nmf_results.update(nmf_result)
                nmf_scores.update(nmf_score)
        else:
            print_log('Not parallelizing ...')
            nmf_result, nmf_score = _nmf_and_score([matrix, ks[0], n_clusterings, init, solver, tol, max_iter,
                                                    random_state, alpha, l1_ratio, shuffle_, nls_max_iter, sparseness,
                                                    beta, eta])
            nmf_results.update(nmf_result)
            nmf_scores.update(nmf_score)

    else:
        raise ValueError('Unknown method {}.'.format(method))

    return nmf_results, nmf_scores


def _nmf_and_score(args):
    """
    NMF and score for a k.
    :param args:
    :return:
    """

    matrix, k, n_clusterings, init, solver, tol, max_iter, random_state, alpha, l1_ratio, shuffle_, nls_max_iter, sparseness, beta, eta = args

    print_log('NMF and scoring k={} ...'.format(k))

    nmf_result_dict = {}
    nmf_score_dict = {}

    # NMF cluster `n_clustering`
    sample_x_clustering = DataFrame(index=matrix.columns, columns=range(n_clusterings), dtype=int)
    for i in range(n_clusterings):
        if i % 10 == 0:
            print_log('\t(k={}) NMF ({}/{}) ...'.format(k, i, n_clusterings))

        # NMF
        nmf_result = nmf(matrix, k,
                         init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state,
                         alpha=alpha, l1_ratio=l1_ratio, shuffle_=shuffle_, nls_max_iter=nls_max_iter,
                         sparseness=sparseness, beta=beta, eta=eta)[k]

        # Save the first NMF decomposition for each k
        if i == 0:
            nmf_result_dict[k] = nmf_result
            print_log('\t\t(k={}) Saved the 1st NMF decomposition.'.format(k))

        # Column labels are the row index holding the highest value
        sample_x_clustering.iloc[:, i] = argmax(asarray(nmf_result['H']), axis=0)

    # Make co-clustering matrix using NMF labels
    print_log('\t(k={}) Counting co-clusterings of {} NMF ...'.format(k, n_clusterings))
    consensus_clusterings = count_coclusterings(sample_x_clustering)

    # Compute clustering scores, the correlation between cophenetic and Euclidian distances
    nmf_score_dict[k] = cophenet(linkage(consensus_clusterings, 'average'), pdist(consensus_clusterings))[0]
    print_log('\t(k={}) Computed the cophenetic correlations.'.format(k))

    return nmf_result_dict, nmf_score_dict


def nmf(matrix, ks, init='random', solver='cd', tol=1e-6, max_iter=1000, random_state=SEED,
        alpha=0, l1_ratio=0, shuffle_=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_state:
    :param alpha:
    :param l1_ratio:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:
    :return: dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}}
    """

    if isinstance(ks, int):
        ks = [ks]
    else:
        ks = list(set(ks))

    nmf_results = {}
    for k in ks:

        # Compute W, H, and reconstruction error
        model = NMF(n_components=k, init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state,
                    alpha=alpha, l1_ratio=l1_ratio, shuffle=shuffle_, nls_max_iter=nls_max_iter, sparseness=sparseness,
                    beta=beta, eta=eta)
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_

        # Return pandas DataFrame if the input matrix is also a DataFrame
        if isinstance(matrix, DataFrame):
            w = DataFrame(w, index=matrix.index, columns=[i + 1 for i in range(k)])
            h = DataFrame(h, index=[i + 1 for i in range(k)], columns=matrix.columns)

        # Save NMF results
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

    return nmf_results


def nnls_matrix(a, b, method='nnls'):
    """
    Solve a * x = b of (n, k) * (k, m) = (n, m).
    :param a: numpy array; (n, k)
    :param b: numpy array; (n, m)
    :param method: str; {'nnls', 'pinv'}
    :return: numpy array; (k, m)
    """
    if method == 'nnls':
        x = DataFrame(index=a.columns, columns=b.columns)
        for i in range(b.shape[1]):
            x.iloc[:, i] = nnls(a, b.iloc[:, i])[0]
    elif method == 'pinv':
        a_pinv = pinv(a)
        x = dot(a_pinv, b)
        x[x < 0] = 0
        x = DataFrame(x, index=a.columns, columns=b.columns)
    else:
        raise ValueError('Unknown method {}. Choose from [\'nnls\', \'pinv\']'.format(method))
    return x


# ======================================================================================================================
# Simulation
# ======================================================================================================================
def simulate_dataframe_or_series(n_rows, n_cols, n_categories=None):
    """
    Simulate DataFrame (2D) or Series (1D).
    :param n_rows: int;
    :param n_cols: int;
    :param n_categories: None or int; continuous if None and categorical if int
    :return: pandas DataFrame or Series; (`n_rows`, `n_cols`) or (1, `n_cols`)
    """

    # Set up indices and column names
    indices = ['Feature {}'.format(i) for i in range(n_rows)]
    columns = ['Sample {}'.format(i) for i in range(n_cols)]

    # Set up data type: continuous, categorical, or binary
    if n_categories:
        features = DataFrame(random_integers(0, n_categories - 1, (n_rows, n_cols)), index=indices, columns=columns)
    else:
        features = DataFrame(random_sample((n_rows, n_cols)), index=indices, columns=columns)

    if n_rows == 1:  # Return as series if there is only 1 row
        return features.iloc[0, :]
    else:
        return features


# ======================================================================================================================
# Plotting
# ======================================================================================================================
# Color maps
BAD_COLOR = 'wheat'
CMAP_CONTINUOUS = bwr
CMAP_CONTINUOUS.set_bad(BAD_COLOR)
CMAP_CATEGORICAL = Paired
CMAP_CATEGORICAL.set_bad(BAD_COLOR)
CMAP_BINARY = light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad(BAD_COLOR)

# Figure parameters
FIGURE_SIZE = (16, 10)
DPI = 1000


def plot_clustermap(dataframe, figure_size=FIGURE_SIZE, title=None, title_fontsize=20,
                    xticklabels=True, yticklabels=True, xticklabels_rotation=90, yticklabels_rotation=0,
                    row_colors=None, col_colors=None, dpi=DPI, filepath=None):
    """
    Plot heatmap for `dataframe`.
    :param dataframe: pandas DataFrame;
    :param figure_size: tuple; (n_rows, n_cols)
    :param title: str;
    :param title_fontsize: number;
    :param xticklabels: bool;
    :param yticklabels: bool;
    :param xticklabels_rotation: number;
    :param yticklabels_rotation: number;
    :param row_colors: list-like or pandas DataFrame/Series; List of colors to label for either the rows.
        Useful to evaluate whether samples within a group are clustered together. Can use nested lists or DataFrame for
        multiple color levels of labeling. If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param col_colors: list-like or pandas DataFrame/Series; List of colors to label for either the column.
        Useful to evaluate whether samples within a group are clustered together. Can use nested lists or DataFrame for
        multiple color levels of labeling. If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param dpi: int;
    :param filepath: str;
    :return: None
    """

    plt.figure(figsize=figure_size)

    clustergrid = clustermap(dataframe, xticklabels=xticklabels, yticklabels=yticklabels,
                             row_colors=row_colors, col_colors=col_colors, cmap=CMAP_CONTINUOUS)

    if title:
        figuretitle_font_properties = {'fontsize': title_fontsize, 'fontweight': 'bold'}
        plt.suptitle(title, **figuretitle_font_properties)

    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_rotation(xticklabels_rotation)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_rotation(yticklabels_rotation)

    if filepath:
        _save_plot(filepath, dpi=dpi)


def plot_clusterings(dataframe, figure_size=FIGURE_SIZE, title='Clustering Labels', title_fontsize=20, dpi=DPI,
                     filepath=None):
    """
    Plot clustering matrix.
    :param dataframe: pandas DataFrame; (n_clusterings, n_samples)
    :param figure_size: tuple; (n_rows, n_cols)
    :param title: str;
    :param title_fontsize: number;
    :param dpi: int;
    :param filepath: str;
    :return: None
    """

    a = asarray(dataframe)
    a.sort()

    plt.figure(figsize=figure_size)
    heatmap(DataFrame(a, index=dataframe.index), cmap=CMAP_CATEGORICAL, xticklabels=False,
            cbar_kws={'orientation': 'horizontal'})

    if title:
        figuretitle_font_properties = {'fontsize': title_fontsize}
        plt.suptitle(title, **figuretitle_font_properties)

    plt.gca().set_xlabel('Sample')

    for t in plt.gca().get_yticklabels():
        t.set_weight('bold')

    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks(list(range(1, a.max() + 1)))

    if filepath:
        _save_plot(filepath, dpi=dpi)


def plot_nmf_result(nmf_results=None, k=None, w_matrix=None, h_matrix=None, normalize=False, max_std=3,
                    figure_size=FIGURE_SIZE, title=None, title_fontsize=20, dpi=DPI, filepath=None):
    """
    Plot `nmf_results` dictionary (can be generated by `ccal.analyze.nmf` function).
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param k: int; k for NMF
    :param w_matrix: pandas DataFrame
    :param h_matrix: Pandas DataFrame
    :param normalize: bool; normalize W and H matrices or not ('-0-' normalization on the component axis)
    :param max_std: number; threshold to clip standardized values
    :param figure_size: tuple; (n_rows, n_cols)
    :param title: str;
    :param title_fontsize: number;
    :param dpi: int;
    :param filepath: str;
    :return: None
    """
    # Check for W and H matrix
    if isinstance(nmf_results, dict) and k:
        w_matrix = nmf_results[k]['W']
        h_matrix = nmf_results[k]['H']
    elif not (isinstance(w_matrix, DataFrame) and isinstance(h_matrix, DataFrame)):
        raise ValueError('Need either: 1) NMF result ({k: {W:w, H:h, ERROR:error}) and k; or 2) W and H matrices.')

    # Initialize a PDF
    if filepath:
        establish_path(filepath)
        if not filepath.endswith('.pdf'):
            filepath += '.pdf'
        pdf = PdfPages(filepath)

    # Set up aesthetic properties
    figuretitle_font_properties = {'fontsize': title_fontsize, 'fontweight': 'bold'}
    axtitle_font_properties = {'fontsize': title_fontsize * 0.9, 'fontweight': 'bold'}
    label_font_properties = {'fontsize': title_fontsize * 0.81, 'fontweight': 'bold'}

    # Plot W and H
    plt.figure(figsize=figure_size)
    gridspec = GridSpec(10, 16)
    ax_w = plt.subplot(gridspec[1:, :5])
    ax_h = plt.subplot(gridspec[3:8, 7:])
    if not title:
        title = 'NMF Result for k={}'.format(w_matrix.shape[1])
    plt.suptitle(title, **figuretitle_font_properties)
    # Plot W
    if normalize:
        w_matrix = normalize_pandas_object(w_matrix, method='-0-', axis=0).clip(-max_std, max_std)
    heatmap(w_matrix, cmap=CMAP_CONTINUOUS, yticklabels=False, ax=ax_w)
    ax_w.set_title('W Matrix', **axtitle_font_properties)
    ax_w.set_xlabel('Component', **label_font_properties)
    ax_w.set_ylabel('Feature', **label_font_properties)
    # Plot H
    if normalize:
        h_matrix = normalize_pandas_object(h_matrix, method='-0-', axis=1).clip(-max_std, max_std)
    heatmap(h_matrix, cmap=CMAP_CONTINUOUS, xticklabels=False, cbar_kws={'orientation': 'horizontal'}, ax=ax_h)
    ax_h.set_title('H Matrix', **axtitle_font_properties)
    ax_h.set_xlabel('Sample', **label_font_properties)
    ax_h.set_ylabel('Component', **label_font_properties)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=dpi, bbox_inches='tight')

    # Plot cluster map for W
    clustergrid = clustermap(w_matrix, standard_scale=0, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('W Matrix for k={}'.format(w_matrix.shape[1]), **figuretitle_font_properties)
    clustergrid.ax_heatmap.set_xlabel('Component', **label_font_properties)
    clustergrid.ax_heatmap.set_ylabel('Feature', **label_font_properties)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_fontweight('bold')
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_visible(False)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=dpi, bbox_inches='tight')

    # Plot cluster map for H
    clustergrid = clustermap(h_matrix, standard_scale=1, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('H Matrix for k={}'.format(w_matrix.shape[1]), **figuretitle_font_properties)
    clustergrid.ax_heatmap.set_xlabel('Sample', **label_font_properties)
    clustergrid.ax_heatmap.set_ylabel('Component', **label_font_properties)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_visible(False)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_fontweight('bold')
        t.set_rotation(0)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=dpi, bbox_inches='tight')

    if filepath:
        pdf.close()


def plot_clustering_scores(scores, figure_size=FIGURE_SIZE, title='Clustering Score vs. k', title_fontsize=20, dpi=DPI,
                           filepath=None):
    """
    Plot `scores` dictionary.
    :param scores: dict or pandas DataFrame; {k: _score_dataframe_against_series}
    :param figure_size: tuple;
    :param title: str;
    :param title_fontsize: number;
    :param dpi: int;
    :param filepath: str;
    :return: None
    """

    if isinstance(scores, DataFrame):
        scores = scores.to_dict()
        scores = scores.popitem()[1]

    plt.figure(figsize=figure_size)

    if title:
        plt.suptitle(title, fontsize=title_fontsize, fontweight='bold')

    pointplot(x=[k for k, v in scores.items()], y=[v for k, v in scores.items()])

    label_font_properties = {'fontsize': title_fontsize * 0.81, 'fontweight': 'bold'}
    plt.gca().set_xlabel('k', **label_font_properties)
    plt.gca().set_ylabel('Score', **label_font_properties)

    if filepath:
        _save_plot(filepath, dpi=dpi)


def _save_plot(filepath, dpi=DPI):
    """
    Establish filepath and save plot with resolution of dpi.
    :param filepath: str;
    :param dpi: int;
    :return: None
    """

    establish_path(filepath)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
