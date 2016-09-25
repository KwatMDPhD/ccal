"""
Computational Cancer Analysis Library

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis Laboratory, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis Laboratory, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov
"""
# TODO: optimize return

from numpy import finfo

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
    Check if `libraries_needed` are installed; if not, then install using pip.
    :param libraries_needed: iterable; library names
    :return: None
    """
    from pip import get_installed_distributions, main

    print_log('Checking library dependencies ...')

    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]

    # If any of the `libraries_needed` is not in the currently installed libraries, then install it using pip
    for lib in libraries_needed:
        if lib not in libraries_installed:
            print_log('{} not found; installing it using pip ...'.format(lib))
            main(['install', lib])


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
    Get the current time.
    :param time_only: bool; exclude year, month, and date or not
    :return: str; the current time
    """
    from datetime import datetime

    if time_only:
        formatter = '%H%M%S'
    else:
        formatter = '%Y%m%d-%H%M%S'
    return datetime.now().strftime(formatter)


# =====================================================================================================================
# Operate on strings
# =====================================================================================================================
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
# Operate on files
# ======================================================================================================================
def establish_path(filepath):
    """
    If the path up to the deepest directory in `filepath` doesn't exist, make the path up to the deepest directory.
    :param filepath: str;
    :return: None
    """
    from os import mkdir
    from os.path import abspath, split, isdir, isfile, islink

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
    from pandas import read_csv

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
    df.column.name = column_name

    return df


def write_gct(pandas_object, filepath, descriptions=None):
    """
    Write a `pandas_object` to a `filepath` as a .gct.
    :param pandas_object: pandas DataFrame or Serires; (n_samples, m_features)
    :param filepath: str;
    :param descriptions: iterable; (n_rows of `pandas_object`); description column for the .gct
    :return: None
    """
    from pandas import Series, DataFrame

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
    obj.column.name = None

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
    :return: pandas DataFrame
    """
    # TODO: test

    from pandas import read_csv

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
    # TODO: test

    obj = pandas_object.copy()

    # Add description column if missing
    if obj.columns[0] != 'Description':
        if descriptions:
            obj.insert(0, 'Description', descriptions)
        else:
            obj.insert(0, 'Description', obj.index)

    # Set row and column name
    obj.index.name = 'Name'
    obj.column.name = None

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
# Simulate
# ======================================================================================================================
def make_random_dataframe_or_series(n_rows, n_cols, n_categories=None):
    """
    Simulate DataFrame (2D) or Series (1D).
    :param n_rows: int;
    :param n_cols: int;
    :param n_categories: None or int; continuous if None and categorical if int
    :return: pandas DataFrame or Series; (`n_rows`, `n_cols`) or (1, `n_cols`)
    """
    from numpy.random import random_integers, random_sample
    from pandas import DataFrame

    # Set up indices and column names
    indices = ['Feature {}'.format(i) for i in range(n_rows)]
    columns = ['Sample {}'.format(i) for i in range(n_cols)]

    # Set up data type: continuous, categorical, or binary
    if n_categories:
        features = DataFrame(random_integers(0, n_categories - 1, (n_rows, n_cols)), index=indices, columns=columns)
    else:
        features = DataFrame(random_sample((n_rows, n_cols)), index=indices, columns=columns)

    if n_rows == 1:  # Return series if there is only 1 row
        return features.iloc[0, :]
    else:  # Return dataframe if there is more than 1 row
        return features


# ======================================================================================================================
# Write equations
# ======================================================================================================================
def exponential_function(x, a, k, c):
    """
    Apply exponential function on `x`.
    :param x:
    :param a:
    :param k:
    :param c:
    :return:
    """
    from numpy import exp

    return a * exp(k * x) + c


# ======================================================================================================================#
# Compute
# ======================================================================================================================#
def information_coefficient(x, y, n_grids=25, jitter=1E-10):
    """
    Compute the information coefficient between `x` and `y`, which can be either continuous, categorical, or binary
    :param x: vector;
    :param y: vector;
    :param n_grids: int;
    :param jitter: number;
    :return: float;
    """
    # TODO: optimize import
    from numpy import asarray, sign, sum, sqrt, exp, log, isnan
    from numpy.random import random_sample
    from scipy.stats import pearsonr
    import rpy2.robjects as ro
    from rpy2.robjects.numpy2ri import numpy2ri
    from rpy2.robjects.packages import importr
    ro.conversion.py2ri = numpy2ri
    mass = importr('MASS')
    bcv = mass.bcv
    kde2d = mass.kde2d

    # Can't work with missing any value
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

    # Compute P(x, y), P(x), P(y)
    fxy = asarray(kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + EPS
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Compute mutual information;
    mi = sum(pxy * log(pxy / (asarray([px] * n_grids).T * asarray([py] * n_grids)))) * dx * dy

    # # Get H(x, y), H(x), and H(y)
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


# ======================================================================================================================#
# Work on array-like
# ======================================================================================================================#
# TODO: make sure the normalization when size == 0 or range == 0 is correct
def normalize_series(series, method='-0-', n_ranks=10000):
    """
    Normalize a pandas `series`.
    :param series: pandas Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: int;
    :return: pandas Series;
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
    Make a label-x-sample binary matrix from a Series.
    :param series: pandas Series;
    :return: pandas DataFrame;
    """
    from pandas import DataFrame

    # Make an empty DataFrame (n_unique_labels, n_samples)
    label_x_sample = DataFrame(index=sorted(set(series)), columns=series.index)

    # Binarize each unique label
    for i in label_x_sample.index:
        label_x_sample.ix[i, :] = (series == i).astype(int)

    return label_x_sample


# ======================================================================================================================#
# Work on matrix-like
# ======================================================================================================================#
def normalize_pandas_object(pandas_object, method, axis=None, n_ranks=10000):
    """
    Normalize a pandas object.
    :param pandas_object: pandas DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: int;
    :param axis: int; None for global, 0 for by-column, and 1 for by-row normalization
    :return: pandas DataFrame or Series;
    """
    from pandas import Series, DataFrame

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
                # TODO: implement
                raise ValueError('Normalizing combination of \'rank\' & axix=\'all\' has not been implemented yet.')


def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all `arrays`.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """
    from numpy import ones, isnan

    # Keep all column indices
    not_nan_filter = ones(len(arrays[0]), dtype=bool)

    # Keep column indices without missing value in all arrays
    for a in arrays:
        not_nan_filter &= ~isnan(a)

    return [a[not_nan_filter] for a in arrays]


def get_consensus(clustering_x_sample):
    """
    Count number of co-clusterings.
    :param clustering_x_sample: numpy array; (n_clusterings, n_samples)
    :return: numpy array; (n_samples, n_samples)
    """
    # TODO: enable flexible axis

    from numpy import zeros

    n_clusterings, n_samples = clustering_x_sample.shape

    # Make an empty co-occurence matrix (n_samples, n_samples)
    consensus_clusterings = zeros((n_samples, n_samples))

    # Count the number of co-occurences
    for i in range(n_samples):
        for j in range(n_samples):
            for c_i in range(n_clusterings):
                v1 = clustering_x_sample[c_i, i]
                v2 = clustering_x_sample[c_i, j]
                if v1 and v2 and (v1 == v2):
                    consensus_clusterings[i, j] += 1

    # Normalize by the number of clusterings and return
    return consensus_clusterings / n_clusterings


def mds(dataframe, distance_function=None, mds_seed=SEED, n_init=1000, max_iter=1000, standardize=True):
    """
    Multidimentional scale rows of `pandas_object` from <n_cols>D into 2D.
    :param dataframe: pandas DataFrame; (n_points, n_dimentions)
    :param distance_function: function; capable of computing the distance between 2 vectors
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param n_init: int;
    :param max_iter: int;
    :param standardize: bool;
    :return: pandas DataFrame; (n_points, [x, y])
    """
    from pandas import DataFrame
    from sklearn.manifold import MDS

    if distance_function:  # Use precomputed distances
        mds_obj = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(compare_matrices(dataframe, dataframe, distance_function,
                                                             is_distance=True, axis=1))
    else:  # Use Euclidean distances
        mds_obj = MDS(random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(dataframe)

    # Convert to DataFrame
    coordinates = DataFrame(coordinates, index=dataframe.index, columns=['x', 'y'])

    if standardize:  # Rescale coordinates between 0 and 1
        coordinates = normalize_pandas_object(coordinates, method='0-1', axis=0)

    return coordinates


# ======================================================================================================================
# Associate
# ======================================================================================================================
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
    from pandas import DataFrame

    # Copy and rotate matrices to make the comparison by row
    if axis == 1:
        m1 = matrix1.copy()
        m2 = matrix2.copy()
    else:
        m1 = matrix1.T
        m2 = matrix2.T

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


def compute_against_reference(features, target, function=information_coefficient, n_features=0.95, ascending=False,
                              n_samplings=30, confidence=0.95, n_perms=30):
    """
    Compute scores[i] = `features`[i] vs. `target` using `function`.
    Compute confidence interval (CI) for `n_features` features. And compute p-val and FDR (BH) for all features.
    :param features: pandas DataFrame; (n_features, n_samples); must have row and column indices
    :param target: pandas Series; (n_samples); must have name and indices, which must match `features`'s column index
    :param function: function; scoring function
    :param n_features: int or float; number threshold if >= 1, and percentile threshold if < 1
    :param ascending: bool; True if score increase from top to bottom, and False otherwise
    :param n_samplings: int; number of sampling for confidence interval bootstrapping; must be > 2 to compute
    :param confidence: float; fraction compute confidence interval
    :param n_perms: int; number of permutations for permutation test
    :return: pandas DataFrame (n_features, n_scores),
    """
    # TODO: refactor

    import math

    from numpy import array, empty
    from numpy.random import choice, shuffle
    from pandas import DataFrame, merge
    import scipy.stats as stats
    from statsmodels.sandbox.stats.multicomp import multipletests

    ### Compute scores: scores[i] = `features`[i] vs. `target`
    print_log('Computing scores ...')
    scores = features.apply(lambda row: function(row, target), axis=1)
    scores = DataFrame(scores, index=features.index, columns=['score'])

    ### Compute confidence interval using bootstrapped distribution
    print_log('Computing {} CI using a distribution created by {} bootstrapping ...'.format(confidence, n_samplings))

    n_samples = math.ceil(0.632 * features.shape[1])
    if n_samples < 3:  # Can't bootstrap only if there is less than 3 samples in 63% of the samples
        print_log('Can\'t bootstrap because 0.632 * n_samples < 3.')

    else:  # Compute confidence interval for limited features
        if n_features < 1:  # Limit using percentile
            above_quantile = scores.ix[:, 'score'] >= scores.ix[:, 'score'].quantile(n_features)
            print_log('Bootstrapping {} features (> {} percentile) ...'.format(sum(above_quantile), n_features))
            below_quantile = scores.ix[:, 'score'] <= scores.ix[:, 'score'].quantile(1 - n_features)
            print_log('Bootstrapping {} features (< {} percentile) ...'.format(sum(below_quantile), 1 - n_features))
            indices_to_bootstrap = scores.index[above_quantile | below_quantile].tolist()
        else:  # Limit using numbers
            indices_to_bootstrap = scores.index[:n_features].tolist() + scores.index[-n_features:].tolist()
            print_log('Bootstrapping top & bottom {} features ...'.format(n_features))

        # Bootstrap: for `n_sampling` times, randomly choose 63% of the samples, score, and create score distribution
        sampled_scores = DataFrame(index=indices_to_bootstrap, columns=range(n_samplings))
        for c_i in sampled_scores:
            # Randomize
            ramdom_samples = choice(features.columns.tolist(), int(n_samples)).tolist()
            sampled_features = features.ix[indices_to_bootstrap, ramdom_samples]
            sampled_target = target.ix[ramdom_samples]
            # Score
            sampled_scores.ix[:, c_i] = sampled_features.apply(lambda r: function(r, sampled_target), axis=1)

        # Compute the score confidence interval using bootstrapped score distribution
        z_critical = stats.norm.ppf(q=confidence)
        confidence_intervals = sampled_scores.apply(lambda r: z_critical * (r.std() / math.sqrt(n_samplings)), axis=1)
        confidence_intervals = DataFrame(confidence_intervals,
                                         index=indices_to_bootstrap, columns=['{} MoE'.format(confidence)])

        # Merge
        scores = merge(scores, confidence_intervals, how='outer', left_index=True, right_index='True')

    ### Compute P-values and FDRs
    p_values_and_fdrs = DataFrame(index=features.index, columns=['Local P-value', 'Global P-value', 'FDR'])

    # Compute scores using permuted target
    permutation_scores = empty((features.shape[0], n_perms))
    shuffled_target = array(target)
    for i in range(n_perms):
        shuffle(shuffled_target)
        permutation_scores[:, i] = features.apply(lambda r: function(r, shuffled_target), axis=1)

    # Compute local and global P-values
    all_permutation_scores = permutation_scores.flatten()
    for i, (idx, f) in enumerate(scores.iterrows()):
        # Compute local p-value
        local_pval = float(sum(permutation_scores[i, :] > float(f.ix['score'])) / n_perms)
        if not local_pval:
            local_pval = float(1 / n_perms)
        p_values_and_fdrs.ix[idx, 'Local P-value'] = local_pval

        # Compute global p-value
        global_pval = float(sum(all_permutation_scores > float(f.ix['score'])) / (n_perms * features.shape[0]))
        if not global_pval:
            global_pval = float(1 / (n_perms * features.shape[0]))
        p_values_and_fdrs.ix[idx, 'Global P-value'] = global_pval

    # Compute global permutation FDRs
    p_values_and_fdrs.ix[:, 'FDR'] = multipletests(p_values_and_fdrs.ix[:, 'Global P-value'], method='fdr_bh')[1]

    # Merge
    scores = merge(scores, p_values_and_fdrs, left_index=True, right_index=True)

    return scores.sort_values('score', ascending=ascending)


# ======================================================================================================================
# Cluster
# ======================================================================================================================
def consensus_cluster(matrix, ks, max_std=3, n_clusterings=50, filepath_prefix=None):
    """
    Consensus cluster `matrix`'s columns into k clusters.
    :param matrix: pandas DataFrame; (n_features, m_samples)
    :param ks: iterable; list of ks used for clustering
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consensus clustering
    :param filepath_prefix: str;
    :return: pandas DataFrame and Series; assignment matrix (n_ks, n_samples) and the cophenetic correlations (n_ks)
    """
    from numpy import empty
    from numpy.random import random_integers
    from pandas import DataFrame
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster, cophenet

    # '-0-' normalize by features and clip values `max_std` standard deviation away; then '0-1' normalize by features
    clipped_matrix = normalize_pandas_object(matrix, method='-0-', axis=1).clip(-max_std, max_std)
    normalized_matrix = normalize_pandas_object(clipped_matrix, method='0-1', axis=1)

    # Make sample-distance matrix
    print_log('Making sample-distance matrix ...')
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
        clustering_labels = empty((n_clusterings, matrix.shape[1]))
        for i in range(n_clusterings):
            if i % 10 == 0:
                print_log('\tPermuting distance matrix with repeat and clustering ({}/{}) ...'.format(i, n_clusterings))
            randomized_column_indices = random_integers(0, distance_matrix.shape[1] - 1, distance_matrix.shape[1])
            ward = AgglomerativeClustering(n_clusters=k)
            ward.fit(distance_matrix.iloc[randomized_column_indices, randomized_column_indices])

            # Assign labels to the samples selected by permutation with repeat
            clustering_labels[i, randomized_column_indices] = ward.labels_

        # Make co-assignment matrix using labels created by clusterings of permuted-distance matrix
        print_log('\tMaking Counting co-assignments during {} permuted-distance-matrix clusterings ...'.format(n_clusterings))
        coassignments = get_consensus(clustering_labels)

        # Convert co-assignments to distance
        distances = 1 - coassignments

        # Cluster distance matrix to assign the final label
        ward = linkage(distances, method='ward')
        consensus_clustering_labels.ix[k, :] = fcluster(ward, k, criterion='maxclust')

        # Compute clustering scores, the correlation between cophenetic and Euclidean distances
        cophenetic_correlations[k] = cophenet(ward, pdist(distances))[0]
        print_log('Computed cophenetic correlations.')

    # Save
    if filepath_prefix:
        establish_path(filepath_prefix)
        write_gct(consensus_clustering_labels, filepath_prefix + '_labels.gct')
        write_dictionary(cophenetic_correlations, filepath_prefix + '_clustering_scores.txt',
                         key_name='k', value_name='cophenetic_correlation')

    return consensus_clustering_labels, cophenetic_correlations


# ======================================================================================================================
# NMF
# ======================================================================================================================
def nmf_and_score(matrix, ks, method='cophenetic_correlation', n_clusterings=30,
                  init='random', solver='cd', tol=1e-6, max_iter=1000, random_state=SEED, alpha=0, l1_ratio=0,
                  shuffle_=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
    """
    Perform NMF with k from `ks` and score each NMF decomposition.
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
    :return: 2 dicts; {k: {W:w, H:h, ERROR:error}} and {k: score}
    """
    from numpy import empty, asarray, argmax
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, cophenet

    if isinstance(ks, int):
        ks = [ks]

    nmf_results = {}
    scores = {}

    if method == 'cophenetic_correlation':
        print_log('Scoring NMF with cophenetic correlation of consensus-clustering ({} clusterings) ...'.format(
            n_clusterings))
        for k in ks:
            print_log('k={} ...'.format(k))

            # NMF cluster `n_clustering` times
            clustering_labels = empty((n_clusterings, matrix.shape[1]), dtype=int)
            for i in range(n_clusterings):
                if i % 10 == 0:
                    print_log('\tNMF ({}/{}) ...'.format(i, n_clusterings))

                nmf_result = nmf(matrix, k,
                                 init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state,
                                 alpha=alpha, l1_ratio=l1_ratio, shuffle_=shuffle_, nls_max_iter=nls_max_iter,
                                 sparseness=sparseness, beta=beta, eta=eta)[k]

                # Save the first NMF decomposition for each k
                if i == 0:
                    nmf_results[k] = nmf_result
                    print_log('\t\tSaved the 1st NMF decomposition.')

                # Column labels are the row index holding the highest value
                clustering_labels[i, :] = argmax(asarray(nmf_result['H']), axis=0)

            # Consensus cluster `n_clustering` sets of NMF labels
            print_log('\tConsensus clustering {} sets of NMF labels ...'.format(n_clusterings))
            consensus_clusterings = get_consensus(clustering_labels)

            # Compute clustering scores, the correlation between cophenetic and Euclidian distances
            scores[k] = cophenet(linkage(consensus_clusterings, 'average'), pdist(consensus_clusterings))[0]
            print_log('\tComputed the cophenetic correlations.')

    else:
        raise ValueError('Unknown method {}.'.format(method))

    return nmf_results, scores


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
    :return: dict; {k: {W:w, H:h, ERROR:error}}
    """
    from pandas import DataFrame
    from sklearn.decomposition import NMF

    if isinstance(ks, int):
        ks = [ks]

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
