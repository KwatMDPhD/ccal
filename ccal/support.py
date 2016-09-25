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


def read_gct(filepath, fill_na=None, drop_description=True):
    """
    Read a .gct (`filepath`) and convert it into a pandas DataFrame.
    :param filepath: str;
    :param fill_na: *; value to replace NaN in the DataFrame
    :param drop_description: bool; drop the Description column (column 2 in the .gct) or not
    :return: pandas DataFrame; [n_samples, n_features (or n_features + 1 if not dropping the Description column)]
    """
    from pandas import read_csv

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
    from pandas import Series, DataFrame

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
    # TODO: test

    from pandas import read_csv

    return read_csv(filepath, sep='\t', index_col=0)


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
    obj.index.name = 'Name'
    if descriptions:
        obj.insert(0, 'Description', descriptions)
    else:
        obj.insert(0, 'Description', obj.index)
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
# Work on array-like
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

    x, y = drop_nan_columns([x, y])
    if len(x) < 3 or len(y) < 3:
        return 0
    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Get bandwidths
    cor, p = pearsonr(x, y)
    bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Get P(x, y), P(x), P(y)
    fxy = asarray(kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + EPS
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Get mutual information;
    mi = sum(pxy * log(pxy / (asarray([px] * n_grids).T * asarray([py] * n_grids)))) * dx * dy

    # # Get H(x, y), H(x), and H(y)
    # hxy = - sum(pxy * log(pxy)) * dx * dy
    # hx = -sum(px * log(px)) * dx
    # hy = -sum(py * log(py)) * dy
    # mi = hx + hy - hxy

    # Get information coefficient
    ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic


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


def explode(series, filepath=None):
    """
    Make a label-x-sample binary matrix from a Series.
    :param series: pandas Series;
    :param filepath: str;
    :return: pandas DataFrame;
    """
    from pandas import DataFrame

    label_x_sample = DataFrame(index=sorted(set(series)), columns=series.index)
    for i in label_x_sample.index:
        label_x_sample.ix[i, :] = (series == i).astype(int)
    if filepath:
        establish_path(filepath)
        write_gct(label_x_sample, filepath)

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

    obj = pandas_object.copy()
    print_log('\'{}\' normalizing pandas object on axis={} ...'.format(method, axis))

    if isinstance(obj, Series):
        obj = normalize_series(obj, method=method, n_ranks=n_ranks)
    elif isinstance(obj, DataFrame):
        if not (axis == 0 or axis == 1):  # Normalize globally
            if method == '-0-':
                obj_mean = obj.values.mean()
                obj_std = obj.values.std()
                if obj_std == 0:
                    print_log('Warning: tried to \'-0-\' normalize but the standard deviation is 0.')
                    obj = obj / obj.size
                else:
                    obj = (obj - obj_mean) / obj_std
            elif method == '0-1':
                obj_min = obj.values.min()
                obj_max = obj.values.max()
                obj_range = obj_max - obj_min
                if obj_range == 0:
                    print_log('Warning: tried to \'0-1\' normalize but the range is 0.')
                    obj = obj / obj.size
                else:
                    obj = (obj - obj_min) / obj_range
            elif method == 'rank':
                raise ValueError('mehtod=\'rank\' & axix=\'all\' combination has not been implemented yet.')
        else:  # Normalize by row or by column
            obj = obj.apply(normalize_series, **{'method': method, 'n_ranks': n_ranks}, axis=axis)
    return obj


def drop_nan_columns(arrays):
    """
    Keep only not-NaN column positions in all `arrays`.
    :param arrays: iterable of numpy arrays; must have the same length
    :return: list of numpy arrays; none of the arrays contains NaN
    """
    from numpy import ones, isnan

    not_nan_filter = ones(len(arrays[0]), dtype=bool)
    for v in arrays:
        not_nan_filter &= ~isnan(v)
    return [v[not_nan_filter] for v in arrays]


def get_consensus(clustering_x_sample):
    """
    Count number of co-clusterings.
    :param clustering_x_sample: numpy array; (n_clusterings, n_samples)
    :return: numpy array; (n_samples, n_samples)
    """
    from numpy import zeros

    n_clusterings, n_samples = clustering_x_sample.shape

    consensus_clusterings = zeros((n_samples, n_samples))

    for c_i in range(n_clusterings):
        for i in range(n_samples):
            for j in range(n_samples):
                v1 = clustering_x_sample[c_i, i]
                v2 = clustering_x_sample[c_i, j]
                if v1 and v2 and (v1 == v2):
                    consensus_clusterings[i, j] += 1

    # Return normalized consensus clustering
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

    if distance_function:
        mds_obj = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(compare_matrices(dataframe, dataframe, distance_function,
                                                             is_distance=True, axis=1))
    else:
        mds_obj = MDS(random_state=mds_seed, n_init=n_init, max_iter=max_iter)
        coordinates = mds_obj.fit_transform(dataframe)
    coordinates = DataFrame(coordinates, index=dataframe.index, columns=['x', 'y'])

    if standardize:
        coordinates = normalize_pandas_object(coordinates, method='0-1', axis=0)

    return coordinates


# ======================================================================================================================
# Association
# ======================================================================================================================
def compute_against_reference(features, ref, function=information_coefficient, n_features=0.95, ascending=False,
                              n_samplings=30, confidence=0.95, n_perms=30):
    """
    Compute scores[i] = `features`[i] vs. `ref` using `metric` and get CI, p-val, and FDR (BH).
    :param features: pandas DataFrame; (n_features, n_samples); must have indices and columns
    :param ref: pandas Series; (n_samples); must have name and columns, which must match `features`'s
    :param function: function; function to score
    :param n_features: int or float; number threshold if >= 1 and percentile threshold if < 1
    :param ascending: bool; True if score increase from top to bottom, and False otherwise
    :param n_samplings: int; number of sampling for confidence interval bootstrapping; must be > 2 to compute CI
    :param confidence: float; confidence interval
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

    # Compute scores
    print_log('Computing scores using {} ...'.format(function))
    scores = features.apply(lambda r: function(r, ref), axis=1)
    scores = DataFrame(scores, index=features.index, columns=['score'])

    #
    print_log('Bootstrapping to get {} confidence interval ...'.format(confidence))
    n_samples = math.ceil(0.632 * features.shape[1])
    if n_samples < 3:
        print_log('Can\'t bootstrap with 0.632 * n_samples < 3.')
    else:
        # Limit features to be bootstrapped
        if n_features < 1:  # Limit using percentile
            above_quantile = scores.ix[:, 'score'] >= scores.ix[:, 'score'].quantile(n_features)
            print_log('Bootstrapping {} features (> {} percentile) ...'.format(sum(above_quantile), n_features))
            below_quantile = scores.ix[:, 'score'] <= scores.ix[:, 'score'].quantile(1 - n_features)
            print_log('Bootstrapping {} features (< {} percentile) ...'.format(sum(below_quantile), 1 - n_features))
            indices_to_bootstrap = scores.index[above_quantile | below_quantile].tolist()
        else:  # Limit using numbers
            indices_to_bootstrap = scores.index[:n_features].tolist() + scores.index[-n_features:].tolist()
            print_log('Bootstrapping top & bottom {} features ...'.format(n_features))

        # Bootstrap: randomize columns and compute scores for `n_sampling` times
        sampled_scores = DataFrame(index=indices_to_bootstrap, columns=range(n_samplings))
        for c_i in sampled_scores:
            # Randomize
            sample_indices = choice(features.columns.tolist(), int(n_samples)).tolist()
            sampled_features = features.ix[indices_to_bootstrap, sample_indices]
            sampled_ref = ref.ix[sample_indices]
            # Score
            sampled_scores.ix[:, c_i] = sampled_features.apply(lambda r: function(r, sampled_ref), axis=1)

        # Compute the score's confidence interval using bootstrapped scores' distributions
        z_critical = stats.norm.ppf(q=confidence)
        confidence_intervals = sampled_scores.apply(lambda r: z_critical * (r.std() / math.sqrt(n_samplings)), axis=1)
        confidence_intervals = DataFrame(confidence_intervals,
                                         index=indices_to_bootstrap, columns=['{} MoE'.format(confidence)])

        # Merge
        scores = merge(scores, confidence_intervals, how='outer', left_index=True, right_index='True')

    print_log('Performing permutation test with {} permutations ...'.format(n_perms))
    permutation_pvals_and_fdrs = DataFrame(index=features.index, columns=['Local P-value', 'Global P-value', 'FDR'])

    # Compute scores using permuted ref
    permutation_scores = empty((features.shape[0], n_perms))
    shuffled_ref = array(ref)
    for i in range(n_perms):
        shuffle(shuffled_ref)
        permutation_scores[:, i] = features.apply(lambda r: function(r, shuffled_ref), axis=1)
    # Compute local and global permutation P-values
    all_permutation_scores = permutation_scores.flatten()
    for i, (idx, f) in enumerate(scores.iterrows()):
        # Compute local p-value
        local_pval = float(sum(permutation_scores[i, :] > float(f.ix['score'])) / n_perms)
        if not local_pval:
            local_pval = float(1 / n_perms)
        permutation_pvals_and_fdrs.ix[idx, 'Local P-value'] = local_pval
        # Compute global p-value
        global_pval = float(sum(all_permutation_scores > float(f.ix['score'])) / (n_perms * features.shape[0]))
        if not global_pval:
            global_pval = float(1 / (n_perms * features.shape[0]))
        permutation_pvals_and_fdrs.ix[idx, 'Global P-value'] = global_pval

    # Compute global permutation FDRs
    permutation_pvals_and_fdrs.ix[:, 'FDR'] = multipletests(permutation_pvals_and_fdrs.ix[:, 'Global P-value'],
                                                            method='fdr_bh')[1]

    # Merge
    scores = merge(scores, permutation_pvals_and_fdrs, left_index=True, right_index=True)

    return scores.sort_values('score', ascending=ascending)


def compare_matrices(matrix1, matrix2, function, axis=0, is_distance=False, verbose=False):
    """
    Make association or distance matrix of `matrix1` and `matrix2` by row or column.
    :param matrix1: pandas DataFrame;
    :param matrix2: pandas DataFrame;
    :param function: function; function used to compute association or dissociation
    :param axis: int; 0 for by-row and 1 for by-column
    :param is_distance: bool; True for distance and False for association
    :param verbose: bool;
    :return: pandas DataFrame;
    """
    from pandas import DataFrame

    if axis == 1:
        m1 = matrix1.copy()
        m2 = matrix2.copy()
    else:
        m1 = matrix1.T
        m2 = matrix2.T

    compared_matrix = DataFrame(index=m1.index, columns=m2.index, dtype=float)
    n = m1.shape[0]
    for i, (i1, r1) in enumerate(m1.iterrows()):
        if verbose and i % 50 == 0:
            print_log('Comparing {} ({}/{}) ...'.format(i1, i, n))
        for i2, r2 in m2.iterrows():
            compared_matrix.ix[i1, i2] = function(r1, r2)

    if is_distance:
        print_log('Converting association to distance (1 - association) ...')
        compared_matrix = 1 - compared_matrix

    return compared_matrix


# ======================================================================================================================
# Cluster
# ======================================================================================================================
def consensus_cluster(matrix, ks, max_std=3, n_clusterings=50, filepath_prefix=None):
    """
    Consensus cluster `matrix`'s columns into k clusters.
    :param matrix: pandas DataFrame; (n_features, m_samples)
    :param ks: iterable; list of ks used for clustering
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consenssu clustering
    :param filepath_prefix: str;
    :return: pandas DataFrame and Series; assignment matrix (n_ks, n_samples) and the cophenetic correlations (n_ks)
    """
    from numpy import empty
    from numpy.random import random_integers
    from pandas import DataFrame
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster, cophenet

    # '-0-' normalize `matrix` and clip values `max_std` standard deviation away; then '0-1' normalize the output
    clipped_h = normalize_pandas_object(matrix, method='-0-', axis=1).clip(-max_std, max_std)
    normalized_h = normalize_pandas_object(clipped_h, method='0-1', axis=1)

    # Get distance between samples
    print_log('Computing distances between columns ...')
    sample_distances = compare_matrices(normalized_h, normalized_h, information_coefficient, is_distance=True,
                                        verbose=True)

    print_log('Consensus clustering with {} clusterings ...'.format(n_clusterings))
    consensus_clustering_labels = DataFrame(index=ks, columns=list(matrix.columns))
    consensus_clustering_labels.index.name = 'k'
    cophenetic_correlations = {}
    if isinstance(ks, int):
        ks = [ks]
    for k in ks:
        print_log('k={} ...'.format(k))
        # Hierarchical cluster
        clustering_labels = empty((n_clusterings, matrix.shape[1]))
        for i in range(n_clusterings):
            if i % 10 == 0:
                print_log('\tClustering sample distances ({}/{}) ...'.format(i, n_clusterings))
            randomized_column_indices = random_integers(0, sample_distances.shape[1] - 1, sample_distances.shape[1])
            ward = AgglomerativeClustering(n_clusters=k)
            ward.fit(sample_distances.iloc[randomized_column_indices, randomized_column_indices])
            # Assign column labels
            clustering_labels[i, randomized_column_indices] = ward.labels_

        # Consensus cluster hierarchical clustering labels
        print_log('\tConsensus clustering ...')
        consensus_clusterings = get_consensus(clustering_labels)
        # Convert to distances
        distances = 1 - consensus_clusterings

        # Hierarchical cluster the consensus clusterings to assign the final label
        ward = linkage(distances, method='ward')
        consensus_clustering_labels.ix[k, :] = fcluster(ward, k, criterion='maxclust')
        # Compute clustering scores, the correlation between cophenetic and Euclidian distances
        cophenetic_correlations[k] = cophenet(ward, pdist(distances))[0]
        print_log('Computed cophenetic correlations.')

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
    from numpy import zeros, asarray, argmax
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, cophenet

    if isinstance(ks, int):
        ks = [ks]

    nmf_results = {}
    scores = {}
    if method == 'cophenetic_correlation':
        print_log(
            'Scoring NMF with consensus-clustering ({} clusterings) cophenetic correlation ...'.format(n_clusterings))
        for k in ks:
            print_log('k={} ...'.format(k))

            # NMF cluster
            clustering_labels = zeros((n_clusterings, matrix.shape[1]), dtype=int)
            for i in range(n_clusterings):
                if i % 10 == 0:
                    print_log('\tNMF ({}/{}) ...'.format(i, n_clusterings))
                nmf_result = nmf(matrix, k,
                                 init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_state,
                                 alpha=alpha, l1_ratio=l1_ratio, shuffle_=shuffle_, nls_max_iter=nls_max_iter,
                                 sparseness=sparseness, beta=beta, eta=eta)[k]

                # Save 1 NMF result for eack k
                if i == 0:
                    nmf_results[k] = nmf_result
                    print_log('\t\tSaved the 1st NMF decomposition.')

                # Assigning column labels, the row index holding the highest value
                clustering_labels[i, :] = argmax(asarray(nmf_result['H']), axis=0)

            # Consensus cluster NMF clustering labels
            print_log('\tConsensus clustering ...')
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
