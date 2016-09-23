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
import math

from numpy import asarray, array, zeros, empty, exp, argmax
from numpy.random import choice, random_integers, shuffle
from pandas import DataFrame, Series, merge
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
from sklearn.decomposition import NMF
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from statsmodels.sandbox.stats.multicomp import multipletests
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

from .support import SEED, EPS, print_log, establish_path, write_gct, write_dictionary
from .information import information_coefficient

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


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


def get_consensus(clustering_x_sample):
    """
    Count number of co-clusterings.
    :param clustering_x_sample: numpy array; (n_clusterings, n_samples)
    :return: numpy array; (n_samples, n_samples)
    """
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


# ======================================================================================================================
# Cluster
# ======================================================================================================================
def consensus_cluster(h, ks, max_std=3, n_clusterings=50, filepath_prefix=None):
    """
    Consensus cluster H matrix's samples into k clusters.
    :param h: pandas DataFrame; H matrix (n_components, n_samples) from NMF
    :param ks: iterable; list of ks used for clustering states
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consenssu clustering
    :param filepath_prefix: str;
    :return: pandas DataFrame and Series; assignment matrix (n_ks, n_samples) and the cophenetic correlations (n_ks)
    """
    # '-0-' normalize H and clip values `max_std` standard deviation away; then '0-1' normalize the output
    clipped_h = normalize_pandas_object(h, method='-0-', axis=1).clip(-max_std, max_std)
    normalized_h = normalize_pandas_object(clipped_h, method='0-1', axis=1)

    # Get distance between samples
    print_log('Computing distances between samples (columns) ...')
    sample_distances = compare_matrices(normalized_h, normalized_h, information_coefficient, is_distance=True,
                                        verbose=True)

    print_log('Consensus clustering with {} clusterings ...'.format(n_clusterings))
    consensus_clustering_labels = DataFrame(index=ks, columns=list(h.columns))
    consensus_clustering_labels.index.name = 'k'
    cophenetic_correlations = {}
    if isinstance(ks, int):
        ks = [ks]
    for k in ks:
        print_log('k={} ...'.format(k))
        # Hierarchical cluster
        clustering_labels = empty((n_clusterings, h.shape[1]))
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
# Make Onco-GPS
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
    return a * exp(k * x) + c


def make_onco_gps_elements(h_train, states_train, std_max=3, h_test=None, h_test_normalization='as_train',
                           states_test=None,
                           informational_mds=True, mds_seed=SEED, mds_n_init=1000, mds_max_iter=1000,
                           function_to_fit=exponential_function, fit_maxfev=1000,
                           fit_min=0, fit_max=2, pull_power_min=1, pull_power_max=3,
                           n_pulling_components='all', component_pull_power='auto', n_pullratio_components=0,
                           pullratio_factor=5,
                           n_grids=128, kde_bandwidths_factor=1):
    """
    Compute component and sample coordinates. And compute grid probabilities and states.
    :param h_train: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param states_train: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values
    :param h_test: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param h_test_normalization: str or None; {'as_train', 'clip_and_0-1', None}
    :param states_test: iterable of int; (n_samples); sample states
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param mds_n_init: int;
    :param mds_max_iter: int;
    :param function_to_fit: function;
    :param fit_maxfev: int;
    :param fit_min: number;
    :param fit_max: number;
    :param pull_power_min: number;
    :param pull_power_max: number;
    :param n_pulling_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_pull_power: str or number; power to raise components' influence on each sample
    :param n_pullratio_components: number; number if int; percentile if float & < 1
    :param pullratio_factor: number;
    :param n_grids: int;
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :return: pandas DataFrame, DataFrame, numpy array, and numpy array;
             component_coordinates (n_components, [x, y]), samples (n_samples, [x, y, state, annotation]),
             grid_probabilities (n_grids, n_grids), and grid_states (n_grids, n_grids)
    """
    print_log('Making Onco-GPS with {} components, {} samples, and {} states {} ...'.format(*h_train.shape,
                                                                                            len(set(states_train)),
                                                                                            set(states_train)))

    # training_samples = DataFrame(index=h_train.columns, columns=['x', 'y', 'state'])

    # clip and 0-1 normalize the data
    training_h = normalize_pandas_object(normalize_pandas_object(h_train, method='-0-', axis=1).clip(-std_max, std_max),
                                         method='0-1', axis=1)

    # Compute component coordinates
    component_coordinates = mds(training_h, informational_mds=informational_mds,
                                mds_seed=mds_seed, n_init=mds_n_init, max_iter=mds_max_iter, standardize=True)

    # Compute component pulling power
    if component_pull_power == 'auto':
        fit_parameters = fit_columns(training_h, function_to_fit=function_to_fit, maxfev=fit_maxfev)
        print_log('Modeled columns by {}e^({}x) + {}.'.format(*fit_parameters))
        k = fit_parameters[1]
        # Linear transform
        k_normalized = (k - fit_min) / (fit_max - fit_min)
        component_pull_power = k_normalized * (pull_power_max - pull_power_min) + pull_power_min
        print_log('component_pulling_power = {0:.3f}.'.format(component_pull_power))

    # Compute sample coordinates
    training_samples = get_sample_coordinates_via_pulling(component_coordinates, training_h,
                                                          n_influencing_components=n_pulling_components,
                                                          component_pulling_power=component_pull_power)

    # Compute pulling ratios
    ratios = empty(training_h.shape[1])
    if 0 < n_pullratio_components:
        if n_pullratio_components < 1:
            n_pullratio_components = training_h.shape[0] * n_pullratio_components
        for i, (c_idx, c) in enumerate(training_h.iteritems()):
            c_sorted = c.sort_values(ascending=False)
            ratio = float(
                c_sorted[:n_pullratio_components].sum() / max(c_sorted[n_pullratio_components:].sum(), EPS)) * c.sum()
            ratios[i] = ratio
        normalized_ratios = (ratios - ratios.min()) / (ratios.max() - ratios.min()) * pullratio_factor
        training_samples.ix[:, 'pullratio'] = normalized_ratios.clip(0, 1)

    # Load sample states
    training_samples.ix[:, 'state'] = states_train

    # Compute grid probabilities and states
    grid_probabilities = zeros((n_grids, n_grids))
    grid_states = empty((n_grids, n_grids), dtype=int)
    # Get KDE for each state using bandwidth created from all states' x & y coordinates; states starts from 1, not 0
    kdes = zeros((training_samples.ix[:, 'state'].unique().size + 1, n_grids, n_grids))
    bandwidths = asarray([bcv(asarray(training_samples.ix[:, 'x'].tolist()))[0],
                          bcv(asarray(training_samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor
    for s in sorted(training_samples.ix[:, 'state'].unique()):
        coordinates = training_samples.ix[training_samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = kde2d(asarray(coordinates.ix[:, 'x'], dtype=float), asarray(coordinates.ix[:, 'y'], dtype=float),
                    bandwidths, n=asarray([n_grids]), lims=asarray([0, 1, 0, 1]))
        kdes[s] = asarray(kde[2])
    # Assign the best KDE probability and state for each grid
    for i in range(n_grids):
        for j in range(n_grids):
            grid_probabilities[i, j] = max(kdes[:, j, i])
            grid_states[i, j] = argmax(kdes[:, i, j])

    if isinstance(h_test, DataFrame):
        print_log('Focusing on samples from testing H matrix ...')
        # Normalize testing H
        if h_test_normalization == 'as_train':
            testing_h = h_test
            for r_idx, r in h_train.iterrows():
                if r.std() == 0:
                    testing_h.ix[r_idx, :] = testing_h.ix[r_idx, :] / r.size()
                else:
                    testing_h.ix[r_idx, :] = (testing_h.ix[r_idx, :] - r.mean()) / r.std()
        elif h_test_normalization == 'clip_and_0-1':
            testing_h = normalize_pandas_object(
                normalize_pandas_object(h_test, method='-0-', axis=1).clip(-std_max, std_max),
                method='0-1', axis=1)
        elif not h_test_normalization:
            testing_h = h_test
        else:
            raise ValueError('Unknown normalization method for testing H {}.'.format(h_test_normalization))

        # Compute testing-sample coordinates
        testing_samples = get_sample_coordinates_via_pulling(component_coordinates, testing_h,
                                                             n_influencing_components=n_pulling_components,
                                                             component_pulling_power=component_pull_power)
        testing_samples.ix[:, 'state'] = states_test
        return component_coordinates, testing_samples, grid_probabilities, grid_states
    else:
        return component_coordinates, training_samples, grid_probabilities, grid_states


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
        c = c.mask(c < c.sort_values().tolist()[-n_influencing_components], other=0)
        x = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'x']) / sum(c ** component_pulling_power)
        y = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'y']) / sum(c ** component_pulling_power)
        sample_coordinates.ix[sample, ['x', 'y']] = x, y
    return sample_coordinates


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
    print_log('Computing scores using {} ...'.format(function))
    scores = features.apply(lambda r: function(r, ref), axis=1)
    scores = DataFrame(scores, index=features.index, columns=['score'])

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

        # Randomly sample columns and compute scores using the sampled columns
        sampled_scores = DataFrame(index=indices_to_bootstrap, columns=range(n_samplings))
        for c in sampled_scores:
            sample_indices = choice(features.columns.tolist(), int(n_samples)).tolist()
            sampled_features = features.ix[indices_to_bootstrap, sample_indices]
            sampled_ref = ref.ix[sample_indices]
            sampled_scores.ix[:, c] = sampled_features.apply(lambda r: function(r, sampled_ref), axis=1)

        # Get confidence intervals
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


# ======================================================================================================================#
# Normalize
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
