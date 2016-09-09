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
import math

from numpy import asarray, array, zeros, empty, argmax, dot
from numpy.random import choice, shuffle
from numpy.linalg import pinv
from pandas import DataFrame, merge
from scipy.optimize import nnls
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from statsmodels.sandbox.stats.multicomp import multipletests
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

from .support import SEED, print_log, establish_path, write_gct, normalize_pandas_object, compare_matrices, \
    consensus_cluster, mds, fit_columns, exponential_function, get_sample_coordinates_via_pulling
from .information import information_coefficient

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


def nmf_and_score(matrix, ks, method='cophenetic_correlation', n_clusterings=30, initialization='random',
                  max_iteration=200, seed=SEED, regularizer=0, randomize_coordinate_order=False, filepath_prefix=None):
    """
    Perform NMF with k from `ks` and score each NMF result.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param method: str; {'cophenetic_correlation'}
    :param n_clusterings: int; number of NMF clusterings
    :param initialization: str; {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration: int; number of NMF iterations
    :param seed: int;
    :param randomize_coordinate_order: bool;
    :param regularizer: int, NMF's alpha
    :param filepath_prefix: str; `filepath_prefix`_k{k}_{w, h}.gct and  will be saved
    :return: 2 dicts; {k: {W:w, H:h, ERROR:error}} and {k: score}
    """
    nmf_results = {}
    scores = {}
    if method == 'cophenetic_correlation':
        if isinstance(ks, int):
            ks = [ks]
        for k in ks:
            print_log('Computing NMF score using cophenetic correlation for k={} ...'.format(k))

            # NMF cluster
            clustering_labels = empty((n_clusterings, matrix.shape[1]))
            for i in range(n_clusterings):
                print_log('NMF clustering (k={} @ {}/{}) ...'.format(k, i, n_clusterings))
                nmf_result = nmf(matrix, k, initialization=initialization, max_iteration=max_iteration,
                                 seed=seed, regularizer=regularizer,
                                 randomize_coordinate_order=randomize_coordinate_order)[k]
                # Save 1 NMF result for eack k
                if i == 0:
                    nmf_results[k] = nmf_result
                    print_log('\tSaved the 1st NMF decomposition.')
                # Assigning column labels, the row index holding the highest value
                clustering_labels[i, :] = argmax(asarray(nmf_result['H']), axis=0)

            # Consensus cluster NMF clustering labels
            print_log('Consensus clustering NMF clustering labels ...')
            consensus_clusterings = consensus_cluster(clustering_labels)

            # Compute clustering scores, the correlation between cophenetic and Euclidian distances
            scores[k] = cophenet(linkage(consensus_clusterings, 'average'), pdist(consensus_clusterings))[0]
            print_log('Computed the cophenetic correlation coefficient.')
    else:
        raise ValueError('Unknown method {}.'.format(method))

    if filepath_prefix:
        save_nmf_results(nmf_results, filepath_prefix)

    return nmf_results, scores


def nmf(matrix, ks,
        initialization='random', max_iteration=200, seed=SEED, regularizer=0, randomize_coordinate_order=False,
        filepath_prefix=None):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: int or iterable; k or ks to be used in the NMF
    :param initialization: str; {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration: int; number of NMF iterations
    :param seed: int;
    :param randomize_coordinate_order: bool;
    :param regularizer: int, NMF's alpha
    :param filepath_prefix: str; `filepath_prefix`_k{k}_{w, h}.gct and  will be saved
    :return: dict; {k: {W:w, H:h, ERROR:error}}
    """
    nmf_results = {}
    if isinstance(ks, int):
        ks = [ks]
    for k in ks:
        print_log('NMF (k={} & max_iteration={}) ...'.format(k, max_iteration))
        model = NMF(n_components=k, init=initialization, max_iter=max_iteration, random_state=seed, alpha=regularizer,
                    shuffle=randomize_coordinate_order)

        # Compute W, H, and reconstruction error
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_
        if isinstance(matrix, DataFrame):
            # Return pandas DataFrame if the input matrix is also a DataFrame
            w = DataFrame(w, index=matrix.index, columns=[i + 1 for i in range(k)])
            h = DataFrame(h, index=[i + 1 for i in range(k)], columns=matrix.columns)
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

    if filepath_prefix:
        save_nmf_results(nmf_results, filepath_prefix)

    return nmf_results


def nnls_matrix(a, b, method='nnls'):
    """
    Solve `a`x = `b`. (n, k) * (k, m) = (n, m)
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


def save_nmf_results(nmf_results, filepath_prefix):
    """
    Save `nmf_results` dictionary.
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param filepath_prefix: str; `filepath_prefix`_k{k}_{w, h}.gct and  will be saved
    :return: None
    """
    establish_path(filepath_prefix)
    for k, v in nmf_results.items():
        write_gct(v['W'], filepath_prefix + '_nmf_k{}w.gct'.format(k))
        write_gct(v['H'], filepath_prefix + '_nmf_k{}h.gct'.format(k))


def define_states(h, ks, max_std=3, n_clusterings=50, filepath_prefix=None):
    """
    Consensus cluster H matrix's samples into k clusters.
    :param h: pandas DataFrame; H matrix (n_components, n_samples) from NMF
    :param ks: iterable; list of ks used for clustering states
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consenssu clustering
    :param filepath_prefix: str; `filepath_prefix`_labels.txt and `filepath_prefix`_memberships.gct will be saved
    :return: pandas DataFrame, Series, and DataFrame; assignment matrix (n_ks, n_samples),
                                                      the cophenetic correlations (n_ks), and
                                                      membership matrix (n_ks, n_samples)
    """
    # Standardize H and clip extreme values
    standardized_clipped_h = normalize_pandas_object(h, axis=1).clip(-max_std, max_std)

    # Get association between samples
    sample_associations = compare_matrices(standardized_clipped_h, standardized_clipped_h, information_coefficient)

    consensus_clustering_labels = DataFrame(index=ks, columns=list(h.columns) + ['cophenetic_correlation'])
    consensus_clustering_labels.index.name = 'state'
    if any(ks):
        for k in ks:
            print_log('Defining states by consensus clustering for k={} ...'.format(k))

            # Hierarchical cluster
            clustering_labels = empty((n_clusterings, h.shape[1]))
            for i in range(n_clusterings):
                print_log('Hierarchical clustering sample associations (k={} @ {}/{}) ...'.format(k, i, n_clusterings))
                ward = AgglomerativeClustering(n_clusters=k)
                ward.fit(sample_associations)
                # Assign column labels
                clustering_labels[i, :] = ward.labels_

            # Consensus cluster hierarchical clustering labels
            print_log('Consensus hierarchical clustering labels ...')
            consensus_clusterings = consensus_cluster(clustering_labels)

            # Convert to distances
            distances = 1 - consensus_clusterings

            # Hierarchical cluster the consensus clusterings to assign the final label
            ward = linkage(distances, method='ward')
            consensus_clustering_labels.ix[k, sample_associations.index] = fcluster(ward, k, criterion='maxclust')

            # Compute clustering scores, the correlation between cophenetic and Euclidian distances
            consensus_clustering_labels.ix[k, 'cophenetic_correlation'] = cophenet(ward, pdist(distances))[0]
            print_log('Computed the cophenetic correlation coefficient.')

        # Compute membership matrix
        memberships = consensus_clustering_labels.iloc[:, :-1].apply(lambda label: label == int(label.name),
                                                                     axis=1).astype(int)

        if filepath_prefix:
            establish_path(filepath_prefix)
            consensus_clustering_labels.to_csv(filepath_prefix + '_labels.txt', sep='\t')
            write_gct(memberships, filepath_prefix + '_memberships.gct')
    else:
        raise ValueError('Invalid value passed to ks.')

    return consensus_clustering_labels.iloc[:, :-1], consensus_clustering_labels.iloc[:, -1:], memberships


def make_onco_gps(h_train, states_train, std_max=3, h_test=None, h_test_normalization='as_train', states_test=None,
                  informational_mds=True, mds_seed=SEED, mds_n_init=1000, mds_max_iter=1000,
                  function_to_fit=exponential_function, fit_maxfev=1000,
                  fit_min=0, fit_max=2, pull_power_min=1, pull_power_max=3,
                  n_pulling_components='all', component_pull_power='auto', n_pullratio_components=0, pullratio_factor=5,
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
    training_h = normalize_pandas_object(normalize_pandas_object(h_train, axis=1).clip(-std_max, std_max),
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
            ratio = float(c_sorted[:n_pullratio_components].sum() / c_sorted[n_pullratio_components:].sum()) * c.sum()
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
            testing_h = normalize_pandas_object(normalize_pandas_object(h_test, axis=1).clip(-std_max, std_max),
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
