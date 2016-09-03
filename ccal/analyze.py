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

from numpy import asarray, array, zeros, empty, argmax
from numpy.random import choice, shuffle
from pandas import DataFrame, merge
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from sklearn.manifold import MDS
from statsmodels.sandbox.stats.multicomp import multipletests
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d

from .support import SEED, print_log, write_gct, normalize_pandas_object, compare_matrices, exponential_function, \
    establish_path, consensus_cluster
from .information import information_coefficient


def nmf_and_score(matrix, ks, method='cophenetic_correlation', n_clusterings=30,
                  initialization='random', max_iteration=200, seed=SEED, regularizer=0,
                  randomize_coordinate_order=False):
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
    :return: 2 dicts; {k: {W:w, H:h, ERROR:error}} and {k: score}
    """
    nmf_results = {}
    scores = {}
    if method == 'cophenetic_correlation':
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

    return nmf_results, scores


def nmf(matrix, ks,
        initialization='random', max_iteration=200, seed=SEED, regularizer=0, randomize_coordinate_order=False):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: int or iterable; k or ks to be used in the NMF
    :param initialization: str; {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration: int; number of NMF iterations
    :param seed: int;
    :param randomize_coordinate_order: bool;
    :param regularizer: int, NMF's alpha
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

    return nmf_results


def define_states(h, ks, max_std=3, n_clusterings=50, filename_prefix=None):
    """
    Consensus cluster H matrix's samples into k clusters.
    :param h: pandas DataFrame; H matrix (n_components, n_samples) from NMF
    :param ks: iterable; list of ks used for clustering states
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consenssu clustering
    :param filename_prefix: str; `filename_prefix`_labels.txt and `filename_prefix`_memberships.gct will be saved
    :return: pandas DataFrame, Series, and DataFrame; assignment matrix (n_ks, n_samples),
                                                      the cophenetic correlations (n_ks), and
                                                      membership matrix (n_ks, n_samples)
    """
    # Standardize H and clip extreme values
    standardized_clipped_h = normalize_pandas_object(h, axis=1).clip(-max_std, max_std)

    # Get association between samples
    sample_associations = compare_matrices(standardized_clipped_h, standardized_clipped_h, information_coefficient,
                                           axis=1)

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

        if filename_prefix:
            establish_path(filename_prefix)
            consensus_clustering_labels.to_csv(filename_prefix + '_labels.txt', sep='\t')
            write_gct(memberships, filename_prefix + '_memberships.gct')
    else:
        raise ValueError('Invalid value passed to ks.')

    return consensus_clustering_labels.iloc[:, :-1], consensus_clustering_labels.iloc[:, -1:], memberships


def make_onco_gps(h, states, std_max=3, n_grids=128, informational_mds=True, mds_seed=SEED, kde_bandwidths_factor=1,
                  n_influencing_components='all', sample_stretch_factor='auto'):
    """
    :param h: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param states: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values
    :param n_grids: int;
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param sample_stretch_factor: str or number; power to raise components' influence on each sample; 'auto' to automate
    :return: None
    """
    unique_states = sorted(set(states))
    print_log('Creating Onco-GPS with {} samples, {} components, and {} states {} ...'.format(*reversed(h.shape),
                                                                                              len(unique_states),
                                                                                              unique_states))

    # Compute component coordinates
    # Standardize H and clip values with extreme standard deviation
    normalized_clipped_h = normalize_pandas_object(normalize_pandas_object(h, axis=1).clip(-std_max, std_max),
                                                   method='0-1', axis=1)
    # Project the H's components from <n_sample>D to 2D
    if informational_mds:
        mds = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=1000, max_iter=1000)
        components_coordinates = mds.fit_transform(compare_matrices(normalized_clipped_h, normalized_clipped_h,
                                                                    information_coefficient, is_distance=True))
    else:
        mds = MDS(random_state=mds_seed, n_init=1000, max_iter=1000)
        components_coordinates = mds.fit_transform(normalized_clipped_h)
    # 0-1 normalize the coordinates
    x_min = min(components_coordinates[:, 0])
    x_max = max(components_coordinates[:, 0])
    x_range = x_max - x_min
    y_min = min(components_coordinates[:, 1])
    y_max = max(components_coordinates[:, 1])
    y_range = y_max - y_min
    for i, (x, y) in enumerate(components_coordinates):
        components_coordinates[i, 0] = (x - x_min) / x_range
        components_coordinates[i, 1] = (y - y_min) / y_range

    # Get sample states and compute coordinates
    samples = DataFrame(index=h.columns, columns=['state', 'x', 'y'])
    # Get sample states
    samples.ix[:, 'state'] = states
    # Compute sample coordinates
    if sample_stretch_factor == 'auto':
        print_log('Computing the sample_stretch_factor ...')
        x = array(range(normalized_clipped_h.shape[0]))
        y = asarray(normalized_clipped_h.apply(sorted).apply(sum, axis=1)) / normalized_clipped_h.shape[1]
        a, k, c = curve_fit(exponential_function, x, y)[0]
        print_log('\tModeled H columns by {}e^({}x) + {}.'.format(a, k, c))
        k_min, k_max = 0, 2
        stretch_factor_min, stretch_factor_max = 1, 3
        k_normalized = (k - k_min) / (k_max - k_min)
        sample_stretch_factor = k_normalized * (stretch_factor_max - stretch_factor_min) + stretch_factor_min
        print_log('\tsample_stretch_factor = {0:.3f}.'.format(sample_stretch_factor))
    for sample in samples.index:
        col = h.ix[:, sample]
        if n_influencing_components == 'all':
            n_influencing_components = h.shape[0]
        col = col.mask(col < col.sort_values()[-n_influencing_components], other=0)
        x = sum(col ** sample_stretch_factor * components_coordinates[:, 0]) / sum(col ** sample_stretch_factor)
        y = sum(col ** sample_stretch_factor * components_coordinates[:, 1]) / sum(col ** sample_stretch_factor)
        samples.ix[sample, ['x', 'y']] = x, y

    # Compute grid probabilities and states
    grid_probabilities = zeros((n_grids, n_grids))
    grid_states = empty((n_grids, n_grids))
    # Get KDE for each state using bandwidth created from all states' x & y coordinates
    kdes = zeros((len(unique_states) + 1, n_grids, n_grids))
    bandwidths = asarray([bcv(asarray(samples.ix[:, 'x'].tolist()))[0],
                          bcv(asarray(samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor
    for s in unique_states:
        coordinates = samples.ix[samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = kde2d(asarray(coordinates.ix[:, 'x'], dtype=float), asarray(coordinates.ix[:, 'y'], dtype=float),
                    bandwidths, n=asarray([n_grids]), lims=asarray([0, 1, 0, 1]))
        kdes[s] = asarray(kde[2])
    # Assign the best KDE probability and state for each grid
    for i in range(n_grids):
        for j in range(n_grids):
            grid_probabilities[i, j] = max(kdes[:, j, i])
            grid_states[i, j] = argmax(kdes[:, i, j])

    return components_coordinates, samples, grid_probabilities, grid_states


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
    :return: pandas DataFrame (nfeatures, nscores),
    """
    print_log('Computing scores using {} metric and ...'.format(function))
    scores = features.apply(lambda r: function(r, ref), axis=1)
    scores = DataFrame(scores, index=features.index, columns=[function])

    print_log('Bootstrapping to get {} confidence interval ...'.format(confidence))
    n_samples = math.ceil(0.632 * features.shape[1])
    if n_samples < 3:
        print_log('Can\'t bootstrap with 0.632 * n_samples < 3.')
    else:
        # Limit features to be bootstrapped
        if n_features < 1:  # Limit using percentile
            above_quantile = scores.ix[:, function] >= scores.ix[:, function].quantile(n_features)
            print_log('Bootstrapping {} features (> {} percentile) ...'.format(sum(above_quantile), n_features))
            below_quantile = scores.ix[:, function] <= scores.ix[:, function].quantile(1 - n_features)
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
        confidence_intervals = DataFrame(index=indices_to_bootstrap, columns=['{} MoE'.format(confidence)])
        z_critical = stats.norm.ppf(q=confidence)
        sampled_scores.apply(lambda r: z_critical * (r.std() / math.sqrt(n_samplings)), axis=1)

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
        local_pval = float(sum(permutation_scores[i, :] > float(f.ix[function])) / n_perms)
        if not local_pval:
            local_pval = float(1 / n_perms)
        permutation_pvals_and_fdrs.ix[idx, 'Local P-value'] = local_pval
        # Compute global p-value
        global_pval = float(sum(all_permutation_scores > float(f.ix[function])) / (n_perms * features.shape[0]))
        if not global_pval:
            global_pval = float(1 / (n_perms * features.shape[0]))
        permutation_pvals_and_fdrs.ix[idx, 'Global P-value'] = global_pval

    # Compute global permutation FDRs
    permutation_pvals_and_fdrs.ix[:, 'FDR'] = multipletests(permutation_pvals_and_fdrs.ix[:, 'Global P-value'],
                                                            method='fdr_bh')[1]

    # Merge
    scores = merge(scores, permutation_pvals_and_fdrs, left_index=True, right_index=True)

    return scores.sort_values(function, ascending=ascending)
