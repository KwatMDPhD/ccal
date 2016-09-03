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

from .support import SEED, print_log, write_gct, normalize_pandas_object, compare_matrices, \
    exponential_function
from .information import information_coefficient


def nmf_and_score(matrix, ks, method='cophenetic_correlation', n_assignments=30):
    """
    Perform NMF with k from `ks` and score each computation.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param method: str; {'cophenetic_correlation'}
    :param n_assignments: int; number of assignments used to make `assigment_matrix` when using 'cophenetic_correlation'
    :return: 2 dicts; {k: {W:w, H:h, ERROR:error}} and {k: score}
    """
    scores = {}
    if method == 'cophenetic_correlation':
        nmf_results = {}
        for k in ks:
            print_log('Computing clustering score for k={} using method {} ...'.format(k, method))

            # Make assignment matrix
            assignment_matrix = empty((n_assignments, matrix.shape[1]))
            for i in range(n_assignments):
                print_log('Running NMF ({}/{}) ...'.format(i, n_assignments))
                nmf_result = nmf(matrix, k)[k]
                # Save the 1st NMF result for each k
                if i == 0:
                    nmf_results[k] = nmf_result
                # Assignment a col with the highest index value
                assignment_matrix[i, :] = argmax(asarray(nmf_result['H']), axis=0)

            # Make assignment distance matrix (n_cols, n_cols)
            assignment_distance_matrix = zeros((matrix.shape[1], matrix.shape[1]))
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[1])[i:]:
                    for a in range(n_assignments):
                        if assignment_matrix[a, i] == assignment_matrix[a, j]:
                            assignment_distance_matrix[i, j] += 1
            normalized_assignment_distance_matrix = assignment_distance_matrix / n_assignments

            # Compute the cophenetic correlation coefficient of the hierarchically clustered distances and
            # the normalized assignment distances
            print_log('Computing the cophenetic correlation coefficient ...')
            score = cophenet(linkage(normalized_assignment_distance_matrix, 'average'),
                             pdist(normalized_assignment_distance_matrix))[0]
            scores[k] = score
    else:
        raise ValueError('Unknown method {}.'.format(method))

    return nmf_results, scores


def nmf(matrix, ks,
        initialization='random', max_iteration=200, seed=SEED, randomize_coordinate_order=False, regularizer=0):
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
        print_log('Performing NMF with k={} ...'.format(k))
        model = NMF(n_components=k, init=initialization, max_iter=max_iteration, random_state=seed, alpha=regularizer,
                    shuffle=randomize_coordinate_order)

        # Compute W, H, and reconstruction error
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_
        if isinstance(matrix, DataFrame):
            c = ['C{}'.format(i + 1) for i in range(k)]
            w = DataFrame(w, index=matrix.index, columns=c)
            h = DataFrame(h, index=c, columns=matrix.columns)
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

    return nmf_results


def define_states(h, n_states, max_std=3, n_clusterings=50, filename_prefix=None):
    """
    Cluster H matrix's samples into k clusters.
    :param h: pandas DataFrame; (n_components, n_samples), H matrix from NMF
    :param n_states: iterable; list of ks used for clustering states
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of consensus clusterings
    :param filename_prefix: str; file path to save the assignment matrix
    :return: pandas DataFrame, Series, and DataFrame; assignment matrix (n_ks, n_samples),
                                                      the cophenetic correlations (n_ks), and
                                                      membership matrix (n_ks, n_samples)
    """
    # Standardize H and clip extreme values
    standardized_clipped_h = normalize_pandas_object(h).clip(-max_std, max_std)

    # Get association between samples
    sample_associations = compare_matrices(standardized_clipped_h, standardized_clipped_h, information_coefficient,
                                           axis=1)

    # Assign labels using each k
    labels = DataFrame(index=n_states, columns=list(sample_associations.index) + ['cophenetic_correlation'])
    labels.index.name = 'state'
    if any(n_states):
        for k in n_states:
            # For n_clusterings times, cluster sample associations and assign labels using this k
            n_clusterings_labels = DataFrame(index=range(n_clusterings), columns=sample_associations.index)
            for i in range(n_clusterings):
                print_log('Clustering sample associations with k = {} ({}/{}) ...'.format(k, i, n_clusterings))
                ward = AgglomerativeClustering(n_clusters=k)
                ward.fit(sample_associations)
                n_clusterings_labels.iloc[i, :] = ward.labels_

            # Count co-clustering between samples
            n_coclusterings = DataFrame(index=n_clusterings_labels.columns, columns=n_clusterings_labels.columns)
            n_coclusterings.fillna(0, inplace=True)
            for r, s in n_clusterings_labels.iterrows():
                print_log('Counting co-clustering between samples with k = {} ({}/{}) ...'.format(k, r, n_clusterings))
                for i in s.index:
                    for j in s.index:
                        if i == j or s.ix[i] == s.ix[j]:
                            n_coclusterings.ix[i, j] += 1
            # Normalize by the nclustering and convert to distances
            distances = 1 - n_coclusterings / n_clusterings

            # Cluster the distances and assign the final label using this k
            ward = linkage(distances, method='ward')
            labels_ = fcluster(ward, k, criterion='maxclust')
            labels.ix[k, sample_associations.index] = labels_

            # Compute the correlation between cophenetic and Euclidian distances between samples
            labels.ix[k, 'cophenetic_correlation'] = cophenet(ward, pdist(distances))[0]

        # Compute membership matrix
        memberships = labels.iloc[:, :-1].apply(lambda label: label == int(label.name), axis=1).astype(int)

        if filename_prefix:
            labels.to_csv(filename_prefix + '_labels.txt', sep='\t')
            write_gct(memberships, filename_prefix + '_memberships.gct')
    else:
        raise ValueError('Invalid value passed to n_states.')

    return labels.iloc[:, :-1], labels.iloc[:, -1:], memberships


def make_onco_gps(h, states, std_max=3, n_grids=128,
                  informational_mds=True, mds_seed=SEED,
                  kde_bandwidths_factor=1, n_influencing_components='all', sample_stretch_factor='auto'):
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

    samples = DataFrame(index=h.columns, columns=['state', 'x', 'y'])

    # Get sample states
    samples.ix[:, 'state'] = states

    # Get component coordinates
    # Standardize H and clip values with extreme standard deviation
    normalized_clipped_h = normalize_pandas_object(normalize_pandas_object(h).clip(-std_max, std_max), method='0-1')
    # Project the H's components from <n_sample>D to 2D
    if informational_mds:
        mds = MDS(dissimilarity='precomputed', random_state=mds_seed, n_init=1000, max_iter=1000)
        components_coordinates = mds.fit_transform(compare_matrices(normalized_clipped_h, normalized_clipped_h,
                                                                    information_coefficient, is_distance=True,
                                                                    report_progress=False))
    else:
        mds = MDS(random_state=mds_seed, n_init=1000, max_iter=1000)
        components_coordinates = mds.fit_transform(normalized_clipped_h)
    x_min = min(components_coordinates[:, 0])
    x_max = max(components_coordinates[:, 0])
    x_range = x_max - x_min
    y_min = min(components_coordinates[:, 1])
    y_max = max(components_coordinates[:, 1])
    y_range = y_max - y_min
    # 0-1 normalize the coordinates
    for i, (x, y) in enumerate(components_coordinates):
        components_coordinates[i, 0] = (x - x_min) / x_range
        components_coordinates[i, 1] = (y - y_min) / y_range

    # Get sample coordinates
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

    # Get KDE for each state using bandwidth created from all states' x & y coordinates
    kdes = zeros((len(unique_states) + 1, n_grids, n_grids))
    bandwidths = array([mass.bcv(array(samples.ix[:, 'x'].tolist()))[0],
                        mass.bcv(array(samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor
    for s in unique_states:
        coordinates = samples.ix[samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = mass.kde2d(array(coordinates.ix[:, 'x'], dtype=float), array(coordinates.ix[:, 'y'], dtype=float),
                         bandwidths, n=array([n_grids]), lims=array([0, 1, 0, 1]))
        kdes[s] = array(kde[2])
    # Assign the best KDE probability and state for each grid
    grid_probabilities = zeros((n_grids, n_grids))
    grid_states = empty((n_grids, n_grids))
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
    # Set computing function
    print_log('Computing scores using {} metric ...'.format(function))
    # Compute and rank
    scores = empty(features.shape[0])
    for i, (idx, s) in enumerate(features.iterrows()):
        if i % 1000 is 0:
            print_log('\t{}/{} ...'.format(i, features.shape[0]))
        scores[i] = function(s, ref)
    scores = DataFrame(scores, index=features.index, columns=[function]).sort_values(function)

    print_log('Bootstrapping to get {} confidence interval ...'.format(confidence))
    n_samples = math.ceil(0.632 * features.shape[1])
    if n_samplings < 2:
        print_log('Not bootstrapping because number of sampling < 3.')
    elif n_samples < 3:
        print_log('Not bootstrapping because 0.632 * number of sample < 3.')
    else:
        # Limit features to be bootstrapped
        if n_features < 1:  # limit using percentile
            above_quantile = scores.ix[:, function] >= scores.ix[:, function].quantile(n_features)
            print_log('Bootstrapping {} features vs. reference > {} percentile ...'.format(sum(above_quantile),
                                                                                           n_features))
            below_quantile = scores.ix[:, function] <= scores.ix[:, function].quantile(1 - n_features)
            print_log('Bootstrapping {} features vs. reference < {} percentile ...'.format(sum(below_quantile),
                                                                                           1 - n_features))
            indices_to_bootstrap = scores.index[above_quantile | below_quantile].tolist()
        else:  # limit using numbers
            indices_to_bootstrap = scores.index[:n_features].tolist() + scores.index[-n_features:].tolist()
            print_log('Bootstrapping top & bottom {} features vs. reference ...'.format(len(indices_to_bootstrap)))

        # Random sample columns and compute scores using the sampled columns
        sampled_scores = DataFrame(index=indices_to_bootstrap, columns=range(n_samplings))
        for c in sampled_scores:
            sample_indices = choice(features.columns.tolist(), int(n_samples)).tolist()
            sampled_features = features.ix[indices_to_bootstrap, sample_indices]
            sampled_ref = ref.ix[sample_indices]
            for idx, s in sampled_features.iterrows():
                sampled_scores.ix[idx, c] = function(s, sampled_ref)

        # Get confidence intervals
        confidence_intervals = DataFrame(index=indices_to_bootstrap, columns=['{} MoE'.format(confidence)])
        z_critical = stats.norm.ppf(q=confidence)
        for i, s in sampled_scores.iterrows():
            std = s.std()
            moe = z_critical * (std / math.sqrt(s.size))
            confidence_intervals.ix[i, 0] = moe
        scores = merge(scores, confidence_intervals, how='outer', left_index=True, right_index='True')

    print_log('Performing permutation test with {} permutations ...'.format(n_perms))
    permutation_pvals_and_fdrs = DataFrame(index=features.index, columns=['Local P-value', 'Global P-value', 'FDR'])
    # Compute scores using permuted ref
    permutation_scores = empty((features.shape[0], n_perms))
    shuffled_ref = array(ref)
    for i in range(n_perms):
        shuffle(shuffled_ref)
        for j, (idx, s) in enumerate(features.iterrows()):
            permutation_scores[j, i] = function(s, shuffled_ref)

    # Compute permutation P-values and FDRs
    all_permutation_scores = permutation_scores.flatten()
    for i, (idx, f) in enumerate(scores.iterrows()):
        local_pval = float(sum(permutation_scores[i, :] > float(f.ix[function])) / n_perms)
        if not local_pval:
            local_pval = float(1 / n_perms)
        permutation_pvals_and_fdrs.ix[idx, 'Local P-value'] = local_pval

        global_pval = float(sum(all_permutation_scores > float(f.ix[function])) / (n_perms * features.shape[0]))
        if not global_pval:
            global_pval = float(1 / (n_perms * features.shape[0]))
        permutation_pvals_and_fdrs.ix[idx, 'Global P-value'] = global_pval

    permutation_pvals_and_fdrs.ix[:, 'FDR (BH)'] = multipletests(permutation_pvals_and_fdrs.ix[:, 'Global P-value'],
                                                                 method='fdr_bh')[1]
    scores = merge(scores, permutation_pvals_and_fdrs, left_index=True, right_index=True)

    return scores.sort_values(function, ascending=ascending)
