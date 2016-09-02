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
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from statsmodels.sandbox.stats.multicomp import multipletests

from .information import information_coefficient
from .support import SEED, print_log, establish_path, write_gct, normalize_pandas_object, compare_matrices
from .visualize import DPI, plot_features_against_reference


def nmf(matrix, ks, initialization='random', max_iteration=200, seed=SEED, randomize_coordinate_order=False,
        regularizer=0):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix: numpy array or pandas DataFrame; (nsamples, nfeatures); the matrix to be factorized by NMF
    :param ks: array-like; list of ks to be used in the factorization
    :param initialization: string; {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration: int; number of NMF iterations
    :param seed: int;
    :param randomize_coordinate_order: bool;
    :param regularizer: int, NMF's alpha
    :return: dict; {k: {W:w, H:h, ERROR:error}}
    """
    nmf_results = {}  # dict (key:k; value:dict (key:w, h, err; value:w matrix, h matrix, and reconstruction error))
    for k in ks:
        print_log('Performing NMF with k={} ...'.format(k))
        model = NMF(n_components=k, init=initialization, max_iter=max_iteration, random_state=seed, alpha=regularizer,
                    shuffle=randomize_coordinate_order)

        # Compute W, H, and reconstruction error
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_
        if isinstance(matrix, pd.DataFrame):
            c = ['C{}'.format(i + 1) for i in range(k)]
            w = pd.DataFrame(w, index=matrix.index, columns=c)
            h = pd.DataFrame(h, index=c, columns=matrix.columns)
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

    return nmf_results


def nmf_and_score(matrix, ks, method='cophenetic_correlation', nassignment=30):
    """
    Perform NMF with multiple k and score each computation.
    :param matrix: numpy array or pandas DataFrame; (nsamples, nfeatures); the matrix to be factorized by NMF
    :param ks: array-like; list of ks to be used in the factorization
    :param method: string; {'intra_inter_ratio', 'cophenetic_correlation'}
    :param nassignment: int; number of assignments used to make `assigment_matrix` when using 'cophenetic_correlation'
    :return: 2 dict; {k: {W:w, H:h, ERROR:error}} and {k: score}
    """
    nrow, ncol = matrix.shape
    scores = {}

    if method == 'intra_inter_ratio':
        nmf_results = nmf(matrix, ks)
        for k, nmf_result in nmf_results.items():
            print_log('Computing clustering score for k={} using method {} ...'.format(k, method))

            assignments = {}  # dict (key: assignment index; value: samples)
            # Cluster of a sample is the index with the highest value in corresponding H column
            for assigned_sample in zip(np.argmax(nmf_result['H'], axis=0), matrix):
                if assigned_sample[0] not in assignments:
                    assignments[assigned_sample[0]] = set()
                    assignments[assigned_sample[0]].add(assigned_sample[1])
                else:
                    assignments[assigned_sample[0]].add(assigned_sample[1])

            # Compute intra vs. inter clustering distances
            assignment_scores_per_k = np.zeros(nmf_result['H'].shape[1])
            for sidx, (a, samples) in enumerate(assignments.items()):
                for s in samples:
                    # Compute the distance to samples with the same assignment
                    intra_distance = []
                    for other_s in samples:
                        if other_s == s:
                            continue
                        else:
                            intra_distance.append(distance.euclidean(matrix.ix[:, s], matrix.ix[:, other_s]))
                    # Compute the distance to samples with different assignment
                    inter_distance = []
                    for other_a in assignments.keys():
                        if other_a == a:
                            continue
                        else:
                            for other_s in assignments[other_a]:
                                inter_distance.append(distance.euclidean((matrix.ix[:, s]), matrix.ix[:, other_s]))
                    # Compute assignment score
                    score = np.mean(intra_distance) / np.mean(inter_distance)
                    if not np.isnan(score):
                        assignment_scores_per_k[sidx] = score

            scores[k] = assignment_scores_per_k.mean()
            print_log('Score for k={}: {}'.format(k, assignment_scores_per_k.mean()))

    elif method == 'cophenetic_correlation':
        nmf_results = {}
        for k in ks:
            print_log('Computing clustering score for k={} using method {} ...'.format(k, method))

            # Make assignment matrix (nassignment, ncol assingments from H)
            assignment_matrix = np.empty((nassignment, ncol))
            for i in range(nassignment):
                print_log('Running NMF ({}/{}) ...'.format(i, nassignment))
                nmf_result = nmf(matrix, [k])[k]
                # Save the 1st NMF result for each k
                if i == 0:
                    nmf_results[k] = nmf_result
                # Assignment a col with the highest index value
                assignment_matrix[i, :] = np.argmax(np.asarray(nmf_result['H']), axis=0)

            # Make assignment distance matrix (ncol, ncol)
            assignment_distance_matrix = np.zeros((ncol, ncol))
            for i in range(ncol):
                for j in range(ncol)[i:]:
                    for a in range(nassignment):
                        if assignment_matrix[a, i] == assignment_matrix[a, j]:
                            assignment_distance_matrix[i, j] += 1

            # Normalize assignment distance matrix by the nassignment
            normalized_assignment_distance_matrix = assignment_distance_matrix / nassignment

            print_log('Computing the cophenetic correlation coefficient ...')

            # Compute the cophenetic correlation coefficient of the hierarchically clustered distances and
            # the normalized assignment distances
            score = cophenet(linkage(normalized_assignment_distance_matrix, 'average'),
                             pdist(normalized_assignment_distance_matrix))[0]
            scores[k] = score
            print_log('Score for k={}: {}'.format(k, score))
    else:
        raise ValueError('Unknown method {}.'.format(method))

    return nmf_results, scores


def get_states_from_h(h, n_states, nclustering=50, filename_prefix=None):
    """
    Cluster H matrix's samples into k clusters.
    :param h: pandas DataFrame (n_component, n_sample), H matrix from NMF
    :param n_states: array-like, list of ks used for clustering states
    :param nclustering: int, number of consensus clustering to perform
    :param filename_prefix: str; file path to save the assignment matrix (n_k, n_samples)
    :return: pandas DataFrame (n_k, n_samples), array-like (n_k), assignment matrix and the cophenetic correlations
    """
    # Standardize H and clip values less than -3 and more than 3
    standardized_h = normalize_pandas_object(h)
    standardized_clipped_h = standardized_h.clip(-3, 3)

    # Get association between samples
    sample_associations = compare_matrices(standardized_clipped_h, standardized_clipped_h, information_coefficient,
                                           axis=1)

    # Assign labels using each k
    labels = pd.DataFrame(index=n_states, columns=list(sample_associations.index) + ['cophenetic_correlation'])
    labels.index.name = 'state'
    if any(n_states):
        for k in n_states:
            # For nclustering times, cluster sample associations and assign labels using this k
            nclustering_labels = pd.DataFrame(index=range(nclustering), columns=sample_associations.index)
            for i in range(nclustering):
                print_log(
                    'Clustering sample associations and assigning labels with k = {} ({}/{}) ...'.format(k, i,
                                                                                                         nclustering))
                ward = AgglomerativeClustering(n_clusters=k)
                ward.fit(sample_associations)
                nclustering_labels.iloc[i, :] = ward.labels_

            # Count co-clustering between samples
            ncoclusterings = pd.DataFrame(index=nclustering_labels.columns, columns=nclustering_labels.columns)
            ncoclusterings.fillna(0, inplace=True)
            for i, s in nclustering_labels.iterrows():
                print_log('Counting co-clustering between samples with k = {} ({}/{}) ...'.format(k, i, nclustering))
                for i in s.index:
                    for j in s.index:
                        if i == j or s.ix[i] == s.ix[j]:
                            ncoclusterings.ix[i, j] += 1

            # Normalize by the nclustering and convert to distances
            distances = 1 - ncoclusterings / nclustering

            # Cluster the distances and assign the final label using this k
            ward = linkage(distances, method='ward')
            labels_ = fcluster(ward, k, criterion='maxclust')
            labels.ix[k, sample_associations.index] = labels_

            # Compute the cophenetic correlation, the correlation between cophenetic and Euclidian distances between samples
            labels.ix[k, 'cophenetic_correlation'] = cophenet(ward, pdist(distances))[0]

        # Compute membership matrix
        memberships = labels.iloc[:, :-1].apply(lambda s: s == int(s.name), axis=1).astype(int)

        if filename_prefix:
            labels.to_csv(filename_prefix + '_labels.txt', sep='\t')
            write_gct(memberships, filename_prefix + '_memberships.gct')
    else:
        raise ValueError('No number of clusters passed.')

    return labels.iloc[:, :-1], memberships, labels.iloc[:, -1:]


def rank_features_against_reference(features, ref, features_type='continuous', ref_type='continuous',
                                    features_ascending=False, ref_ascending=False, ref_sort=True,
                                    metric='information_coef', n_features=0.95, n_samplings=30, confidence=0.95,
                                    n_perms=30, title=None, title_size=16, annotation_label_size=9, plot_colname=False,
                                    result_filename=None, figure_filename=None, figure_size='auto', dpi=DPI):
    """
    Compute features vs. `ref`.
    :param features: pandas DataFrame (nfeatures, nelements), must have indices and columns
    :param ref: pandas Series (nelements), must have name and columns, which must match 'features`'s
    :param features_type: str, {continuous, categorical, binary}
    :param ref_type: str, {continuous, categorical, binary}
    :param features_ascending: bool, True if features score increase from top to bottom, False otherwise
    :param ref_ascending: bool, True if ref values increase from left to right, False otherwise
    :param ref_sort: bool, sort each ref or not
    :param metric: str, {information_coef}
    :param n_features: int or float, number threshold if >= 1 and quantile threshold if < 1
    :param n_samplings: int, number of sampling for confidence interval bootstrapping; must be > 2 for CI computation
    :param confidence: float, confidence interval
    :param n_perms: int, number of permutations for permutation test
    :param title: str, plot title
    :param title_size: int, title text size
    :param annotation_label_size: int, annotation text size
    :param plot_colname: bool, plot column names or not
    :param result_filename: str, file path to the output result
    :param figure_filename: str, file path to the output figure
    :param figure_size: 'auto' or tuple;
    :param dpi: int;
    :return: None
    """
    print_log('Computing features vs. {} using {} metric ...'.format(ref.name, metric))

    # Convert features into pandas DataFrame
    if isinstance(features, pd.Series):
        features = pd.DataFrame(features).T

    # Use intersecting columns
    col_intersection = set(features.columns) & set(ref.index)
    if not col_intersection:
        raise ValueError('features and ref have 0 intersecting columns, having {} and {} columns respectively'.format(
            features.shape[1], ref.size))
    else:
        print_log('features and ref have {} intersecting columns, having {} and {} columns respectively ...'.format(
            len(col_intersection), features.shape[1], ref.size))
        features = features.ix[:, col_intersection]
        ref = ref.ix[col_intersection]

    # Drop rows having all 0 values
    features = features.ix[(features != 0).any(axis=1)]

    # Sort reference
    if ref_sort:
        ref = ref.sort_values(ascending=ref_ascending)
        features = features.reindex_axis(ref.index, axis=1)

    # Compute scores
    scores = compute_against_reference(features, ref, metric=metric, nfeatures=n_features, ascending=features_ascending,
                                       nsampling=n_samplings, confidence=confidence, nperm=n_perms)
    features = features.reindex(scores.index)

    if result_filename:
        establish_path(os.path.split(result_filename)[0])
        pd.merge(features, scores, left_index=True, right_index=True).to_csv(result_filename, sep='\t')
        print_log('Saved the result as {}.'.format(result_filename))

    # Make annotations
    annotations = pd.DataFrame(index=features.index)
    for idx, s in features.iterrows():
        if '{} MoE'.format(confidence) in scores.columns:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}'.format(scores.ix[idx, metric]) \
                                                + '({0:.3f})'.format(scores.ix[idx, '{} MoE'.format(confidence)])
        else:
            annotations.ix[idx, 'IC(\u0394)'] = '{0:.3f}(x.xxx)'.format(scores.ix[idx, metric])

    annotations['P-val'] = ['{0:.3f}'.format(x) for x in scores.ix[:, 'Global P-value']]
    annotations['FDR'] = ['{0:.3f}'.format(x) for x in scores.ix[:, 'FDR (BH)']]

    # Limit features to be plotted
    if n_features < 1:
        above_quantile = scores.ix[:, metric] >= scores.ix[:, metric].quantile(n_features)
        print_log('Plotting {} features vs. reference > {} quantile ...'.format(sum(above_quantile), n_features))
        below_quantile = scores.ix[:, metric] <= scores.ix[:, metric].quantile(1 - n_features)
        print_log('Plotting {} features vs. reference < {} quantile ...'.format(sum(below_quantile), 1 - n_features))
        indices_to_plot = features.index[above_quantile | below_quantile].tolist()
    else:
        indices_to_plot = features.index[:n_features].tolist() + features.index[-n_features:].tolist()
        print_log('Plotting top & bottom {} features vs. reference ...'.format(len(indices_to_plot)))

    plot_features_against_reference(features.ix[indices_to_plot, :], ref, annotations.ix[indices_to_plot, :],
                                    feature_type=features_type, ref_type=ref_type,
                                    figure_size=figure_size, title=title, title_size=title_size,
                                    annotation_header=' ' * 7 + 'IC(\u0394)' + ' ' * 9 + 'P-val' + ' ' * 4 + 'FDR',
                                    annotation_label_size=annotation_label_size, plot_colname=plot_colname,
                                    output_filepath=figure_filename, dpi=dpi)


def compute_against_reference(features, ref, metric='information_coef', nfeatures=0.95, ascending=False,
                              nsampling=30, confidence=0.95, nperm=30):
    """
    Compute scores[i] = `features`[i] vs. `ref` with computation using `metric` and get CI, p-val, and FDR (BH).
    :param features: pandas DataFrame (nfeatures, nelements), must have indices and columns
    :param ref: pandas Series (nelements), must have indices, which must match 'features`'s columns
    :param metric: str, {information_coef}
    :param nfeatures: int or float, number threshold if >= 1 and quantile threshold if < 1
    :param ascending: bool, True if score increase from top to bottom, False otherwise
    :param nsampling: int, number of sampling for confidence interval bootstrapping; must be > 2 for CI computation
    :param confidence: float, confidence intrval
    :param nperm: int, number of permutations for permutation test
    :return: pandas DataFrame (nfeatures, nscores),
    """
    # Set computing function
    if metric is 'information_coef':
        function = information_coefficient
    else:
        raise ValueError('Unknown metric {}.'.format(metric))
    print_log('Computing scores using {} metric ...'.format(metric))

    # Compute and rank
    scores = np.empty(features.shape[0])
    for i, (idx, s) in enumerate(features.iterrows()):
        if i % 1000 is 0:
            print_log('\t{}/{} ...'.format(i, features.shape[0]))
        scores[i] = function(s, ref)
    scores = pd.DataFrame(scores, index=features.index, columns=[metric]).sort_values(metric)

    print_log('Bootstrapping to get {} confidence interval ...'.format(confidence))
    nsample = math.ceil(0.632 * features.shape[1])
    if nsampling < 2:
        print_log('Not bootstrapping because number of sampling < 3.')
    elif nsample < 3:
        print_log('Not bootstrapping because 0.632 * number of sample < 3.')
    else:
        # Limit features to be bootstrapped
        if nfeatures < 1:  # limit using quantile
            above_quantile = scores.ix[:, metric] >= scores.ix[:, metric].quantile(nfeatures)
            print_log('Bootstrapping {} features vs. reference > {} quantile ...'.format(sum(above_quantile),
                                                                                         nfeatures))
            below_quantile = scores.ix[:, metric] <= scores.ix[:, metric].quantile(1 - nfeatures)
            print_log('Bootstrapping {} features vs. reference < {} quantile ...'.format(sum(below_quantile),
                                                                                         1 - nfeatures))
            indices_to_bootstrap = scores.index[above_quantile | below_quantile].tolist()
        else:  # limit using numbers
            indices_to_bootstrap = scores.index[:nfeatures].tolist() + scores.index[-nfeatures:].tolist()
            print_log('Bootstrapping top & bottom {} features vs. reference ...'.format(len(indices_to_bootstrap)))

        # Random sample columns and compute scores using the sampled columns
        sampled_scores = pd.DataFrame(index=indices_to_bootstrap, columns=range(nsampling))
        for c in sampled_scores:
            sample_indices = np.random.choice(features.columns.tolist(), int(nsample)).tolist()
            sampled_features = features.ix[indices_to_bootstrap, sample_indices]
            sampled_ref = ref.ix[sample_indices]
            for idx, s in sampled_features.iterrows():
                sampled_scores.ix[idx, c] = function(s, sampled_ref)

        # Get confidence intervals
        confidence_intervals = pd.DataFrame(index=indices_to_bootstrap, columns=['{} MoE'.format(confidence)])
        z_critical = stats.norm.ppf(q=confidence)
        for i, s in sampled_scores.iterrows():
            std = s.std()
            moe = z_critical * (std / math.sqrt(s.size))
            confidence_intervals.ix[i, 0] = moe
        scores = pd.merge(scores, confidence_intervals, how='outer', left_index=True, right_index='True')

    print_log('Performing permutation test with {} permutations ...'.format(nperm))
    permutation_pvals_and_fdrs = pd.DataFrame(index=features.index,
                                              columns=['Local P-value', 'Global P-value', 'FDR (BH)'])
    # Compute scores using permuted ref
    permutation_scores = np.empty((features.shape[0], nperm))
    shuffled_ref = np.array(ref)
    for i in range(nperm):
        np.random.shuffle(shuffled_ref)
        for j, (idx, s) in enumerate(features.iterrows()):
            permutation_scores[j, i] = function(s, shuffled_ref)

    # Compute permutation P-values and FDRs
    all_permutation_scores = permutation_scores.flatten()
    for i, (idx, f) in enumerate(scores.iterrows()):
        local_pval = float(sum(permutation_scores[i, :] > float(f.ix[metric])) / nperm)
        if not local_pval:
            local_pval = float(1 / nperm)
        permutation_pvals_and_fdrs.ix[idx, 'Local P-value'] = local_pval

        global_pval = float(sum(all_permutation_scores > float(f.ix[metric])) / (nperm * features.shape[0]))
        if not global_pval:
            global_pval = float(1 / (nperm * features.shape[0]))
        permutation_pvals_and_fdrs.ix[idx, 'Global P-value'] = global_pval

    permutation_pvals_and_fdrs.ix[:, 'FDR (BH)'] = multipletests(permutation_pvals_and_fdrs.ix[:, 'Global P-value'],
                                                                 method='fdr_bh')[1]
    scores = pd.merge(scores, permutation_pvals_and_fdrs, left_index=True, right_index=True)

    return scores.sort_values(metric, ascending=ascending)
