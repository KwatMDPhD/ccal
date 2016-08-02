"""
Computational Cancer Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov


Description:
Main analysis module for CCAL.
"""
import os
import math

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns

from . import SEED
from .support import print_log, establish_path
from .visualize import CMAP_CONTINUOUS, plot_nmf_result, plot_features_and_reference
from .information import information_coefficient, cmi_diff, cmi_ratio


# ======================================================================================================================
# Feature selection
# ======================================================================================================================
def rank_features_against_reference(features, ref, features_type='continuous', ref_type='continuous',
                                    features_ascending=False, ref_ascending=False, ref_sort=True,
                                    metric='information_coef', nsampling=30, confidence=0.95, nperm=30, nfeatures=0.95,
                                    title=None, title_size=16, annotation_label_size=9, plot_colname=False,
                                    result_filename=None, figure_filename=None):
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
    :param nsampling: int, number of sampling for confidence interval bootstrapping; must be > 2 for CI computation
    :param confidence: float, confidence interval
    :param nperm: int, number of permutations for permutation test
    :param nfeatures: int or float, number threshold if >= 1 and quantile threshold if < 1
    :param title: str, plot title
    :param title_size: int, title text size
    :param annotation_label_size: int, annotation text size
    :param plot_colname: bool, plot column names or not
    :param result_filename: str, file path to the output result
    :param figure_filename: str, file path to the output figure
    :return: None
    """
    if features.shape[0] is 1 or isinstance(features, pd.Series):
        features = pd.DataFrame(features).T

    print_log('Computing features vs. {} using {} metric ...'.format(ref.name, metric))

    # TODO: preserve order
    col_intersection = set(features.columns) & set(ref.index)
    if not col_intersection:
        raise ValueError(
            'No intersecting columns from features and ref, which have {} and {} columns respectively'.format(
                features.shape[1], ref.size))
    else:
        print_log(
            'Using {} intersecting columns from features and ref, which have {} and {} columns respectively ...'.format(
                len(col_intersection), features.shape[1], ref.size))
        features = features.ix[:, col_intersection]
        ref = ref.ix[col_intersection]

    if ref_sort:
        ref = ref.sort_values(ascending=ref_ascending)
        features = features.reindex_axis(ref.index, axis=1)

    scores = compute_against_reference(features, ref, metric=metric, ascending=features_ascending,
                                       nsampling=nsampling, confidence=confidence, nperm=nperm)
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
    # TODO: use the same features for the bootstrapping
    if nfeatures < 1:
        above_quantile = scores.ix[:, metric] >= scores.ix[:, metric].quantile(nfeatures)
        print_log('Plotting {} features vs. reference > {} quantile ...'.format(sum(above_quantile), nfeatures))
        below_quantile = scores.ix[:, metric] <= scores.ix[:, metric].quantile(1 - nfeatures)
        print_log('Plotting {} features vs. reference < {} quantile ...'.format(sum(below_quantile), 1 - nfeatures))
        indices_to_plot = features.index[above_quantile | below_quantile].tolist()
    else:
        indices_to_plot = features.index[:nfeatures].tolist() + features.index[-nfeatures:].tolist()
        print_log('Plotting top and bottom {} features vs. reference ...'.format(len(indices_to_plot)))

    plot_features_and_reference(features.ix[indices_to_plot, :], ref, annotations.ix[indices_to_plot, :],
                                features_type=features_type, ref_type=ref_type, title=title, title_size=title_size,
                                annotation_header=' ' * 7 + 'IC(\u0394)' + ' ' * 9 + 'P-val' + ' ' * 4 + 'FDR',
                                annotation_label_size=annotation_label_size,
                                plot_colname=plot_colname, figure_filename=figure_filename)


def compute_against_reference(features, ref, metric='information_coef', nfeatures=0, ascending=False,
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
    if metric is 'information_coef':
        function = information_coefficient
    elif metric is 'information_cmi_diff':
        function = cmi_diff
    elif metric is 'information_cmi_ratio':
        function = cmi_ratio
    else:
        raise ValueError('Unknown metric {}.'.format(metric))

    print_log('Computing scores using {} metric ...'.format(metric))
    scores = np.empty(features.shape[0])
    for i, (idx, s) in enumerate(features.iterrows()):
        if i % 100 is 0:
            print_log('\t{}/{} ...'.format(i, features.shape[0]))
        scores[i] = function(s, ref)
    scores = pd.DataFrame(scores, index=features.index, columns=[metric])

    print_log('Bootstrapping to get {} confidence interval ...'.format(confidence))
    nsample = math.ceil(0.632 * features.shape[1])
    if nsampling < 2:
        print_log('Not bootstrapping because number of sampling < 3.')
    elif nsample < 3:
        print_log('Not bootstrapping because 0.632 * number of sample < 3.')
    else:
        # Random sample columns and compute scores using the sampled columns
        sampled_scores = np.empty((features.shape[0], nsampling))
        for i in range(nsampling):
            sample_indices = np.random.choice(features.columns.tolist(), int(nsample)).tolist()
            sampled_features = features.ix[:, sample_indices]
            sampled_ref = ref.ix[sample_indices]
            for j, (idx, s) in enumerate(sampled_features.iterrows()):
                sampled_scores[j, i] = function(s, sampled_ref)
        # Get confidence intervals
        confidence_intervals = pd.DataFrame(index=features.index, columns=['{} MoE'.format(confidence)])
        z_critical = stats.norm.ppf(q=confidence)

        for i, f in enumerate(sampled_scores):
            std = f.std()
            moe = z_critical * (std / math.sqrt(f.size))
            confidence_intervals.iloc[i, 0] = moe
        scores = pd.merge(scores, confidence_intervals, left_index=True, right_index=True)

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


def compare_matrices(matrix1, matrix2, axis=0, function=information_coefficient, is_distance=False):
    """
    Make association or distance matrix of the rows of `matrix1` and `matrix2`.
    :param matrix1: pandas DataFrame,
    :param matrix2: pandas DataFrame,
    :param axis: int, 0 for row-wise and 1 for column-wise
    :param function: function, function for computing association or dissociation
    :param is_distance: bool, True for distance and False for association
    :return: pandas DataFrame
    """
    if axis is 1:
        matrix1 = matrix1.T
        matrix2 = matrix2.T

    compared_matrix = pd.DataFrame(index=matrix1.index, columns=matrix2.index, dtype=float)
    nrow = matrix1.shape[0]
    for i, (i1, r1) in enumerate(matrix1.iterrows()):
        print_log('Comparing {} ({}/{}) vs. ...'.format(i1, i + 1, nrow))
        for i2, r2 in matrix2.iterrows():
            compared_matrix.ix[i1, i2] = function(r1, r2)

    if is_distance:
        print_log('Converting association to is_distance (is_distance = 1 - association) ...')
        compared_matrix = 1 - compared_matrix

    return compared_matrix


# ======================================================================================================================
# NMF
# ======================================================================================================================
def nmf(matrix, ks, initialization='random', max_iteration=200, seed=SEED, randomize_coordinate_order=False,
        regularizer=0, plot=False):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix: numpy array (nsamples, nfeatures), the matrix to be factorized by NMF
    :param ks: array-like, list of ks to be used in the factorization
    :param initialization: str, {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration: int, number of NMF iterations
    :param seed: int, random seed
    :param randomize_coordinate_order: bool,
    :param regularizer: int, NMF's alpha
    :param plot: bool, whether to plot the NMF results
    :return: dict, NMF result per k (key: k; value: dict(key: w, h, err; value: w matrix, h matrix, and error))
    """
    nmf_results = {}  # dict (key:k; value:dict (key:w, h, err; value:w matrix, h matrix, and reconstruction error))
    for k in ks:
        print_log('Performing NMF with k={} ...'.format(k))
        model = NMF(n_components=k,
                    init=initialization,
                    max_iter=max_iteration,
                    random_state=seed,
                    alpha=regularizer,
                    shuffle=randomize_coordinate_order)

        # Compute W, H, and reconstruction error
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

        if plot:
            print_log('\tPlotting W and H matrices ...')
            plot_nmf_result(nmf_results, k)

    return nmf_results


def nmf_and_score(matrix, ks, method='cophenetic_correlation', nassignment=20):
    """
    Perform NMF with multiple k and score each computation.
    :param matrix: numpy array (nsamples, nfeatures), the matrix to be factorized by NMF
    :param ks: array-like, list of ks to be used in the factorization
    :param method: str, {'intra_inter_ratio', 'cophenetic_correlation'}
    :param nassignment: int, number of assignments used to make `assigment_matrix` when using 'cophenetic_correlation'
    :return: dict, NMF result per k
                    (key: k; value: dict (key: w, h, err; value: w matrix, h matrix, and reconstruction error)) and
                    score per k (key:k; value:score)
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
                print_log('Running NMF #{} (total number of assignments={}) ...'.format(i, nassignment))
                nmf_result = nmf(matrix, [k])[k]
                # Save the 1st NMF result for each k
                if i == 0:
                    nmf_results[k] = nmf_result
                # Assignment a col with the highest index value
                assignment_matrix[i, :] = np.argmax(nmf_result['H'], axis=0)

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


# ======================================================================================================================
# Onco GPS functions
# ======================================================================================================================
def oncogps_define_state():
    """
    Compute the OncoGPS states by consensus clustering.
    :return:
    """


def oncogps_map():
    """
    Plot and map OncoGPS.
    :return:
    """


def oncogps_populate_map():
    """
    Populate samples on a Onco GPS map with features.
    :return:
    """
