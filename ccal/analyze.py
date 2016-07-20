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

from .support import verbose_print, establish_path
from .visualize import CMAP_CONTINUOUS, plot_nmf_result, plot_features_and_reference
from .information import information_coefficient, cmi_diff, cmi_ratio

# ======================================================================================================================
# Global variables
# ======================================================================================================================
# Path to testing data directory
SEED = 20121020


# ======================================================================================================================
# Feature selection
# ======================================================================================================================
def rank_features_against_reference(features, ref,
                                    features_type='continuous', ref_type='continuous',
                                    features_ascending=False, ref_ascending=False, ref_sort=True,
                                    metric='information_coef', nsampling=30, confidence=0.95, nperm=30,
                                    title=None, n_features=0, rowname_size=24,
                                    output_prefix=None, figure_type='.png'):
    """
    Compute features vs. `ref`.
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param ref: pandas Series (m_elements), must have name and columns, which must match 'features`'s
    :param features_type: str, {continuous, categorical, binary}
    :param ref_type: str, {continuous, categorical, binary}
    :param features_ascending: bool, True if features score increase from top to bottom, False otherwise
    :param ref_ascending: bool, True if ref values increase from left to right, False otherwise
    :param ref_sort: bool, sort each ref or not
    :param metric: str, {information_coef}
    :param nsampling: int, number of sampling for confidence interval bootstrapping
    :param confidence: float, confidence interval
    :param nperm: int, number of permutations for permutation test
    :param title: string for the title of heatmap
    :param n_features: int or float, number threshold if >= 1 and quantile threshold if < 1
    :param rowname_size: int, the maximum length of a feature name label
    :param output_prefix: str, file path prefix to save the result (.txt) and figure (`figure_type`)
    :param figure_type: str, file type to save the output figure
    :return: None
    """
    verbose_print('Computing features vs. {} using {} metric ...'.format(ref.name, metric))

    # Establish output file path
    if output_prefix:
        output_prefix = os.path.abspath(output_prefix)
        establish_path(output_prefix)

    # Use only the intersecting columns
    # TODO: preserve order
    col_intersection = set(features.columns) & set(ref.index)
    if not col_intersection:
        raise ValueError(
            'No intersecting columns from features and ref, which have {} and {} columns respectively'.format(
                features.shape[1], ref.size))
    verbose_print(
        'Using {} intersecting columns from features and ref, which have {} and {} columns respectively ...'.format(
            len(col_intersection), features.shape[1], ref.size))
    features = features.ix[:, col_intersection]
    ref = ref.ix[col_intersection]

    # Sort ref and use its sorted indices to sort features indices
    if ref_sort:
        ref = ref.sort_values(ascending=ref_ascending)
        features = features.reindex_axis(ref.index, axis=1)

    # Compute scores, sorted by information coefficient
    scores = compute_against_reference(features, ref, metric=metric, ascending=features_ascending,
                                       nsampling=nsampling, confidence=confidence, nperm=nperm)
    features = features.reindex(scores.index)

    if output_prefix:
        filename = output_prefix + '.txt'
        # TODO: add scores
        features.to_csv(filename, sep='\t')
        verbose_print('Saved the result as {}.'.format(filename))

    # Plot features panel
    verbose_print('Plotting top {} features vs. ref ...'.format(n_features))
    # Make annotation
    annotations = pd.DataFrame(index=features.index)
    annotations['IC'] = ['{0:.2f}'.format(x) for x in scores.ix[:, 'information_coef']]
    annotations['P'] = ['{0:.2f}'.format(x) for x in scores.ix[:, 'Global P-Value']]
    annotations['CI'] = scores.ix[:, '{} CI'.format(confidence)].tolist()

    if n_features < 1:
        indices_to_plot = features.iloc[:, -1] >= features.iloc[:, -1].quantile(n_features)
        indices_to_plot |= features.iloc[:, -1] <= features.iloc[:, -1].quantile(1 - n_features)
    else:
        indices_to_plot = features.index[:n_features].tolist() + features.index[-n_features:].tolist()
    plot_features_and_reference(features.ix[indices_to_plot, :], ref, annotations.ix[indices_to_plot, :],
                                features_type=features_type, ref_type=ref_type,
                                title=title, rowname_size=rowname_size,
                                filename_prefix=output_prefix, figure_type=figure_type)


def compute_against_reference(features, ref, metric='information_coef', ascending=False, nsampling=30, confidence=0.95,
                              nperm=30):
    """
    Compute scores[i] = `features`[i] vs. `ref` with computation using `metric` and get CI, p-val, and FDR (BH).
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param ref: pandas Series (m_elements), must have indices, which must match 'features`'s columns
    :param metric: str, {information_coef}
    :param ascending: bool, True if score increase from top to bottom, False otherwise
    :param nsampling: int, number of sampling for confidence interval bootstrapping
    :param confidence: float, confidence intrval
    :param nperm: int, number of permutations for permutation test
    :return: pandas DataFrame (n_features, n_scores),
    """
    # Compute score[i] = <features>[i] vs. <ref>
    if metric is 'information_coef':
        function = information_coefficient
    elif metric is 'information_cmi_diff':
        function = cmi_diff
    elif metric is 'information_cmi_ratio':
        function = cmi_ratio
    else:
        raise ValueError('Unknown metric {}.'.format(metric))

    print('Computing with metric {} ...'.format(metric))
    scores = pd.DataFrame([function(s, ref) for idx, s in features.iterrows()],
                          index=features.index, columns=[metric])

    print('Bootstrapping to get {} confidence interval ...'.format(confidence))
    confidence_intervals = pd.DataFrame(index=features.index, columns=['{} CI'.format(confidence)])
    # Random sample elements
    nsample = math.ceil(0.632 * features.shape[0])
    sampled_scores = np.empty((features.shape[0], nsampling))
    for i in range(nsampling):
        sample_indices = np.random.choice(features.columns.tolist(), int(nsample)).tolist()
        sampled_features = features.ix[:, sample_indices]
        sampled_ref = ref.ix[sample_indices]
        # Compute sample score
        for j, (idx, s) in enumerate(sampled_features.iterrows()):
            sampled_scores[j, i] = function(s, sampled_ref)
    # Get confidence interval
    z_critical = stats.norm.ppf(q=confidence)
    for i, f in enumerate(sampled_scores):
        mean = f.mean()
        stdev = f.std()
        moe = z_critical * (stdev / math.sqrt(f.size))
        confidence_intervals.iloc[i] = '<{0:.2f}, {0:.2f}>'.format(mean - moe, mean + moe)

    print('Performing permutation test with {} permutations ...'.format(nperm))
    permutation_pvals_and_fdrs = pd.DataFrame(index=features.index,
                                              columns=['Local P-Value', 'Global P-Value', 'FDR (BH)'])
    permutation_scores = np.empty((features.shape[0], nperm))
    # Permute ref and compute score against it
    shuffled_ref = np.array(ref)
    for i in range(nperm):
        np.random.shuffle(shuffled_ref)
        for j, (idx, s) in enumerate(features.iterrows()):
            permutation_scores[j, i] = information_coefficient(s, shuffled_ref)
    # Compute permutation p-value
    all_permutation_scores = permutation_scores.flatten()
    for i, (idx, f) in enumerate(scores.iterrows()):
        # Local P-Value
        local_pval = float(sum(permutation_scores[i, :] > float(f)) / nperm)
        if not local_pval:
            local_pval = float(1 / nperm)
        permutation_pvals_and_fdrs.ix[idx, 'Local P-Value'] = local_pval
        # Global P-Value
        global_pval = float(sum(all_permutation_scores > float(f)) / (nperm * features.shape[0]))
        if not global_pval:
            global_pval = float(1 / (nperm * features.shape[0]))
        permutation_pvals_and_fdrs.ix[idx, 'Global P-Value'] = global_pval
    # Compute permutation FDR
    permutation_pvals_and_fdrs.ix[:, 'FDR (BH)'] = multipletests(permutation_pvals_and_fdrs.ix[:, 'Global P-Value'],
                                                                 method='fdr_bh')[1]
    results = pd.concat([scores, confidence_intervals, permutation_pvals_and_fdrs], axis=1)
    return results.sort_values('information_coef', ascending=ascending)


def compare_matrices(matrix1, matrix2, is_distance=False, result_filename=None,
                     figure_filename=None):
    """
    Make association or distance matrix of the rows of `feature1` and `feature2`.
    :param matrix1: pandas DataFrame,
    :param matrix2: pandas DataFrame,
    :param is_distance: bool, True for distance and False for association
    :param result_filename: str, filepath to save the result
    :param figure_filename: str, filepath to save the figure
    :return:
    """
    association_matrix = pd.DataFrame(index=matrix1.index, columns=matrix2.index, dtype=float)
    features1_nrow = matrix1.shape[0]
    for i, (i1, r1) in enumerate(matrix1.iterrows()):
        verbose_print('Features 1 {} ({}/{}) vs. features 2 ...'.format(i1, i + 1, features1_nrow))
        for i2, r2 in matrix2.iterrows():
            association_matrix.ix[i1, i2] = information_coefficient(r1, r2)
    if is_distance:
        verbose_print('Converting association to is_distance (is_distance = 1 - association) ...')
        association_matrix = 1 - association_matrix
    if result_filename:
        establish_path(result_filename)
        association_matrix.to_csv(result_filename, sep='\t')
        verbose_print('Saved the resulting matrix as {}.'.format(result_filename))

    verbose_print('Plotting the resulting matrix ...')
    ax = sns.clustermap(association_matrix, cmap=CMAP_CONTINUOUS)
    plt.setp(ax.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    if figure_filename:
        establish_path(figure_filename)
        association_matrix.to_csv(figure_filename, sep='\t')
        verbose_print('Saved the resulting figure as {}.'.format(figure_filename))


# ======================================================================================================================
# NMF
# ======================================================================================================================
def nmf(matrix, ks, initialization='random', max_iteration=200, seed=SEED, randomize_coordinate_order=False,
        regularizer=0, plot=False):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix: numpy array (n_samples, n_features), the matrix to be factorized by NMF
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
        verbose_print('Performing NMF with k={} ...'.format(k))
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
            verbose_print('\tPlotting W and H matrices ...')
            plot_nmf_result(nmf_results, k)

    return nmf_results


def nmf_and_score(matrix, ks, method='cophenetic_correlation', nassignment=20):
    """
    Perform NMF with multiple k and score each computation.
    :param matrix: numpy array (n_samples, n_features), the matrix to be factorized by NMF
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
            verbose_print('Computing clustering score for k={} using method {} ...'.format(k, method))

            assignments = {}  # dict (key: assignemnt index; value: samples)
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
            verbose_print('Score for k={}: {}'.format(k, assignment_scores_per_k.mean()))

    elif method == 'cophenetic_correlation':
        nmf_results = {}
        for k in ks:
            verbose_print('Computing clustering score for k={} using method {} ...'.format(k, method))

            # Make assignment matrix (nassignment, ncol assingments from H)
            assignment_matrix = np.empty((nassignment, ncol))
            for i in range(nassignment):
                verbose_print('Running NMF #{} (total number of assignments={}) ...'.format(i, nassignment))
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

            verbose_print('Computing the cophenetic correlation coefficient ...')

            # Compute the cophenetic correlation coefficient of the hierarchically clustered distances and
            # the normalized assignment distances
            score = cophenet(linkage(normalized_assignment_distance_matrix, 'average'),
                             pdist(normalized_assignment_distance_matrix))[0]
            scores[k] = score
            verbose_print('Score for k={}: {}'.format(k, score))
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
