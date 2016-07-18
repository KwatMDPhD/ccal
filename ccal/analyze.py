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

import numpy as np
import pandas as pd
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
PATH_TEST_DATA = os.path.join('data', 'test')

SEED = 20121020

TESTING = False


# ======================================================================================================================
# Feature selection
# ======================================================================================================================
def rank_features_against_references(features, refs, metric, ref_type='continuous', feat_type='continuous', direction='pos',
                                     sort_ref=True, n_features=0.95, rowname_size=25, out_file=None, title=''):
    """
    Compute features vs. each ref in `refs`.
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param refs: pandas DataFrame (n_features, m_elements), must have indices and columns, which must match 'features`'s
    :param metric: str, {information}
    :param direction: str, {pos, neg}
    :param ref_type: str, {continuous, categorical, binary}
    :param feat_type: str, {continuous, categorical, binary}
    :param sort_ref: bool, sort each ref or not
    :param n_features: int or float, number threshold if >= 1 and quantile threshold if < 1
    :param rowname_size: int, the maximum length of a feature name label
    :param out_file: str,file path to save the result (.txt) and figure (.TODO)
    :param title: string for the title of heatmap
    :return: None
    """
    if out_file:
        establish_path(out_file)

    for i, (idx, ref) in enumerate(refs.iterrows()):
        # verbose_print('Computing features vs. {} ({}/{}) using {} metric ...'.format(idx, i + 1, refs.shape[0], metric))

        # Use only the intersecting columns
        col_intersection = set(features.columns) & set(ref.index)
        # verbose_print(
        #    'Using {} intersecting columns from features and ref, which have {} and {} columns respectively ...'.format(
        #        len(col_intersection), features.shape[1], ref.size))
        features = features.ix[:, col_intersection]
        ref = ref.ix[col_intersection]

        ref = ref.apply(pd.to_numeric, errors='coerce')

        # print(ref)
        
        # Sort ref and features
        if sort_ref:
            ref = ref.sort_values(ascending = False)
            features = features.reindex_axis(ref.index, axis=1)

        # Compute scores, join them in features, and rank features based on scores
        scores = compute_against_reference(features, ref, metric)

        # Normalize 
        # verbose_print('Plotting top {} features vs. ref ...'.format(n_features))
        if ref_type is 'continuous':
            # verbose_print('Normalizing continuous features and ref ...')
            ref = (ref - ref.mean()) / ref.std()
            for i, (idx, s) in enumerate(features.iterrows()):
                mean = s.mean()
                std = s.std()
                for j, v in enumerate(s):
                    features.iloc[i, j] = (v - mean) / std

        features = features.join(scores)
        # TODO: decide what metric to sort by
                    
        features.sort_values(features.columns[-1], ascending = (direction == 'neg'), inplace=True)

        # Plot features panel
                    
        if n_features < 1:
            indices_to_plot = features.iloc[:, -1] >= features.iloc[:, -1].quantile(n_features_to_plot)
            indices_to_plot |= features.iloc[:, -1] <= features.iloc[:, -1].quantile(1 - n_features_to_plot)
        elif n_features >= 1:
            indices_to_plot = features.index[:n_features].tolist() + features.index[
                                                                             -n_features:].tolist()
        
        plot_features_and_reference(pd.DataFrame(features.ix[indices_to_plot, features.columns[:-1]]),
                                    ref, pd.DataFrame(features.ix[indices_to_plot, features.columns[-1]]),
                                    ref_type=ref_type, feat_type=feat_type, rowname_size=rowname_size,
                                    out_file=out_file, title=title)

def compute_against_reference(features, ref, metric):
    """
    Compute scores[i] = `features`[i] vs. `ref` with computation using `metric`.
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param ref: pandas Series (m_elements), must have indices, which must match 'features`'s columns
    :param metric: str, {information}
    :return: pandas DataFrame (n_features, 1),
    """
    # Compute score[i] = <features>[i] vs. <ref>
    if metric is 'information_coeff':
        return pd.DataFrame([information_coefficient(ref, row[1]) for row in features.iterrows()],
                            index=features.index, columns=['Information Coeff'])
    elif metric is 'information_cmi_diff':
        return pd.DataFrame([cmi_diff(ref, row[1]) for row in features.iterrows()],
                            index=features.index, columns=['information_cmi_diff'])
    elif metric is 'information_cmi_ratio':
        return pd.DataFrame([cmi_ratio(ref, row[1]) for row in features.iterrows()],
                            index=features.index, columns=['information_cmi_ratio'])
    else:
        raise ValueError('Unknown metric {}.'.format(metric))


def compare_features_against_features(features1, features2, is_distance=False, result_filename=None, figure_filename=None):
    """
    Make association or distance matrix of the rows of `feature1` and `feature2`.
    :param features1: pandas DataFrame,
    :param features2: pandas DataFrame,
    :param is_distance: bool, True for distance and False for association
    :param result_filename: str, filepath to save the result
    :param figure_filename: str, filepath to save the figure
    :return:
    """
    association_matrix = pd.DataFrame(index=features1.index, columns=features2.index, dtype=float)
    features1_nrow = features1.shape[0]
    for i, (i1, r1) in enumerate(features1.iterrows()):
        verbose_print('Features 1 {} ({}/{}) vs. features 2 ...'.format(i1, i + 1, features1_nrow))
        for i2, r2 in features2.iterrows():
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
    :return: dict, NMF result per k (key:k; value:dict(key:w, h, err; value:w matrix, h matrix, and reconstruction error))
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
    :return: dict, NMF result per k (key: k; value: dict (key: w, h, err; value: w matrix, h matrix, and reconstruction error)) and score per k (key:k; value:score)
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

            # Compute the cophenetic correlation coefficient of the hierarchically clustered distances and the normalized assignment distances
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
