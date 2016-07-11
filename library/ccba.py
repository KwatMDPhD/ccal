"""
Computational Cancer Biology Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Biology, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Biology, UCSD Cancer Center


Description:
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.decomposition import NMF

from support import verbose_print
from information import information_coefficient

# ======================================================================================================================
# Global variables
# ======================================================================================================================
# Path to testing data directory
PATH_TEST_DATA = os.path.join('data', 'test')

SEED = 20121020

TESTING = False


# ======================================================================================================================
# Information functions
# ======================================================================================================================
def compute_against_reference(dataframe, reference, metric, columns_to_sort=None):
    """
    Compute score[i] = `dataframe`[i] vs. `reference` and append score as a column to `dataframe`.
    :param dataframe: pandas DataFrame,
    :param reference: array-like,
    :param metric: str, {'information'}
    :param columns_to_sort: str, column name
    :return: None, modify `dataframe` inplace
    """
    if not columns_to_sort:
        columns_to_sort = ['information']

    # Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>
    if 'information' in metric:
        dataframe.ix[:, 'information'] = pd.Series(
            [information_coefficient(np.array(row[1]), reference) for row in dataframe.iterrows()],
            index=dataframe.index)
    else:
        raise ValueError('Unknown metric {}'.format(metric))

    # Sort
    dataframe.sort(columns_to_sort, inplace=True)


# ======================================================================================================================
# NMF functions
# ======================================================================================================================
def nmf(matrix, ks, initialization='random', max_iteration=200, seed=SEED, randomize_coordinate_order=False,
        regulatizer=0):
    """
    Nonenegative matrix factorize `matrix` with k from `ks`.
    :param matrix:
    :param ks:
    :param initialization: {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration:
    :param seed:
    :param randomize_coordinate_order:
    :param regulatizer:
    :return: dict, NMF result per k (key:k; value:dict(key:w, h, err; value:w matrix, h matrix, and reconstruction error))
    """
    nmf_results = {}  # dict (key:k; value:dict (key:w, h, err; value:w matrix, h matrix, and reconstruction error))
    for k in ks:
        verbose_print('Perfomring NMF with k {} ...'.format(k))
        model = NMF(n_components=k,
                    init=initialization,
                    max_iter=max_iteration,
                    random_state=seed,
                    alpha=regulatizer,
                    shuffle=randomize_coordinate_order)

        # Compute W, H, and reconstruction error
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_
        nmf_results[k] = {'W': w, 'H': h, 'ERROR': err}

    return nmf_results


# TODO: set the default nassignemnt
def nmf_and_score(matrix, ks, method='cophenetic_correlation', nassignment=100, verbose=False):
    """
    Perform NMF with multiple k and score each computation.
    :param matrix:
    :param ks:
    :param method: {'intra_inter_ratio', 'cophenetic_correlation'}.
    :param nassignment: number of assignments used to make <assigment_matrix> when using <cophenetic_correlation>.
    :param verbose:
    :return: Dictionaries of NMF result per k (key:k; value:dict (key:w, h, err; value:w matrix, h matrix, and reconstruction error))
                and score per k (key:k; value:score)
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
