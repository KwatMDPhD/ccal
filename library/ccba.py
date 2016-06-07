"""
Cancer Computational Biology Analysis Library v0.1

Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center
"""

import os
from scipy.spatial import distance
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from library.support import *
from library.visualize import *
from library.information import *

# ======================================================================================================================
# Global variables
# ======================================================================================================================
# Path to CCBA dicrectory (repository)
PATH_CCBA = '/Users/Kwat/binf/ccba/'
# Path to testing data directory
PATH_TEST_DATA = os.path.join(PATH_CCBA, 'data', 'test')
SEED = 20121020
TESTING = False


# ======================================================================================================================
# Information functions
# ======================================================================================================================
def make_heatmap_panel(dataframe, reference, metrics, columns_to_sort=None, title=None, verbose=False):
    """
    Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>.

    :param
    """
    if not columns_to_sort:
        columns_to_sort = ['I']

    # Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>
    if 'I' in metrics:
        dataframe.ix[:, 'I'] = pd.Series(
            [information_coefficient(np.array(row[1]), reference) for row in dataframe.iterrows()],
            index=dataframe.index)

    # Sort
    dataframe.sort(columns_to_sort, inplace=True)

    # Plot
    if verbose:
        print('Plotting')
        plot_heatmap_panel(dataframe, reference, metrics, title=title)


# ======================================================================================================================
# NMF functions
# ======================================================================================================================
def nmf(matrix, ks, initialization='random', max_iteration=200, seed=SEED, randomize_coordinate_order=False,
        regulatizer=0, verbose=False):
    """
    Nonenegative matrix mactorize <matrix> with k from <ks>.
    :param matrix:
    :param ks:
    :param initialization: {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}
    :param max_iteration:
    :param seed:
    :param randomize_coordinate_order:
    :param regulatizer:
    :param verbose:
    """
    nmf_results = {}  # dict(key:k; value:dict(key:w, h, err; value:w matrix, h matrix, and reconstruction error))
    for k in ks:
        if verbose:
            print('Perfomring NMF with k {} ...'.format(k))
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
def nmf_and_score_k(matrix, ks, method='cophenetic_correlation', nassignment=100, verbose=False):
    """
    Perform NMF with multiple k and score each computation.
    :param matrix:
    :param ks:
    :param method: {'intra_and_inter_ratio', 'cophenetic_correlation'}.
    :param nassignment: number of assignments used to make <assigment_matrix> when using <cophenetic_correlation>.
    :param verbose:
    :return:
    """
    nrow, ncol = matrix.shape
    nmf_results = scores = {}

    if method == 'intra_and_inter_ratio':
        nmf_results = nmf(matrix, ks)
        for k, nmf_result in nmf_results.items():
            if verbose:
                print('Computing clustering score for k={} using method {} ...'.format(k, method))

            assignments = {}  # dictionary(key: assignemnt index; value: samples)
            # Cluster of a sample is the index with the highest value
            for assigned_sample in zip(np.argmax(nmf_result['H'], axis=0), matrix):
                if assigned_sample[0] not in assignments:
                    assignments[assigned_sample[0]] = set()
                    assignments[assigned_sample[0]].add(assigned_sample[1])
                else:
                    assignments[assigned_sample[0]].add(assigned_sample[1])

            # Compute intra vs. inter clustering distances
            assignment_scores_per_k = np.zeros(nmf_result['H'].shape[1])
            for sidx, a, samples in enumerate(assignments.items()):
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

            scores[k] = {'mean': assignment_scores_per_k.mean(), 'std': assignment_scores_per_k.std()}

    elif method == 'cophenetic_correlation':
        for k in ks:
            if verbose:
                print('Computing clustering score for k={} using method {} ...'.format(k, method))

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

            if verbose:
                print('Computing the cophenetic correlation coefficient ...')
            # Compute the cophenetic correlation coefficient of the hierarchically clustered distances and the normalized assignment distances
            scores[k] = cophenet(linkage(normalized_assignment_distance_matrix, 'average'),
                                 pdist(normalized_assignment_distance_matrix))[0]

    return nmf_results, scores


# ======================================================================================================================
# Onco GPS functions
# ======================================================================================================================
def oncogps_define_state(verbose=False):
    """
    Compute the OncoGPS states by consensus clustering.
    """


def oncogps_map(verbose=False):
    """
    Map OncoGPS.
    """


def oncogps_populate_map(verbose=False):
    """
    """
