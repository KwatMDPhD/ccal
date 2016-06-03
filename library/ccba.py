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
from library.support import *
from library.visualize import *
from library.information import *

# Define global variable
# Path to CCBA dicrectory (repository)
PATH_CCBA = '/Users/Kwat/binf/ccba/'
# Path to testing data directory
PATH_TEST_DATA = os.path.join(PATH_CCBA, 'data', 'test')
SEED = 20121020
TESTING = False


def make_heatmap_panel(dataframe, reference, metrics, columns_to_sort=None, title=None, verbose=False):
    """
    Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>.

    :param
    """
    if not columns_to_sort:
        columns_to_sort = ['IC']

    # Compute score[i] = <dataframe>[i] vs. <reference> and append score as a column to <dataframe>
    if 'IC' in metrics:
        dataframe.ix[:, 'IC'] = pd.Series(
            [information_coefficient(np.array(row[1]), reference) for row in dataframe.iterrows()],
            index=dataframe.index)

    # Sort
    dataframe.sort(columns_to_sort, inplace=True)

    # Plot
    if verbose:
        print('Plotting')
        plot_heatmap_panel(dataframe, reference, metrics, title=title)


# ======================================================================================================================
# NMF
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
            print('Perfomring NMF with k {}'.format(k))
        model = NMF(n_components=k,
                    init=initialization,
                    max_iter=max_iteration,
                    random_state=seed,
                    alpha=regulatizer,
                    shuffle=randomize_coordinate_order)

        # Compute W, H, and reconstruction error
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_
        nmf_results[k] = {'w': w, 'h': h, 'err': err}
    return nmf_results


def score_k(matrix, ks, method=2, verbose=False):
    """
    Select k for NMF.
    """
    nmf_results = nmf(matrix, ks)
    scores = {}

    if method == 2:
        # Intra vs. inter clustering distances
        for k, nmf_result in nmf_results.items():
            if verbose:
                print('Computing clustering score for k={} ...'.format(k))

            # Cluster of a sample is the index with the highest value
            cluster_assignments = np.argmax(nmf_result['H'], axis=0)

            # Cluster
            clusters = {}  # dictionary(key: cluster index; value: samples)
            for cluster_samples in zip(cluster_assignments, matrix):
                if cluster_samples[0] not in clusters:
                    clusters[cluster_samples[0]] = set()
                    clusters[cluster_samples[0]].add(cluster_samples[1])
                else:
                    clusters[cluster_samples[0]].add(cluster_samples[1])

            # Compute score
            samples_scores_for_this_k = np.zeros(nmf_result['H'].shape[1])
            i = 0
            for c, samples in clusters.items():
                for s in samples:
                    # Compute the distance to all samples from the same cluster
                    intra_cluster_distance = []
                    for other_s in samples:
                        if other_s == s:
                            continue
                        else:
                            intra_cluster_distance.append(
                                distance.euclidean(matrix.ix[:, s], matrix.ix[:, other_s]))
                    # Compute the distance to all samples from the other cluster
                    inter_cluster_distance = []
                    for other_c in clusters.keys():
                        if other_c == c:
                            continue
                        else:
                            for other_s in clusters[other_c]:
                                inter_cluster_distance.append(
                                    distance.euclidean((matrix.ix[:, s]), matrix.ix[:, other_s]))
                    score = np.mean(intra_cluster_distance) / np.mean(inter_cluster_distance)
                    if not np.isnan(score):
                        # Add this sample's score
                        samples_scores_for_this_k[i] = score
                    i += 1

            scores[k] = {'mean': samples_scores_for_this_k.mean(), 'std': samples_scores_for_this_k.std()}
    return nmf_results, scores


# ======================================================================================================================
# Onco GPS
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
