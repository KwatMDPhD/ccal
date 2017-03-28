"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from numpy import argmax, asarray, zeros
from numpy.random import random_integers, seed
from pandas import DataFrame, read_csv
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

from .. import RANDOM_SEED
from ..machine_learning.matrix_decompose import nmf
from ..mathematics.information import information_coefficient
from ..support.log import print_log
from ..support.parallel_computing import parallelize
from .score import compute_similarity_matrix


# ==============================================================================
# Hierarchical consensus cluster
# ==============================================================================
def hierarchical_consensus_cluster(matrix,
                                   ks,
                                   d=None,
                                   n_jobs=1,
                                   function=information_coefficient,
                                   n_clusterings=100,
                                   random_seed=RANDOM_SEED):
    """
    Consensus cluster matrix's columns into k clusters.
    :param matrix: DataFrame; (n_features, m_samples)
    :param ks: iterable; list of ks used for clustering
    :param d: str or DataFrame; sample-distance matrix
    :param n_jobs; int;
    :param function: function; distance function
    :param n_clusterings: int; number of clusterings for the consensus 
    clustering
    :param random_seed: int;
    :return: DataFrame and Series; assignment matrix (n_ks, n_samples) and 
    cophenetic correlation coefficients (n_ks)
    """

    if isinstance(ks, int):
        ks = [ks]

    if isinstance(d, DataFrame):
        print_log('Loading precomputed sample-distance matrix ...')
        if isinstance(d, str):
            d = read_csv(d, sep='\t', index_col=0)
    else:
        # Compute sample-distance matrix
        print_log('Computing sample-distance matrix ...')
        d = compute_similarity_matrix(matrix, matrix, function,
                                      is_distance=True)

    # Consensus cluster distance matrix
    print_log('{} consensus clusterings ...'.format(n_clusterings))

    cs = DataFrame(index=ks, columns=list(matrix.columns))
    cs.index.name = 'K'

    cccs = {}

    for k in ks:
        print_log('K={} ...'.format(k))

        # For n_clusterings times, permute distance matrix with repeat,
        # and cluster

        # Make sample x clustering matrix
        sample_x_clustering = DataFrame(
            index=matrix.columns, columns=range(n_clusterings))
        seed(random_seed)
        for i in range(n_clusterings):
            if i % 10 == 0:
                print_log(
                    '\tPermuting sample-distance matrix with repeat and '
                    'clustering ({}/{}) ...'.
                        format(i, n_clusterings))

            # Randomize samples with repeat and cluster
            hc = AgglomerativeClustering(n_clusters=k)
            is_ = random_integers(0, matrix.shape[1] - 1, matrix.shape[1])
            hc.fit(d.iloc[is_, is_])

            # Assign cluster labels to the random samples
            sample_x_clustering.iloc[is_, i] = hc.labels_

        # Make consensus matrix using labels created by clusterings of
        # randomized distance matrix
        print_log(
            '\tMaking consensus matrix from {} '
            'randomized-sample-distance-matrix hierarchical clusterings...'.
                format(n_clusterings))
        consensus_matrix = _get_consensus(sample_x_clustering)

        # Hierarchical cluster consensus_matrix's distance matrix and compute
        #  cophenetic correlation coefficient
        hc, ccc = _hierarchical_cluster_consensus_matrix(consensus_matrix)
        # Get labels from hierarchical clustering
        cs.ix[k, :] = fcluster(hc, k, criterion='maxclust')
        # Save cophenetic correlation coefficients
        cccs[k] = ccc

    return d, cs, cccs


def _hierarchical_cluster_consensus_matrix(consensus_matrix,
                                           force_diagonal=True,
                                           method='ward'):
    """
    Hierarchical cluster consensus_matrix and compute cophenetic correlation 
    coefficient.
    Convert consensus_matrix into distance matrix. Hierarchical cluster the 
    distance matrix. And compute the
    cophenetic correlation coefficient.
    :param consensus_matrix: DataFrame;
    :param force_diagonal: bool;
    :param method: str; method parameter for scipy.cluster.hierarchy.linkage
    :return: ndarray and float; linkage (Z) and cophenetic correlation 
    coefficient
    """

    # Convert consensus matrix into distance matrix
    distance_matrix = 1 - consensus_matrix

    if force_diagonal:
        for i in range(distance_matrix.shape[0]):
            distance_matrix.iloc[i, i] = 0

    # Cluster consensus matrix to assign the final label
    hc = linkage(consensus_matrix, method=method)

    # Compute cophenetic correlation coefficient
    ccc = pearsonr(pdist(distance_matrix), cophenet(hc))[0]

    return hc, ccc


# ==============================================================================
# NMF consensus cluster
# ==============================================================================
def nmf_consensus_cluster(matrix,
                          ks,
                          n_jobs=1,
                          n_clusterings=100,
                          algorithm='Lee & Seung',
                          init=None,
                          solver='cd',
                          tol=1e-7,
                          max_iter=1000,
                          random_seed=RANDOM_SEED,
                          alpha=0.0,
                          l1_ratio=0.0,
                          verbose=0,
                          shuffle_=False,
                          nls_max_iter=2000,
                          sparseness=None,
                          beta=1,
                          eta=0.1):
    """
    Perform NMF with k from ks and score each NMF decomposition.

    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); 
    the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param n_jobs: int;
    :param n_clusterings: int;

    :param algorithm: str; 'Alternating Least Squares' or 'Lee & Seung'

    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_seed:
    :param alpha:
    :param l1_ratio:
    :param verbose:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:

    :return: dict; {k: {
                        w: W matrix (n_rows, k),
                        h: H matrix (k, n_columns),
                        e: Reconstruction Error,
                        ccc: Cophenetic Correlation Coefficient
                        }
                    }
    """

    if isinstance(ks, int):
        ks = [ks]

    nmfs = {}

    print_log(
        'Computing cophenetic correlation coefficient of {} NMF consensus '
        'clusterings (n_jobs={}) ...'.
            format(n_clusterings, n_jobs))

    args = [[
        matrix, k, n_clusterings, algorithm, init, solver, tol, max_iter,
        random_seed, alpha, l1_ratio, verbose, shuffle_, nls_max_iter,
        sparseness, beta, eta
    ] for k in ks]

    for nmf_ in parallelize(_nmf_and_score, args, n_jobs=n_jobs):
        nmfs.update(nmf_)

    return nmfs


def _nmf_and_score(args):
    """
    NMF and score using 1 k.
    :param args:
    :return: dict; {k: {
                        w: W matrix (n_rows, k),
                        h: H matrix (k, n_columns),
                        e: Reconstruction Error,
                        ccc: Cophenetic Correlation Coefficient
                        }
                    }
    """

    matrix, k, n_clusterings, algorithm, init, solver, tol, max_iter, \
    random_seed, alpha, l1_ratio, verbose, shuffle_, nls_max_iter, \
    sparseness, beta, eta = args

    print_log('NMF and scoring k={} ...'.format(k))

    # NMF cluster n_clustering
    sample_x_clustering = DataFrame(
        index=matrix.columns, columns=range(n_clusterings), dtype=int)

    # Save the 1st NMF decomposition for each k
    nmfs = {}

    for i in range(n_clusterings):
        if i % 10 == 0:
            print_log('\t(k={}) NMF ({}/{}) ...'.format(k, i, n_clusterings))

        # NMF
        nmf_ = nmf(matrix,
                   k,
                   algorithm=algorithm,
                   init=init,
                   solver=solver,
                   tol=tol,
                   max_iter=max_iter,
                   random_seed=random_seed + i,
                   alpha=alpha,
                   l1_ratio=l1_ratio,
                   verbose=verbose,
                   shuffle_=shuffle_,
                   nls_max_iter=nls_max_iter,
                   sparseness=sparseness,
                   beta=beta,
                   eta=eta)[k]

        # Save the 1st NMF decomposition for each k
        if i == 0:
            nmfs[k] = nmf_
            print_log('\t\t(k={}) Saved the 1st NMF decomposition.'.format(k))

        # Column labels are the row index holding the highest value
        sample_x_clustering.iloc[:, i] = argmax(asarray(nmf_['h']), axis=0)

    # Make consensus matrix using NMF labels
    print_log('\t(k={}) Making consensus matrix from {} NMF clusterings ...'.
              format(k, n_clusterings))
    consensus_matrix = _get_consensus(sample_x_clustering)

    # Hierarchical cluster consensus_matrix's distance matrix and compute
    # cophenetic correlation coefficient
    hierarchical_clustering, cophenetic_correlation_coefficient = \
        _hierarchical_cluster_consensus_matrix(
            consensus_matrix)
    nmfs[k]['ccc'] = cophenetic_correlation_coefficient

    return nmfs


# ==============================================================================
# Consensus
# ==============================================================================
def _get_consensus(sample_x_clustering):
    """
    Count number of co-clusterings.
    :param sample_x_clustering: DataFrame; (n_samples, n_clusterings)
    :return: DataFrame; (n_samples, n_samples)
    """

    sample_x_clustering_array = asarray(sample_x_clustering)

    n_samples, n_clusterings = sample_x_clustering_array.shape

    # Make sample x sample matrix
    coclusterings = zeros((n_samples, n_samples))

    # Count the number of co-clusterings
    for i in range(n_samples):
        for j in range(n_samples):
            for c_i in range(n_clusterings):
                v1 = sample_x_clustering_array[i, c_i]
                v2 = sample_x_clustering_array[j, c_i]
                if v1 and v2 and (v1 == v2):
                    coclusterings[i, j] += 1

    # Normalize by the number of clusterings and return
    coclusterings /= n_clusterings

    return DataFrame(
        coclusterings,
        index=sample_x_clustering.index,
        columns=sample_x_clustering.index)
