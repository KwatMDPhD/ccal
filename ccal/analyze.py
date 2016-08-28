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
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.decomposition import NMF
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from .support import SEED, print_log, establish_path, write_gct, normalize_pandas_object, compare_matrices
from .visualize import plot_nmf_result, plot_features_against_reference
from .information import information_coefficient, cmi_diff, cmi_ratio


# ======================================================================================================================
# Feature selection
# ======================================================================================================================
def rank_features_against_reference(features, ref, features_type='continuous', ref_type='continuous',
                                    features_ascending=False, ref_ascending=False, ref_sort=True,
                                    metric='information_coef', nfeatures=0.95, nsampling=30, confidence=0.95, nperm=30,
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

    scores = compute_against_reference(features, ref, metric=metric, nfeatures=nfeatures, ascending=features_ascending,
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
    if nfeatures < 1:
        above_quantile = scores.ix[:, metric] >= scores.ix[:, metric].quantile(nfeatures)
        print_log('Plotting {} features vs. reference > {} quantile ...'.format(sum(above_quantile), nfeatures))
        below_quantile = scores.ix[:, metric] <= scores.ix[:, metric].quantile(1 - nfeatures)
        print_log('Plotting {} features vs. reference < {} quantile ...'.format(sum(below_quantile), 1 - nfeatures))
        indices_to_plot = features.index[above_quantile | below_quantile].tolist()
    else:
        indices_to_plot = features.index[:nfeatures].tolist() + features.index[-nfeatures:].tolist()
        print_log('Plotting top & bottom {} features vs. reference ...'.format(len(indices_to_plot)))

    plot_features_against_reference(features.ix[indices_to_plot, :], ref, annotations.ix[indices_to_plot, :],
                                    features_type=features_type, ref_type=ref_type, title=title, title_size=title_size,
                                    annotation_header=' ' * 7 + 'IC(\u0394)' + ' ' * 9 + 'P-val' + ' ' * 4 + 'FDR',
                                    annotation_label_size=annotation_label_size,
                                    plot_colname=plot_colname, figure_filename=figure_filename)


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
    scores = pd.DataFrame(scores, index=features.index, columns=[metric]).sort_values(metric)

    print_log('Bootstrapping to get {} confidence interval ...'.format(confidence))
    nsample = math.ceil(0.632 * features.shape[1])
    if nsampling < 2:
        print_log('Not bootstrapping because number of sampling < 3.')
    elif nsample < 3:
        print_log('Not bootstrapping because 0.632 * number of sample < 3.')
    else:
        # Limit features to be bootstrapped
        if nfeatures < 1:
            above_quantile = scores.ix[:, metric] >= scores.ix[:, metric].quantile(nfeatures)
            print_log('Bootstrapping {} features vs. reference > {} quantile ...'.format(sum(above_quantile),
                                                                                         nfeatures))
            below_quantile = scores.ix[:, metric] <= scores.ix[:, metric].quantile(1 - nfeatures)
            print_log('Bootstrapping {} features vs. reference < {} quantile ...'.format(sum(below_quantile),
                                                                                         1 - nfeatures))
            indices_to_bootstrap = scores.index[above_quantile | below_quantile].tolist()
        else:
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


# ======================================================================================================================
# NMF
# ======================================================================================================================
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


def nmf_bcv(x, nmf, nfold=2, nrepeat=1):
    """
    Bi-crossvalidation of NMF as in Owen and Perry (2009).
    Note that this implementation does not require the intermediates to be non-negative. Details of how to add this
    constraint can be found on page 11 (beginning of section 5) of Owen and Perry (2009); the authors did not seem to
    consider it especially important for quality of model selection.
    :param x: data array to be decomposed, (nsamples, nfeatures)
    :param nmf: sklearn NMF object, already initialized
    :param nfold: number of folds for cross-validation (O&P suggest 2)
    :param nrepeat: how many times to repeat, to average out variation based on which rows and columns were held out
    :return: mean_error, mean mse across nrepeat
    """
    errors = []
    for rep in range(nrepeat):
        kf_rows = KFold(x.shape[0], nfold, shuffle=True)
        kf_cols = KFold(x.shape[1], nfold, shuffle=True)
        for row_train, row_test in kf_rows:
            for col_train, col_test in kf_cols:
                a = x[row_test][:, col_test]
                base_error = mean_squared_error(a, np.zeros(a.shape))
                b = x[row_test][:, col_train]
                c = x[row_train][:, col_test]
                d = x[row_train][:, col_train]
                nmf.fit(d)
                hd = nmf.components_
                wd = nmf.transform(d)
                wa = np.dot(b, hd.T)
                ha = np.dot(wd.T, c)
                a_prime = np.dot(wa, ha)
                a_notzero = a != 0
                scaling_factor = np.mean(np.divide(a_prime, a)[a_notzero])
                scaled_a_prime = a_prime / scaling_factor
                error = mean_squared_error(a, scaled_a_prime) / base_error
                errors.append(error)
    mean_error = np.mean(errors)
    return mean_error


# ======================================================================================================================
# Onco GPS functions
# ======================================================================================================================
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
            labels.to_csv(filename_prefix + '_labels', sep='\t')
            write_gct(memberships, filename_prefix + '_memberships.gct')
    else:
        raise ValueError('No number of clusters passed.')

    return labels.iloc[:, :-1], memberships, labels.iloc[:, -1:]


# ======================================================================================================================
# GSEA functions
# ======================================================================================================================
def ssgsea(exp_data, sets_to_genes, alpha=0.25):
    """
    Single-sample GSEA as described in Barbie et al. (2009)
    :param exp_df: Pandas DataFrame or Series of expression values, (n_samples, n_genes) or (n_genes,)
    :param sets_to_genes: dictionary with set names as keys and sets of genes as values, e.g. {'set1': {'g1', 'g2'}}
    :param alpha: weighting factor
    :return: Pandas DataFrame or Series of expression projected onto gene sets
    """
    if isinstance(exp_data, pd.Series):
        return ssgsea_per_sample(exp_data, sets_to_genes, alpha=alpha)
    elif isinstance(exp_data, pd.DataFrame):
        return exp_data.apply(ssgsea_per_sample, axis=1, args=(sets_to_genes, alpha))
    else:
        raise ValueError("exp_data must be Pandas DataFrame or Series")


def ssgsea_per_sample(exp_series, sets_to_genes, alpha=0.25):
    sorted_exp_series = exp_series.sort_values(ascending=False)
    enrichment_scores = _base_gsea(sorted_exp_series.index, sets_to_genes, collect_func=np.sum, alpha=alpha)
    return enrichment_scores


def max_abs(x):
    return x[np.argmax(np.abs(x))]


def gsea(ranked_genes, sets_to_genes, alpha=0.25):
    enrichment_scores = _base_gsea(ranked_genes, sets_to_genes, collect_func=max_abs, alpha=alpha)
    return enrichment_scores


# From the itertools recipes. Should it go in support.py?
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _base_gsea(ranked_genes, sets_to_genes, collect_func, alpha=0.25):
    """
    Basic idea:
    -make weighted ecdf for hits
    -make ecdf for misses
    -take elementwise difference
    -collect_func() to get result (max_abs() for gsea, np.sum() for ssgsea)

    Might still be able to speed up using array funcs rather than iterating?
    """
    n_genes = len(ranked_genes)
    ranks = list(range(n_genes))
    gene_to_rank = dict(zip(ranked_genes, ranks))
    enrichment_scores = {}
    for set_name, set_genes in sets_to_genes.items():
        ranked_set_genes = [gene for gene in set_genes if gene in gene_to_rank]
        n_non_set_genes = float(n_genes - len(ranked_set_genes))
        hit_ranks = [gene_to_rank[gene] for gene in ranked_set_genes]
        misses = np.ones_like(ranks)
        misses[hit_ranks] = 0
        cum_misses = np.cumsum(misses)
        miss_ecdf = cum_misses / n_non_set_genes
        cum_hits = np.zeros_like(ranks)
        if len(hit_ranks) > 0:
            cum_hit_sum = 0
            sorted_hit_ranks = sorted(hit_ranks)
            # add one so ranks to weight start from 1, not zero
            # however, it's convenient to start at zero otherwise so I can index using the ranks
            weighted_ranks = (np.array(sorted_hit_ranks) + 1) ** alpha
            hit_rank_pairs = list(pairwise(sorted_hit_ranks))  # given [a, b, c, d] yields (a, b), (b, c), (c, d)
            for i, (idx1, idx2) in enumerate(hit_rank_pairs):
                cum_hit_sum += weighted_ranks[i]
                cum_hits[idx1:idx2] = cum_hit_sum
            cum_hit_sum += weighted_ranks[-1]
            cum_hits[sorted_hit_ranks[-1]:] = cum_hit_sum
            weighted_hit_ecdf = cum_hits / cum_hit_sum
        else:
            weighted_hit_ecdf = cum_hits  # still np.zeros_like(ranks)
        ecdf_dif = np.subtract(weighted_hit_ecdf, miss_ecdf)
        enrichment_score = collect_func(ecdf_dif)
        enrichment_scores[set_name] = enrichment_score
    return pd.Series(enrichment_scores)


# ======================================================================================================================
# Bayesian classifier
# ======================================================================================================================
class BayesianClassifier(BaseEstimator, ClassifierMixin):
    """
    Note: still differs from Pablo's R version, so it needs fixing, but hopefully it's a headstart.

    Similar to a Naive Bayes classifier
    Using the assumption of independence of features, it fits a model for each feature a combines them.
    This is done separately for each class, i.e. it fits multiple one-vs-all models in the multiclass case.
    The independence assumption allows for more transparent interpretation at some cost of performance.

    Note that test data should be scaled the same way as training data for meaningful results.
    """

    def __init__(self):
        self.regressions_ = None
        self.classes_ = None
        self.priors_ = None
        self.prior_log_odds_ = None

    def fit(self, x, y):
        """
        :param x: Pandas DataFrame, (n_samples, n_features)
        :param y: Pandas Series, (n_samples,)
        :return: self
        """
        self.classes_ = np.array(sorted(set(y.values)))
        self.priors_ = y.value_counts().loc[self.classes_] / len(y)
        self.prior_log_odds_ = np.log(self.priors_ / (1 - self.priors_))
        self.regressions_ = dict()
        for k in self.classes_:
            self.regressions_[k] = dict()
            y_one_v_all = y.copy()
            y_one_v_all[y != k] = 0
            y_one_v_all[y == k] = 1
            for feature in x.columns:
                logreg = LogisticRegression()
                subdf = x.loc[:, [feature]]
                logreg.fit(subdf, y_one_v_all)
                self.regressions_[k][feature] = logreg
        return self

    def predict_proba(self, x, normalize=True, return_all=False):
        prior_evidence = pd.Series(index=self.classes_)
        log_odds = pd.DataFrame(index=x.index, columns=self.classes_)
        feature_evidence = {k: pd.DataFrame(index=x.index, columns=x.columns) for k in self.classes_}
        for k in self.classes_:
            prior = self.priors_.loc[k]
            prior_odds = prior / (1 - prior)
            prior_log_odds = np.log(prior_odds)
            log_odds.loc[:, k] = prior_log_odds
            prior_evidence.loc[k] = prior_log_odds
            for feature in x.columns:
                logreg = self.regressions_[k][feature]
                subdf = x.loc[:, [feature]]
                class_index = list(logreg.classes_).index(1)
                probs = logreg.predict_proba(subdf)[:, class_index]
                odds = probs / (1 - probs)
                evidence = np.log(odds / prior_odds)
                feature_evidence[k].loc[:, feature] = evidence
                log_odds.loc[:, k] += evidence
        posterior_probs = np.exp(log_odds) / (np.exp(log_odds) + 1)
        if return_all:
            return posterior_probs, feature_evidence
        if normalize:
            posterior_probs = posterior_probs.divide(posterior_probs.sum(axis=1), axis='index')
        return posterior_probs

    def predict(self, x):
        posterior_probs = self.predict_proba(x)
        max_idxs = np.argmax(posterior_probs.values, axis=1)
        return pd.Series(self.classes_[max_idxs], index=x.index)
