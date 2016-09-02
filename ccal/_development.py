import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin


# ======================================================================================================================
# NMF
# ======================================================================================================================
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
