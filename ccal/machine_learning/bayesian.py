import numpy as np
import pandas as pd
from sklearn import BaseEstimator, ClassifierMixin, LogisticRegression


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
        feature_evidence = {
            k: pd.DataFrame(index=x.index, columns=x.columns)
            for k in self.classes_
        }
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
            posterior_probs = posterior_probs.divide(
                posterior_probs.sum(axis=1), axis='index')
        return posterior_probs

    def predict(self, x):
        posterior_probs = self.predict_proba(x)
        max_idxs = np.argmax(posterior_probs.values, axis=1)
        return pd.Series(self.classes_[max_idxs], index=x.index)
