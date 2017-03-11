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

from numpy import average
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


def cross_validate(model, data, target, n_partitions):
    """
    Cross-validata.
    :param model:
    :param data:
    :param target:
    :param n_partitions:
    :return:
    """

    # Initialize indexes for cross validation folds
    folds = KFold(len(data), n_partitions)

    # List to keep cross validation scores
    scores = []

    # For each fold
    for k, (train_index, test_index) in enumerate(folds):
        # Partition training and testing data sets
        x_train, x_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Learn
        model.fit(x_train, y_train.iloc[:, 0])

        # Score the learned fit
        score = pearsonr(model.predict(x_test), y_test.iloc[:, 0])
        scores.append(score)

    return average([s[0] for s in scores]), average([s[1] for s in scores])
