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

from numpy import asarray
from sklearn.svm import SVC, SVR


def classify(training, training_classes, testing, kernel=None):
    """
    Train a classifier using training and predict the classes of testing.
    :param training: array-like; (n_training_samples, n_dimensions)
    :param training_classes: array-like; (1, n_training_samples)
    :param testing: array-like; (n_testing_samples, n_dimensions)
    :return: n_samples; array-like; (1, n_testing_samples)
    """

    clf = SVC(kernel=kernel)
    clf.fit(asarray(training), asarray(training_classes))
    return clf.predict(asarray(testing))


def regress(training, training_classes, testing, kernel=None):
    """
    Train a classifier using training and predict the classes of testing.
    :param training: array-like; (n_training_samples, n_dimensions)
    :param training_classes: array-like; (1, n_training_samples)
    :param testing: array-like; (n_testing_samples, n_dimensions)
    :return: n_samples; array-like; (1, n_testing_samples)
    """

    clf = SVR(kernel=kernel)
    clf.fit(asarray(training), asarray(training_classes))
    return clf.predict(asarray(testing))
