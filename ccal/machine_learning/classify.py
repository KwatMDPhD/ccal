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

from .. import RANDOM_SEED


def classify(training,
             training_classes,
             testing,
             c=1.0,
             kernel='rbf',
             degree=3,
             gamma='auto',
             coef0=0.0,
             shrinking=True,
             probability=False,
             tol=0.001,
             cache_size=200,
             class_weight=None,
             verbose=False,
             max_iter=-1,
             decision_function_shape=None,
             random_state=RANDOM_SEED):
    """

    Train a classifier using training and predict the classes of testing.
    :param training: array-like; (n_training_samples, n_dimensions)
    :param training_classes: array-like; (1, n_training_samples)
    :param testing: array-like; (n_testing_samples, n_dimensions)
    :param c:
    :param kernel:
    :param degree:
    :param gamma:
    :param coef0:
    :param shrinking:
    :param probability:
    :param tol:
    :param cache_size:
    :param class_weight:
    :param verbose:
    :param max_iter:
    :param decision_function_shape:
    :param random_state:
    :return: n_samples; array-like; (1, n_testing_samples)
    """

    clf = SVC(C=c,
              kernel=kernel,
              degree=degree,
              gamma=gamma,
              coef0=coef0,
              shrinking=shrinking,
              probability=probability,
              tol=tol,
              cache_size=cache_size,
              class_weight=class_weight,
              verbose=verbose,
              max_iter=max_iter,
              decision_function_shape=decision_function_shape,
              random_state=random_state)
    clf.fit(asarray(training), asarray(training_classes))
    return clf.predict(asarray(testing))


def regress(training,
            training_classes,
            testing,
            kernel='rbf',
            degree=3,
            gamma='auto',
            coef0=0.0,
            tol=0.001,
            c=1.0,
            epsilon=0.1,
            shrinking=True,
            cache_size=200,
            verbose=False,
            max_iter=-1):
    """

    Train a classifier using training and predict the classes of testing.
    :param training: array-like; (n_training_samples, n_dimensions)
    :param training_classes: array-like; (1, n_training_samples)
    :param testing: array-like; (n_testing_samples, n_dimensions)
    :param kernel:
    :param degree:
    :param gamma:
    :param coef0:
    :param tol:
    :param c:
    :param epsilon:
    :param shrinking:
    :param cache_size:
    :param verbose:
    :param max_iter:
    :return: n_samples; array-like; (1, n_testing_samples)
    """

    clf = SVR(kernel=kernel,
              degree=degree,
              gamma=gamma,
              coef0=coef0,
              tol=tol,
              C=c,
              epsilon=epsilon,
              shrinking=shrinking,
              cache_size=cache_size,
              verbose=verbose,
              max_iter=max_iter)
    clf.fit(asarray(training), asarray(training_classes))
    return clf.predict(asarray(testing))
