"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from numpy import sqrt, exp
from scipy.stats.distributions import t
from scipy.special import stdtr


def exponential_f(x, a, k, c):
    """
    Evaluate specified exponential function at x.
    :param x: array-like; independent variables
    :param a: number; parameter a
    :param k: number; parameter k
    :param c: number; parameter c
    :return: numpy array; (n_independent_variables)
    """

    return a * exp(k * x) + c


def skew_t_pdf(x, df, shape, location, scale):
    """
    Evaluate skew-t PDF (defined by `df`, `shape`, `location`, and `scale`) at `x`.
    :param x: array-like; vector of independent variables used to compute probabilities of the skew-t PDF.
    :param df: number; degree of freedom of the skew-t PDF
    :param shape: number; skewness or shape parameter of the skew-t PDF
    :param location: number; location of the skew-t PDF
    :param scale: number; scale of the skew-t PDF
    :return array-like: skew-t PDF (defined by `df`, `shape`, `location`, and `scale`) evaluated at `x`.
    """

    return (2 / scale) * t._pdf(((x - location) / scale), df) * stdtr(df + 1, shape * ((x - location) / scale) * sqrt(
        (df + 1) / (df + x ** 2)))
