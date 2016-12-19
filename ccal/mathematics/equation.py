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

from numpy import sqrt, exp, cumsum, log, argmax, empty
from scipy.special import stdtr
from scipy.stats.distributions import t


def define_exponential_function(x, a, k, c):
    """
    Evaluate specified exponential function at x.
    :param x: array-like; independent variables
    :param a: number; parameter a
    :param k: number; parameter k
    :param c: number; parameter c
    :return: numpy array; (n_independent_variables)
    """

    return a * exp(k * x) + c


def define_skew_t_pdf(x, df, shape, location, scale):
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


def define_cumulative_area_ratio_function(f_1, f_2, x_grids, direction='+'):
    """
    Make a function from f1 and f2.
    :param f_1: array-like;
    :param f_2: array-like;
    :param x_grids: array-like;
    :param direction: str; {'+', '-'}
    :return: array; array
    """

    d_x = x_grids[1] - x_grids[0]

    # Compute d area
    d_area_1 = f_1 / f_1.sum() * d_x
    d_area_2 = f_2 / f_2.sum() * d_x

    # Compute cumulative area
    if direction == '+':  # Forward
        c_area_1 = cumsum(d_area_1)
        c_area_2 = cumsum(d_area_2)
    elif direction == '-':  # Reverse
        c_area_1 = cumsum(d_area_1[::-1])[::-1]
        c_area_2 = cumsum(d_area_2[::-1])[::-1]
    else:
        raise ValueError('Unknown direction {}; choose from (\'+\', \'-\')'.format(direction))

    return log(c_area_1 / c_area_2)


def define_x_coordinates_for_reflection(function, x_grids):
    """
    Make x_grids for getting reflecting PDF.
    :param function: array-like; (1, x_grids.size)
    :param x_grids: array-like; (1, x_grids.size)
    :return: array; (1, x_grids.size)
    """

    pivot_x = x_grids[argmax(function)]

    x_grids_for_reflection = empty(len(x_grids))
    for i, a_x in enumerate(x_grids):

        distance_to_reflecting_x = abs(a_x - pivot_x) * 2

        if a_x < pivot_x:  # Left of the pivot x
            x_grids_for_reflection[i] = a_x + distance_to_reflecting_x

        else:  # Right of the pivot x
            x_grids_for_reflection[i] = a_x - distance_to_reflecting_x

    return x_grids_for_reflection
