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

from numpy import array, sort
from scipy.optimize import curve_fit



def fit_matrix(matrix, function_to_fit, axis=0, sort_matrix=False, maxfev=1000):
    """
    Fit rows or columns of matrix to function_to_fit.
    :param matrix: pandas DataFrame;
    :param function_to_fit: function;
    :param axis: int;
    :param sort_matrix: bool;
    :param maxfev: int;
    :return: list; fit parameters
    """

    # Copy
    matrix = array(matrix)

    if axis == 1:  # Transpose
        matrix = matrix.T

    if sort_matrix:  # Sort by column
        matrix = sort(matrix, axis=0)

    x = array(range(matrix.shape[0]))
    y = matrix.sum(axis=1) / matrix.shape[1]
    fit_parameters = curve_fit(function_to_fit, x, y, maxfev=maxfev)[0]

    return fit_parameters
