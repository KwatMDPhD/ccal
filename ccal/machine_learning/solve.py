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

from numpy import dot
from numpy.linalg import pinv
from pandas import DataFrame
from scipy.optimize import nnls


def solve_matrix_linear_equation(a, b, method='nnls'):
    """
    Solve a * x = b of (n, k) * (k, m) = (n, m).
    :param a: numpy array; (n, k)
    :param b: numpy array; (n, m)
    :param method: str; {'nnls', 'pinv'}
    :return: numpy array; (k, m)
    """

    if method == 'nnls':
        x = DataFrame(index=a.columns, columns=b.columns)
        for i in range(b.shape[1]):
            x.iloc[:, i] = nnls(a, b.iloc[:, i])[0]

    elif method == 'pinv':
        a_pinv = pinv(a)
        x = dot(a_pinv, b)
        x[x < 0] = 0
        x = DataFrame(x, index=a.columns, columns=b.columns)

    else:
        raise ValueError('Unknown method {}. Choose from [\'nnls\', \'pinv\']'.
                         format(method))

    return x
