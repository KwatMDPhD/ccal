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

import rpy2.robjects as ro
from numpy import finfo, asarray, sign, sqrt, exp, log, isnan
from numpy.random import seed, random_sample
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr

from .. import RANDOM_SEED
from ..support.d2 import drop_nan_columns

EPS = finfo(float).eps

ro.conversion.py2ri = numpy2ri

mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


def information_coefficient(x, y, n_grids=25, jitter=1E-10, random_seed=RANDOM_SEED):
    """
    Compute the information coefficient between x and y, which can be composed of either continuous,
    categorical, or binary values.
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grid lines in a dimention when estimating bandwidths
    :param jitter: number;
    :param random_seed: int or array-like;
    :return: float;
    """

    # Can't work with missing any value
    # not_nan_filter = ~isnan(x)
    # not_nan_filter &= ~isnan(y)
    # x = x[not_nan_filter]
    # y = y[not_nan_filter]
    x, y = drop_nan_columns([x, y])

    # Need at least 3 values to compute bandwidth
    if len(x) < 3 or len(y) < 3:
        return 0

    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)

    # Add jitter
    seed(random_seed)
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Compute bandwidths
    cor, p = pearsonr(x, y)
    bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Compute P(x, y), P(x), P(y)
    fxy = asarray(kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + EPS
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Compute mutual information;
    mi = (pxy * log(pxy / (asarray([px] * n_grids).T * asarray([py] * n_grids)))).sum() * dx * dy

    # # Get H(x, y), H(x), and H(y)
    # hxy = - (pxy * log(pxy)).sum() * dx * dy
    # hx = -(px * log(px)).sum() * dx
    # hy = -(py * log(py)).sum() * dy
    # mi = hx + hy - hxy

    # Compute information coefficient
    ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic
