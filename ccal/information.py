"""
Computational Cancer Analysis Library v0.1

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov
"""
from numpy import asarray, sign, sum, sqrt, exp, log, isnan
from numpy.random import random_sample
from scipy.stats import pearsonr
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr

from .support import EPS, drop_nan_columns

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


def information_coefficient(x, y, n_grids=25, jitter=1E-10):
    """
    Compute the information correlation between `x` and `y`, which can be either continuous, categorical, or binary
    :param x: vector;
    :param y: vector;
    :param n_grids: int;
    :param jitter: number;
    :return: float;
    """
    x, y = drop_nan_columns([x, y])
    if len(x) < 3 or len(y) < 3:
        return 0
    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Get bandwidths
    cor, p = pearsonr(x, y)
    bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Get P(x, y), P(x), P(y)
    fxy = asarray(kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + EPS
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Get mutual information;
    mi = sum(pxy * log(pxy / (asarray([px] * n_grids).T * asarray([py] * n_grids)))) * dx * dy

    # # Get H(x, y), H(x), and H(y)
    # hxy = - sum(pxy * log(pxy)) * dx * dy
    # hx = -sum(px * log(px)) * dx
    # hy = -sum(py * log(py)) * dy
    # mi = hx + hy - hxy

    # Get information coefficient
    ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic
