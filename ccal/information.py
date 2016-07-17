"""
Computational Cancer Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center


Description:
Information computing module for CCAL.
"""
import math

import numpy as np
from scipy.stats import pearsonr
from statsmodels.nonparametric.kernel_density import KDEMultivariate

# TODO
# For bcv():
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri

ro.conversion.py2ri = numpy2ri
from rpy2.robjects.packages import importr

mass = importr('MASS')


def rbcv(x):
    """
    TODO
    :param x: array-like, (n_samples,)
    :return: float, bandwidth
    """
    bandwidth = np.array(mass.bcv(x))[0]
    return bandwidth


# TODO: understand the math
def mutual_information(x, y, var_type='cc', n_grid=25, bandwidth_scaling=None):
    """
    Compute mutual information between `x` and `y`:
    Difference in bandwidth convention means bcv() delta must be divided by 4.
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_type: character string, 'c' (continuous) or 'd' (discrete)
    :param bandwidth_scaling: float
    :return: float, mutual information
    """
    x_not_nan = np.logical_not(np.isnan(x))
    y_not_nan = np.logical_not(np.isnan(y))
    xy_not_nan = x_not_nan & y_not_nan

    jitter_scale = 1E-10
    x_jitter, y_jitter = [jitter_scale * np.random.uniform(size=xy_not_nan.sum()) for _ in [x, y]]

    x = x[xy_not_nan] + x_jitter
    y = y[xy_not_nan] + y_jitter
    xy = [x, y]

    x_grid = np.linspace(x.min(), x.max(), n_grid)
    y_grid = np.linspace(y.min(), y.max(), n_grid)
    xg, yg = np.meshgrid(x_grid, y_grid)
    grid = np.vstack([xg.flatten(), yg.flatten()])

    delta = np.array([rbcv(z) for z in [x, y]]).reshape((2,)) / 4

    if bandwidth_scaling:
        delta *= bandwidth_scaling
    kde = KDEMultivariate(xy, bw=delta, var_type=var_type)

    fxy = kde.pdf(grid).reshape((n_grid, n_grid)).T + np.finfo(float).eps
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=0) * dy
    py = pxy.sum(axis=1) * dx

    pxg, pyg = np.meshgrid(px, py)

    mi = np.sum(pxy * np.log(pxy / (pxg * pyg))) * dx * dy
    return mi


def information_coefficient(x, y, var_type='cc', n_grid=25):
    """
    Compute information coefficient between `x` and `y`.
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_type: character string, 'c' (continuous) or 'd' (discrete)
    :return: float, information coefficient
    """
    rho, p = pearsonr(x, y)
    rho2 = abs(rho)
    bandwidth_scaling = (1 + (-0.75) * rho2)
    mi = mutual_information(x, y, var_type=var_type, n_grid=n_grid, bandwidth_scaling=bandwidth_scaling)
    ic = np.sign(rho) * np.sqrt(1 - np.exp(- 2 * mi))
    return ic


# TODO: refactor
def cmi_diff(x, y):
    """
    Compute the cumulative mutual information based on the difference between joint and marginals CEs.
    $$CMI(x, y) = (max(x) - \bar{x}) CE(y) + (max(y) - \bar{y}) CE(x) - CE(x, y)$$
    :param x: array-like,
    :param y: array-like,
    :return: float, cumulative mutual information
    """
    n = len(x)

    x = x - np.mean(x)
    y = y - np.mean(y)

    lattice_x = np.zeros((n, n), dtype="float_")  # Array of lower x tile coordinates
    lattice_y = np.zeros((n, n), dtype="float_")  # Array of lower y tile coordinates

    lattice_area = np.zeros((n, n), dtype="float_")  # Array of tile areas
    lattice_count = np.zeros((n, n), dtype="uint8")  # Array of tile data counts

    x_order = np.argsort(x, kind='quicksort')
    x_sorted = x[x_order]
    y_order = np.argsort(y, kind='quicksort')
    y_sorted = y[y_order]

    ind = np.arange(0, n, 1)
    ind_x_order = np.argsort(x_order, kind='quicksort')
    ind_x = ind[ind_x_order]
    ind_y_order = np.argsort(y_order, kind='quicksort')
    ind_y = ind[ind_y_order]

    for i in range(n):
        for j in range(n):
            lattice_x[i, j] = x_sorted[i]
            lattice_y[i, j] = y_sorted[j]
            if i < (n - 1) and j < (n - 1):
                lattice_area[i, j] = (x_sorted[i + 1] - x_sorted[i]) * (y_sorted[j + 1] - y_sorted[j])

    for i in range(n):
        lattice_count[ind_x[i], ind_y[i]] = 1

    np.cumsum(lattice_count, axis=1, out=lattice_count)
    np.cumsum(lattice_count, axis=0, out=lattice_count)

    cdf_xy = lattice_count / float(n)
    cdf_x = np.arange(0, n, 1) / float(n)
    cdf_y = cdf_x
    ce_x = 0
    ce_y = 0

    for i in range(n - 1):
        if cdf_x[i] != 0:
            ce_x -= (x_sorted[i + 1] - x_sorted[i]) * cdf_x[i] * math.log(cdf_x[i])
        if cdf_y[i] != 0:
            ce_y -= (y_sorted[i + 1] - y_sorted[i]) * cdf_y[i] * math.log(cdf_y[i])

    ce_xy = 0
    for i in range(n):
        for j in range(n):
            if cdf_xy[i, j] != 0:
                ce_xy = ce_xy - lattice_area[i, j] * cdf_xy[i, j] * math.log(cdf_xy[i, j])

    # Compute the cmi based on the difference between joint and marginals ce's
    cmi = ce_xy - (max(y) - np.mean(y)) * ce_x - (max(x) - np.mean(x)) * ce_y
    if cmi < 0:
        cmi = 0

    return cmi / (np.max(x) * np.max(y))


def cmi_ratio(x, y):
    """
    Compute the cumulative mutual information based on the ratio between joint and marginals CEs.
    $$CMI(x, y) = \int \int P(x,y) \log \left ( \frac{P(x,y)}{P(x)P(y)} \right ) dx dy$$
    :param x: array-like,
    :param y: array-like,
    :return: float, cumulative mutual information
    """
    n = len(x)

    x = x - np.mean(x)
    y = y - np.mean(y)

    lattice_x = np.zeros((n, n), dtype="float_")  # Array of lower x tile coordinates
    lattice_y = np.zeros((n, n), dtype="float_")  # Array of lower y tile coordinates
    lattice_area = np.zeros((n, n), dtype="float_")  # Array of tile areas
    lattice_count = np.zeros((n, n), dtype="uint8")  # Array of tile data counts

    x_order = np.argsort(x)
    x_sorted = x[x_order]
    y_order = np.argsort(y)
    y_sorted = y[y_order]

    ind = np.arange(0, n, 1)
    ind_x_order = np.argsort(x_order)
    ind_x = ind[ind_x_order]
    ind_y_order = np.argsort(y_order)
    ind_y = ind[ind_y_order]

    for i in range(n):
        for j in range(n):
            lattice_x[i, j] = x_sorted[i]
            lattice_y[i, j] = y_sorted[j]
            if i < (n - 1) and j < (n - 1):
                lattice_area[i, j] = (x_sorted[i + 1] - x_sorted[i]) * (y_sorted[j + 1] - y_sorted[j])

    for i in range(n):
        lattice_count[ind_x[i], ind_y[i]] = 1

    np.cumsum(lattice_count, axis=1, out=lattice_count)
    np.cumsum(lattice_count, axis=0, out=lattice_count)
    cdf_xy = lattice_count / float(n)
    cdf_x = np.arange(0, n, 1) / float(n)
    cdf_y = cdf_x

    # Compute the cmi based on the ratio of joint and marginals CE's
    cmi = 0
    for i in range(n):
        for j in range(n):
            if cdf_x[i] != 0 and cdf_y[j] != 0 and cdf_xy[i, j] != 0:
                cmi -= lattice_area[i, j] * cdf_xy[i, j] * math.log(cdf_xy[i, j]) / (cdf_x[i] * cdf_y[j])

    return cmi / (np.max(x) * np.max(y))
