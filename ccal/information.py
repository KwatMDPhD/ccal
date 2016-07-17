"""
Computational Cancer Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov


Description:
Information computing module for CCAL.
"""
import math

import numpy as np
from scipy.stats import pearsonr
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.stats import binom_test

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
def mutual_information(x, y, z=None, n_grid=25, var_types=None, bandwidth_scaling=None):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param z: array-like, (n_samples,), optional, variable on which to condition
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_types: three-character string of 'c' (continuous), 'u' (unordered discrete) or 'o' (ordered discrete)
    :param bandwidth_scaling: float
    :return: float, information coefficient
    """
    n = len(x)
    variables = [x, y]
    if z is not None:
        variables.append(z)
    for v in variables[1:]:
        if len(v) != n:
            raise ValueError("Input arrays have different lengths")
    n_vars = len(variables)
    if var_types is None:
        var_types = ''.join(['c' for _ in range(n_vars)])
        # Todo: guess variable types
    if len(var_types) != n_vars:
        raise ValueError("Number of specified variable types does not match number of variables")
    non_nans = [np.logical_not(np.isnan(v)) for v in variables]
    overlap = [True] * n
    for non_nan in non_nans:
        overlap &= non_nan
    n_overlap = overlap.sum()
    if n_overlap < 3:
        return 0
    jitter_scale = 1E-10
    jitters = [jitter_scale * np.random.uniform(size=n_overlap) for v in variables]
    for i, v in enumerate(variables):
        v = v[overlap] + jitters[i]
    grids = [np.linspace(v.min(), v.max(), n_grid) for v in variables]
    mesh_grids = np.meshgrid(*grids)
    grid_shape = tuple([n_grid] * n_vars)
    grid = np.vstack([mesh_grid.flatten() for mesh_grid in mesh_grids])
    delta = np.array([rbcv(q) for q in variables]).reshape((n_vars,)) / 4
    if bandwidth_scaling is not None:
        delta *= bandwidth_scaling
    kde = KDEMultivariate(variables, bw=delta, var_type=var_types)
    p_joint = kde.pdf(grid).reshape(grid_shape) + np.finfo(float).eps
    ds = [grid[1] - grid[0] for grid in grids]
    ds_prod = np.prod(ds)
    p_joint /= (p_joint.sum() * ds_prod)
    h_joint = - np.sum(p_joint * np.log(p_joint)) * ds_prod
    dx = ds[0]
    dy = ds[1]
    if z is None:
        dx = ds[0]
        dy = ds[1]
        px = p_joint.sum(axis=1) * dy
        py = p_joint.sum(axis=0) * dx
        hx = -np.sum(px * np.log(px)) * dx
        hy = -np.sum(py * np.log(py)) * dy
        mi = hx + hy - h_joint
        return mi
    else:
        dz = ds[2]
        pxz = p_joint.sum(axis=1) * dy
        pyz = p_joint.sum(axis=0) * dx
        pz = p_joint.sum(axis=(0, 1)) * dx * dy
        hxz = -np.sum(pxz * np.log(pxz)) * dx * dz
        hyz = -np.sum(pyz * np.log(pyz)) * dy * dz
        hz = -np.sum(pz * np.log(pz)) * dz
        cmi = hxz + hyz - h_joint - hz
        return cmi


def information_coefficient(x, y, z=None, n_grid=25, var_types=None, n_permutations=0, adaptive=True, alpha=0.05,
                            perm_alpha=0.05):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param z: array-like, (n_samples,), optional, variable on which to condition
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_types: three-character string of 'c' (continuous), 'u' (unordered discrete) or 'o' (ordered discrete)
    :param n_permutations: int, >0 will return a p-value in addition to the information coefficient
    :param adaptive: bool, quit permutations after achieving a specified confidence that the p-value is above (or below)
            alpha
    :param alpha: float, threshold empirical p-value for significance of IC
    :param perm_alpha: float, threshold probability for terminating adaptive permutation
    :return: float, information coefficient; if n_permutations > 0, also the empirical p-value
    Note that if adaptive, the accuracy of the empirical p-value will vary: values closer to alpha will be estimated
    more precisely, while values obviously greater or less than alpha will be estimated less precisely.
    """

    rho, p = pearsonr(x, y)
    rho2 = abs(rho)
    bandwidth_scaling = (1 + (-0.75) * rho2)
    ic_sign = np.sign(rho)
    mi = mutual_information(x, y, z=z, n_grid=n_grid,
                            var_types=var_types, bandwidth_scaling=bandwidth_scaling)
    ic = ic_sign * np.sqrt(1 - np.exp(- 2 * mi))

    if n_permutations > 0:
        n_more_extreme = 0
        trials = 0
        for i in range(n_permutations):
            trials += 1
            # the question is whether I want to have
            # a certain width of confidence interval around my estimate of the pval
            # or just a certain confidence that the pval is greater than 0.05 (current solution)
            pm_x = np.random.permutation(x)
            pm_rho, p = pearsonr(pm_x, y)
            pm_rho2 = abs(pm_rho)
            pm_bandwidth_scaling = (1 + (-0.75) * pm_rho2)
            pm_mi = mutual_information(pm_x, y, z, n_grid=n_grid,
                                       var_types=var_types, bandwidth_scaling=pm_bandwidth_scaling)
            pm_ic_sign = np.sign(pm_rho)
            pm_ic = pm_ic_sign * np.sqrt(1 - np.exp(- 2 * pm_mi))
            # print pm_ic
            if ic > 0:
                if pm_ic >= ic:
                    n_more_extreme += 1
            elif ic < 0:
                if pm_ic <= ic:
                    n_more_extreme += 1
            else:
                n_more_extreme += 1
            if adaptive:
                ge_binom_p = binom_test(n_more_extreme, i + 1, alpha, alternative='greater')
                if ge_binom_p * 2 < perm_alpha:  # * 2 because what I'm doing is two-sided, testing in both directions
                    break
                le_binom_p = binom_test(n_more_extreme, i + 1, alpha, alternative='less')
                if le_binom_p * 2 < perm_alpha:
                    break
        # print i + 1, 'trials'
        p_value = n_more_extreme / float(trials)
        return ic, p_value
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
