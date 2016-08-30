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
from numpy import asarray, sum, prod, array, sign, sqrt, exp,log, linspace, finfo, meshgrid, vstack
from numpy.random import random_sample, permutation
from scipy.stats import pearsonr
from statsmodels.norametric.kernel_density import KDEMultivariate
from scipy.stats import binom_test
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr

from .support import drop_nan_columns

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')


def information_coefficient(x, y, n_grids=25, jitter=1E-10):
    """
    Compute the information correlation between `x` and `y`, which can be either continuous, categorical, or binary
    :param x: vector;
    :param y: vector;
    :param n_grids:
    :param jitter:
    :return:
    """
    x, y = drop_nan_columns([x, y])
    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)
    # TODO: should I check the length of y too?
    if len(x) <= 2:
        return 0
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    cor, p = pearsonr(x, y)
    bandwidth_x = asarray(mass.bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    bandwidth_y = asarray(mass.bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    fxy = asarray(mass.kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[2]) + finfo(
        float).eps
    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    mi = sum(pxy * log(pxy / (array([px] * n_grids).T * array([py] * n_grids)))) * dx * dy

    # hxy = - sum(pxy * log(pxy)) * dx * dy
    # hx = -sum(px * log(px)) * dx
    # hy = -sum(py * log(py)) * dy
    # mi = hx + hy - hxy

    ic = sign(cor) * sqrt(1 - exp(- 2 * mi))

    return ic


def information_coefficient_james(x, y, z=None, n_grid=25, vector_data_types=None, n_perm=0, adaptive=True, alpha=0.05,
                                  perm_alpha=0.05):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param z: array-like, (n_samples,), optional, variable on which to condition
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param vector_data_types: str, 3 chars of 'c' (continuous), 'u' (unordered discrete), or 'o' (ordered discrete)
    :param n_perm: int, >0 will return a p-value in addition to the information coefficient
    :param adaptive: bool, quit permutations after achieving a confidence that the p-value is above (or below) alpha
    :param alpha: float, threshold empirical p-value for significance of IC
    :param perm_alpha: float, threshold probability for terminating adaptive permutation
    :return: float (and float), information coefficient, and the empirical p-value if n_perm > 0
                If adaptive, the accuracy of the empirical p-value will vary: values closer to alpha will be estimated
                more precisely, while values obviously greater or less than alpha will be estimated less precisely.
    """
    vectors = [x, y]
    if z:
        vectors.append(z)
        x, y, z = add_jitter(drop_nan_columns(vectors))
    else:
        x, y = add_jitter(drop_nan_columns(vectors))

    if not vector_data_types:
        # TODO: guess variable types
        vector_data_types = 'c' * len(vectors)
    elif len(vector_data_types) is not len(vectors):
        raise ValueError('Number of specified variable types does not match number of vectors.')

    if len(x) <= len(vector_data_types):
        return 0

    rho, p = pearsonr(x, y)
    rho2 = abs(rho)
    bandwidth_scaling = 1 + (-0.75) * rho2

    mi = mutual_information(x, y, z=z,
                            n_grid=n_grid, vector_data_types=vector_data_types, bandwidth_scaling=bandwidth_scaling)

    ic_sign = sign(rho)
    ic = ic_sign * sqrt(1 - exp(- 2 * mi))

    if n_perm:
        n_more_extreme = 0
        trials = 0
        for i in range(n_perm):
            trials += 1
            # The question is whether I want to have
            # a certain width of confidence interval around my estimate of the pval
            # or just a certain confidence that the pval is greater than 0.05 (current solution)
            pm_x = permutation(x)
            pm_rho, p = pearsonr(pm_x, y)
            pm_rho2 = abs(pm_rho)
            pm_bandwidth_scaling = (1 + (-0.75) * pm_rho2)
            pm_mi = mutual_information(pm_x, y, z, n_grid=n_grid,
                                       vector_data_types=vector_data_types, bandwidth_scaling=pm_bandwidth_scaling)
            pm_ic_sign = sign(pm_rho)
            pm_ic = pm_ic_sign * sqrt(1 - exp(- 2 * pm_mi))
            if (pm_ic <= ic and ic < 0) or (0 < ic and ic <= pm_ic):
                n_more_extreme += 1
            if adaptive:
                ge_binom_p = binom_test(n_more_extreme, i + 1, alpha, alternative='greater')
                # * 2 because what I'm doing is two-sided testing in both directions
                if ge_binom_p * 2 < perm_alpha:
                    break
                le_binom_p = binom_test(n_more_extreme, i + 1, alpha, alternative='less')
                if le_binom_p * 2 < perm_alpha:
                    break
        p_value = n_more_extreme / float(trials)
        return ic, p_value
    else:
        return ic


def mutual_information(x, y, z=None, n_grid=25, vector_data_types=None, bandwidth_scaling=None):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param z: array-like, (n_samples,), optional, variable on which to condition
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param vector_data_types: str, 3 chars of 'c' (continuous), 'u' (unordered discrete), or 'o' (ordered discrete)
    :param bandwidth_scaling: float,
    :return: float, information coefficient
    """
    vectors = [x, y]
    if z:
        vectors.append(z)

    grids = [linspace(v.min(), v.max(), n_grid) for v in vectors]
    mesh_grids = meshgrid(*grids)
    grid_shape = tuple([n_grid] * len(vectors))
    grid = vstack([mesh_grid.flatten() for mesh_grid in mesh_grids])
    delta = array([rbcv(q) for q in vectors]).reshape((len(vectors),)) / 4
    if bandwidth_scaling:
        delta *= bandwidth_scaling
    kde = KDEMultivariate(vectors, bw=delta, var_type=vector_data_types)
    p_joint = kde.pdf(grid).reshape(grid_shape) + finfo(float).eps
    ds = [grid[1] - grid[0] for grid in grids]
    ds_prod = prod(ds)
    p_joint /= (p_joint.sum() * ds_prod)
    h_joint = - sum(p_joint * log(p_joint)) * ds_prod
    dx = ds[0]
    dy = ds[1]
    if z:
        dz = ds[2]
        pxz = p_joint.sum(axis=1) * dy
        pyz = p_joint.sum(axis=0) * dx
        pz = p_joint.sum(axis=(0, 1)) * dx * dy
        hxz = -sum(pxz * log(pxz)) * dx * dz
        hyz = -sum(pyz * log(pyz)) * dy * dz
        hz = -sum(pz * log(pz)) * dz
        cmi = hxz + hyz - h_joint - hz
        return cmi
    else:
        dx = ds[0]
        dy = ds[1]
        px = p_joint.sum(axis=1) * dy
        py = p_joint.sum(axis=0) * dx
        hx = -sum(px * log(px)) * dx
        hy = -sum(py * log(py)) * dy
        mi = hx + hy - h_joint
        return mi


def rbcv(x):
    """
    :param x: array-like, (n_samples,)
    :return: float, bandwidth
    """
    bandwidth = array(mass.bcv(x))[0]
    return bandwidth
