import numpy as np
from scipy.stats import pearsonr
from statsmodels.nonparametric.kernel_density import KDEMultivariate

# For bcv():
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri
from rpy2.robjects.packages import importr
mass = importr("MASS")


def rbcv(x):
    """
    :param x: array-like, (n_samples,)
    :return: float, bandwidth
    """
    bandwidth = np.array(mass.bcv(x))[0]
    return bandwidth


def compute_mutual_information(x, y, n_grid=25, var_types='cc', bandwidth_scaling=None):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_types: two-character string of 'c' (continuous) or 'd' (discrete)
    :param bandwidth_scaling: float
    :return: float, mutual information

    Notes
    -----
    Difference in bandwidth convention means bcv() delta must be divided by 4
    Todo: Bandwidth selection method as robust and fast as R bcv() without invoking R. Maybe in Cython or C.
    """
    x_set = np.logical_not(np.isnan(x))
    y_set = np.logical_not(np.isnan(y))
    overlap = x_set & y_set
    n_overlap = overlap.sum()
    jitter_scale = 1E-10
    x_jitter, y_jitter = [jitter_scale * np.random.uniform(size=n_overlap) for _ in [x, y]]
    x = x[overlap] + x_jitter
    y = y[overlap] + y_jitter
    xy = [x, y]
    x_grid = np.linspace(x.min(), x.max(), n_grid)
    y_grid = np.linspace(y.min(), y.max(), n_grid)
    xg, yg = np.meshgrid(x_grid, y_grid)
    grid = np.vstack([xg.flatten(), yg.flatten()])
    delta = np.array([rbcv(z) for z in [x, y]]).reshape((2,)) / 4
    if bandwidth_scaling:
        delta *= bandwidth_scaling
    kde = KDEMultivariate(xy, bw=delta, var_type=var_types)
    fxy = kde.pdf(grid).reshape((n_grid, n_grid)).T + np.finfo(float).eps
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=0) * dy
    py = pxy.sum(axis=1) * dx
    pxg, pyg = np.meshgrid(px, py)
    mi = np.sum(pxy * np.log(pxy / (pxg * pyg))) * dx * dy
    return mi


def compute_information_coefficient(x, y, n_grid=25, var_types='cc'):
    """
    :param x: array-like, (n_samples,)
    :param y: array-like, (n_samples,)
    :param n_grid: int, number of grid points at which to evaluate kernel density
    :param var_types: two-character string of 'c' (continuous) or 'd' (discrete)
    :return: float, information coefficient
    """
    rho, p = pearsonr(x, y)
    rho2 = abs(rho)
    bandwidth_scaling = (1 + (-0.75)*rho2)
    mi = compute_mutual_information(x, y, n_grid, var_types, bandwidth_scaling)
    ic = np.sign(rho) * np.sqrt(1 - np.exp(- 2 * mi))
    return ic
