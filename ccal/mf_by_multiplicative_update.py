from numpy import full, nan
from numpy.random import random_sample, seed

from .compute_matrix_norm import compute_matrix_norm
from ._update_H_by_multiplicative_update import _update_H_by_multiplicative_update
from ._update_W_by_multiplicative_update import _update_W_by_multiplicative_update
from .RANDOM_SEED import RANDOM_SEED


def mf_by_multiplicative_update(V, k, n_iteration=int(1e3), random_seed=RANDOM_SEED):

    R_norms = full(n_iteration + 1, nan)

    seed(random_seed)

    W = random_sample(size=(V.shape[0], k))

    H = random_sample(size=(k, V.shape[1]))

    R_norms[0] = compute_matrix_norm(V - W @ H)

    for i in range(n_iteration):

        W = _update_W_by_multiplicative_update(V, W, H)

        H = _update_H_by_multiplicative_update(V, W, H)

        R_norms[i + 1] = compute_matrix_norm(V - W @ H)

    return W, H, R_norms
