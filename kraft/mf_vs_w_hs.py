from numpy import full, nan, sum
from numpy.random import random_sample, seed

from .compute_matrix_norm import compute_matrix_norm
from .RANDOM_SEED import RANDOM_SEED
from .update_h_by_multiplicative_update import update_h_by_multiplicative_update


def mf_vs_w_hs(vs, r, weights=None, n_iteration=int(1e2), random_seed=RANDOM_SEED):

    assert len(set(v.shape[0] for v in vs)) == 1

    n = len(vs)

    errors = full((n, n_iteration), nan)

    seed(seed=random_seed)

    w = random_sample(size=(vs[0].shape[0], r))

    hs = [random_sample(size=(r, v.shape[1])) for v in vs]

    v_0_norm = compute_matrix_norm(vs[0])

    if weights is None:

        weights = [v_0_norm / compute_matrix_norm(v) for v in vs]

    for j in range(n_iteration):

        top = sum([weights[i] * vs[i] @ hs[i].T for i in range(n)], axis=0)

        bottom = sum([weights[i] * w @ hs[i] @ hs[i].T for i in range(n)], axis=0)

        w *= top / bottom

        hs = [update_h_by_multiplicative_update(vs[i], w, hs[i]) for i in range(n)]

        errors[:, j] = [compute_matrix_norm(vs[i] - w @ hs[i]) for i in range(n)]

    return w, hs, errors