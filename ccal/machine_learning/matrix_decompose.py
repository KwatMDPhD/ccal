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


import numpy as np
from numpy import divide, dot, finfo, log, matrix, multiply, ndarray, sum
from numpy.random import rand, seed
from pandas import DataFrame
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from .. import RANDOM_SEED


def nmf(matrix_,
        ks,
        algorithm='Lee & Seung',
        init=None,
        solver='cd',
        tol=1e-7,
        max_iter=1000,
        random_seed=RANDOM_SEED,
        alpha=0.0,
        l1_ratio=0.0,
        verbose=0,
        shuffle_=False,
        nls_max_iter=2000,
        sparseness=None,
        beta=1,
        eta=0.1):
    """
    Non-negative matrix factorize matrix with k from ks.

    :param matrix_: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF

    :param algorithm: str; 'Alternating Least Squares' or 'Lee & Seung'

    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_seed:
    :param alpha:
    :param l1_ratio:
    :param verbose:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:

    :return: dict and dict; {k: {w:w_matrix, h:h_matrix, e:reconstruction_error}} and
                            {k: cophenetic correlation coefficient}
    """

    if isinstance(ks, int):
        ks = [ks]
    else:
        ks = list(set(ks))

    nmf_results = {}
    for k in ks:

        # Compute W, H, and reconstruction error
        if algorithm == 'Alternating Least Squares':
            model = NMF(n_components=k,
                        init=init,
                        solver=solver,
                        tol=tol,
                        max_iter=max_iter,
                        random_state=random_seed,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        verbose=verbose,
                        shuffle=shuffle_,
                        nls_max_iter=nls_max_iter,
                        sparseness=sparseness,
                        beta=beta,
                        eta=eta)
            w, h, e = model.fit_transform(
                matrix_), model.components_, model.reconstruction_err_

        elif algorithm == 'Lee & Seung':
            w, h, e = nmf_div(
                matrix_, k, n_max_iterations=max_iter, random_seed=random_seed)

        else:
            raise ValueError(
                'NMF algorithm are: \'Alternating Least Squares\' or \'Lee & Seung\'.'
            )

        # Return pandas DataFrame if the input matrix is also a DataFrame
        if isinstance(matrix_, DataFrame):
            w = DataFrame(w, index=matrix_.index)
            h = DataFrame(h, columns=matrix_.columns)

        # Save NMF results
        nmf_results[k] = {'w': w, 'h': h, 'e': e}

    return nmf_results


# TODO: refactor
# TODO: optimize
def nmf_div(V, k, n_max_iterations=1000, random_seed=RANDOM_SEED):
    """
    Non-negative matrix factorize matrix with k from ks using divergence.
    :param V: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param k: int; number of components
    :param n_max_iterations: int;
    :param random_seed:
    :return:
    """

    eps = finfo(float).eps

    N = V.shape[0]
    M = V.shape[1]
    V = matrix(V)
    seed(random_seed)
    W = rand(N, k)
    H = rand(k, M)
    for t in range(n_max_iterations):
        VP = dot(W, H)
        W_t = matrix.transpose(W)
        H = multiply(H, dot(W_t, divide(V, VP))) + eps
        for i in range(k):
            W_sum = 0
            for j in range(N):
                W_sum += W[j, i]
            for j in range(M):
                H[i, j] = H[i, j] / W_sum
        VP = dot(W, H)
        H_t = matrix.transpose(H)
        W = multiply(W, dot(divide(V, VP + eps), H_t)) + eps
        W = divide(W, ndarray.sum(H, axis=1, keepdims=False))

    err = sum(multiply(V, log(divide(V + eps, VP + eps))) - V + VP) / (M * N)

    return W, H, err


def nmf_bcv(x, nmf, nfold=2, nrepeat=1):
    """
    Bi-crossvalidation of NMF as in Owen and Perry (2009).
    Note that this implementation does not require the intermediates to be non-negative. Details of how to add this
    constraint can be found on page 11 (beginning of section 5) of Owen and Perry (2009); the authors did not seem to
    consider it especially important for quality of model selection.
    :param x: data array to be decomposed, (nsamples, nfeatures)
    :param nmf: sklearn NMF object, already initialized
    :param nfold: number of folds for cross-validation (O&P suggest 2)
    :param nrepeat: how many times to repeat, to average out variation based on which rows and columns were held out
    :return: mean_error, mean mse across nrepeat
    """
    errors = []
    for rep in range(nrepeat):
        kf_rows = KFold(x.shape[0], nfold, shuffle=True)
        kf_cols = KFold(x.shape[1], nfold, shuffle=True)
        for row_train, row_test in kf_rows:
            for col_train, col_test in kf_cols:
                a = x[row_test][:, col_test]
                base_error = mean_squared_error(a, np.zeros(a.shape))
                b = x[row_test][:, col_train]
                c = x[row_train][:, col_test]
                d = x[row_train][:, col_train]
                nmf.fit(d)
                hd = nmf.components_
                wd = nmf.transform(d)
                wa = np.dot(b, hd.T)
                ha = np.dot(wd.T, c)
                a_prime = np.dot(wa, ha)
                a_notzero = a != 0
                scaling_factor = np.mean(np.divide(a_prime, a)[a_notzero])
                scaled_a_prime = a_prime / scaling_factor
                error = mean_squared_error(a, scaled_a_prime) / base_error
                errors.append(error)
    mean_error = np.mean(errors)
    return mean_error
