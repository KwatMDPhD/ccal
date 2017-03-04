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

from pandas import DataFrame
from sklearn.decomposition import NMF

from .. import RANDOM_SEED
from ..support.file import write_gct


def nmf(matrix, ks, init='random', solver='cd', tol=1e-6, max_iter=1000, random_seed=RANDOM_SEED,
        alpha=0, l1_ratio=0, shuffle_=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1):
    """
    Non-negative matrix factorize matrix with k from ks.
    :param matrix: numpy array or pandas DataFrame; (n_samples, n_features); the matrix to be factorized by NMF
    :param ks: iterable; list of ks to be used in the NMF
    :param init:
    :param solver:
    :param tol:
    :param max_iter:
    :param random_seed:
    :param alpha:
    :param l1_ratio:
    :param shuffle_:
    :param nls_max_iter:
    :param sparseness:
    :param beta:
    :param eta:
    :return: dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}}
    """

    if isinstance(ks, int):
        ks = [ks]
    else:
        ks = list(set(ks))

    nmf_results = {}
    for k in ks:

        # Compute W, H, and reconstruction error
        model = NMF(n_components=k, init=init, solver=solver, tol=tol, max_iter=max_iter, random_state=random_seed,
                    alpha=alpha, l1_ratio=l1_ratio, shuffle=shuffle_, nls_max_iter=nls_max_iter, sparseness=sparseness,
                    beta=beta, eta=eta)
        w, h, err = model.fit_transform(matrix), model.components_, model.reconstruction_err_

        # Return pandas DataFrame if the input matrix is also a DataFrame
        if isinstance(matrix, DataFrame):
            w = DataFrame(w, index=matrix.index)
            h = DataFrame(h, columns=matrix.columns)

        # Save NMF results
        nmf_results[k] = {'w': w, 'h': h, 'e': err}

    return nmf_results


def save_nmf_w_h(nmf_results, filepath_prefix):
    """
    Save NMF decompositions.
    :param nmf_results: dict; {k: {w: W matrix, h: H matrix, e: Reconstruction Error}} and
                              {k: Cophenetic Correlation Coefficient}
    :param filepath_prefix: str; filepath_prefix_nmf_k{k}_{w, h}.gct and will be saved
    :return: None
    """

    for k, v in nmf_results.items():
        write_gct(v['w'], filepath_prefix + 'nmf_k{}_w.gct'.format(k))
        write_gct(v['h'], filepath_prefix + 'nmf_k{}_h.gct'.format(k))
