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

from numpy import asarray, cumsum, empty, in1d, max, mean, min, where
from numpy.random import shuffle
from pandas import DataFrame

from ..support.d2 import normalize_2d_or_1d


def convert_genes_to_gene_sets(g_x_s, gss, power=1,
                               statistic='Kolmogorov-Smirnov',
                               n_permutations=0):
    """
    Convert Gene-x-Sample ==> Gene-Set-x-Sample.
    :param g_x_s: DataFrame;
    :param gss: DataFrame;
    :param power: number;
    :param statistic: str;
    :param n_permutations: int;
    :return: DataFrame;
    """

    # Rank normalize columns
    g_x_s = normalize_2d_or_1d(g_x_s, 'rank', axis=0) / \
            g_x_s.shape[0]

    # Make Gene-Set-x-Sample place holder
    gs_x_s = DataFrame(index=gss.index, columns=g_x_s.columns)

    # Loop over gene sets
    for gs_n, gs in gss.iterrows():
        print('Computing {} enrichment ...'.format(gs_n))

        gs = asarray(gs.dropna())

        # Loop over samples
        for s_n, s_v in g_x_s.items():

            # Sort sample values from high to low and compute enrichment score
            s_s_v = s_v.sort_values(ascending=False) ** power
            es = _get_es(s_s_v, gs, statistic=statistic)

            if 0 < n_permutations:  # Compute permutation-normalized
                # enrichment score
                p_ess = empty(n_permutations)
                p_s_v = s_s_v.copy()
                for i in range(n_permutations):
                    # Permute sample values and compute enrichment score
                    shuffle(p_s_v)
                    p_ess[i] = _get_es(p_s_v, gs, statistic=statistic)

                # Compute permutation-normalized enrichment score
                gs_x_s.ix[gs_n, s_n] = es / mean(p_ess)

            else:  # Use enrichment score instead of permutation-normalized
                # enrichment score
                gs_x_s.ix[gs_n, s_n] = es

    return gs_x_s


def _get_es(sv, gs, statistic='Kolmogorov-Smirnov'):
    """
    Compute enrichment score: "Is sorted values enriched in gene set?".
    :param sv: Series; sorted values
    :param gs: array; gene set
    :param statistic: str;
    :return: float; enrichment score
    """

    # Check if each gene (in the sorted order) is in the gene set (hit) or
    # not (miss)
    in_ = in1d(asarray(sv.index), gs, assume_unique=True)

    # Score: values-at-hits / sum(values-at-hits) - is-miss's / number-of-misses
    s = in_.astype(int) * sv / sum(sv.ix[in_]) - (1 - in_.astype(int)) / (
        in_.size - sum(in_))

    # Sum over scores
    cs = cumsum(s)

    # Compute enrichment score
    max_es = max(cs)
    min_es = min(cs)
    if statistic == 'Kolmogorov-Smirnov':
        es = where(abs(min_es) < abs(max_es), max_es, min_es)
    else:
        raise ValueError('Not implemented!')

    # mpl.pyplot.figure(figsize=(8, 5))
    #     ax = mpl.pyplot.gca()
    #     ax.plot(range(in_.size), in_, color='black', alpha=0.16)
    #     ax.plot(range(in_.size), s)
    #     ax.plot(range(in_.size), cs)
    #     mpl.pyplot.show()

    return es
