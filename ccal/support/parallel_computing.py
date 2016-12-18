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

from multiprocessing.pool import Pool

from numpy.random import seed


def parallelize(function, list_of_args, n_jobs, random_seed=None):
    """
    Apply function on list_of_args using parallel computing across n_jobs jobs; n_jobs doesn't have to be the length of
    list_of_args.
    :param function: function;
    :param list_of_args: iterable;
    :param n_jobs: int; 0 <
    :param random_seed: int;
    :return: list;
    """

    if random_seed:
        seed(random_seed)

    with Pool(n_jobs) as p:
        # Each process initializes with the current jobs' randomness (seed & seed index)
        # Any changes to these jobs' randomnesses won't update the current process' randomness (seed & seed index)
        return_ = p.map(function, list_of_args)

    return return_
