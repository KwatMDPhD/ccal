from multiprocessing.pool import Pool

from numpy.random import seed


def multiprocess(callable, arguments, n_job, random_seed=20121020):

    seed(random_seed)

    with Pool(n_job) as process:

        return process.starmap(callable, arguments)
