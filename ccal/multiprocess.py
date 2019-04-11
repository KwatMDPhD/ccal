from multiprocessing.pool import Pool

from numpy.random import seed

from .RANDOM_SEED import RANDOM_SEED


def multiprocess(callable, arguments, n_job, random_seed=RANDOM_SEED):

    seed(random_seed)

    with Pool(n_job) as process:

        return process.starmap(callable, arguments)
