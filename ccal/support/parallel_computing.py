from multiprocessing.pool import Pool

from .system import get_random_state


def parallelize(function, list_of_args, n_jobs):
    """
    Apply function on list_of_args using parallel computing across n_jobs jobs; n_jobs doesn't have to be the length of
    list_of_args.
    :param function: function;
    :param list_of_args: iterable;
    :param n_jobs: int; 0 <
    :return: list;
    """

    get_random_state('parallelize (before)')

    # # TODO: Consider removing since multiprocess also performs this logic
    # n_args = len(list_of_args)
    # if n_args < n_jobs:
    #     n_jobs = n_args
    #     print_log('Changed n_jobs to {} (n_args < n_jobs).'.format(n_jobs))

    if 1 < n_jobs:  # Work in new processes
        with Pool(n_jobs) as p:
            # Each process initializes with the current jobs' randomness (seed & seed index)
            # Any changes to these jobs' randomnesses won't update the current process' randomness (seed & seed index)
            return_ = p.map(function, list_of_args)

    elif n_jobs == 1:  # Work in the current thread
        # The 1st function initializes with the current jobs' randomness (seed & seed index)
        # Each function's randomnesses updates the current process' randomness (seed & seed index)
        return_ = list(map(function, list_of_args))

    else:
        raise ValueError('n_jobs has to be greater than 0.')

    get_random_state('parallelize (after)')
    return return_
