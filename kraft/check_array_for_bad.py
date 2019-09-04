from numpy import isinf, isnan


def check_array_for_bad(array, raise_for_bad=True):

    bad_kinds = []

    is_nan = isnan(array)

    if is_nan.any():

        bad_kinds.append("nan")

    is_inf = isinf(array)

    if is_inf.any():

        bad_kinds.append("inf")

    is_bad = is_nan | is_inf

    n_bad = is_bad.sum()

    if 0 < n_bad and raise_for_bad:

        n_good = array.size - n_bad

        bad_kinds = "|".join(bad_kinds)

        raise ValueError("{} good & {} bad ({}).".format(n_good, n_bad, bad_kinds))

    return is_bad
