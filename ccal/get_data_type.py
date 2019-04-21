from numpy import asarray, isnan, unique


def get_data_type(data):

    data = asarray(data)

    data_good = data[~isnan(data)]

    if all(float(n).is_integer() for n in data_good):

        n_good_unique = unique(data_good).size

        if n_good_unique == 2:

            return "binary"

        elif n_good_unique <= 32:

            return "categorical"

    return "continuous"
