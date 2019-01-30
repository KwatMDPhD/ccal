from numpy.random import seed


def sample_series_randomly_per_value(series, n_per_value, random_seed=20121020):

    indices_selected = []

    seed(random_seed)

    for group_name, group_series in series.groupby(series):

        indices_selected.extend(group_series.sample(n=n_per_value).index.sort_values())

    return series[indices_selected]
