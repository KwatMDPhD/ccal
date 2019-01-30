from numpy.random import seed


def sample_series_randomly_per_value(series, n_per_value=None, random_seed=20121020):

    if n_per_value is None:

        n_per_value = series.value_counts().min()

        print("n_per_value = {}".format(n_per_value))

    indices_selected = []

    seed(random_seed)

    for group_name, group_series in series.groupby(series):

        indices_selected.extend(group_series.sample(n=n_per_value).index.sort_values())

    return series[indices_selected]
