from math import ceil


def get_extreme_series_indices(series, threshold, ascending=True):

    if threshold is None:

        return series.sort_values(ascending=ascending).index.tolist()

    elif 0.5 <= threshold < 1:

        top_and_bottom = (series <= series.quantile(1 - threshold)) | (
            series.quantile(threshold) <= series
        )

    elif 1 <= threshold:

        rank = series.rank(method="dense")

        threshold = min(threshold, ceil(series.size / 2))

        top_and_bottom = (rank <= threshold) | ((rank.max() - threshold) < rank)

    return sorted(series.index[top_and_bottom], key=series.get, reverse=not ascending)
