from math import ceil
from warnings import warn

from pandas import DataFrame, Series

from .iterable import replace_bad_objects_in_iterable
from .str_ import cast_str_to_builtins


def cast_series_to_builtins(series):

    list_ = replace_bad_objects_in_iterable(
        tuple(cast_str_to_builtins(object_) for object_ in series)
    )

    try:

        return Series(list_, index=series.index, dtype=float)

    except (TypeError, ValueError) as exception:

        warn(exception)

        return Series(list_, index=series.index)


def make_membership_df_from_categorical_series(series):

    object_x_index = DataFrame(index=sorted(set(series.dropna())), columns=series.index)

    object_x_index.index.name = series.name

    for object_ in object_x_index.index:

        object_x_index.loc[object_] = (series == object_).astype(int)

    return object_x_index


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
