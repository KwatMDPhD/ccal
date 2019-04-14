from warnings import warn

from pandas import Series

from .cast_str_to_builtin import cast_str_to_builtin
from .replace_bad_objects_in_iterable import replace_bad_objects_in_iterable


def cast_series_to_builtin(series):

    list = replace_bad_objects_in_iterable(
        tuple(cast_str_to_builtin(object) for object in series)
    )

    try:

        return Series(list, index=series.index, dtype=float)

    except (TypeError, ValueError) as exception:

        warn(exception)

        return Series(list, index=series.index)
