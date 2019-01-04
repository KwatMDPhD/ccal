from warnings import warn

from pandas import Series

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
