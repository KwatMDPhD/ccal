from pandas import Series


def make_object_int_and_int_object_for_iterable(iterable):

    object_int = {}

    int_object = {}

    for int, object in enumerate(Series(iterable).sort_values().unique()):

        object_int[object] = int

        int_object[int] = object

    return object_int, int_object
