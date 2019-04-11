from numpy import where


def replace_bad_objects_in_iterable(
    iterable,
    bad_objects=("--", "unknown", "n/a", "N/A", "na", "NA", "nan", "NaN", "NAN"),
    replacement=None,
):

    return tuple(
        where(object in bad_objects, replacement, object) for object in iterable
    )
