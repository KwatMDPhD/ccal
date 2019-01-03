from numpy import asarray, mean, where


def group_iterable(iterable, n, keep_leftover_group=False):

    groups = []

    group = []

    for object_ in iterable:

        group.append(object_)

        if len(group) == n:

            groups.append(group)

            group = []

    if len(group) != 0 and (len(group) == n or keep_leftover_group):

        groups.append(group)

    return groups


def flatten_nested_iterable(iterable, iterable_types=(list, tuple)):

    list_ = list(iterable)

    i = 0

    while i < len(list_):

        while isinstance(list_[i], iterable_types):

            if not len(list_[i]):

                list_.pop(i)

                i -= 1

                break

            else:

                list_[i : i + 1] = list_[i]

        i += 1

    return list_


def replace_bad_objects_in_iterable(
    iterable,
    bad_objects=("--", "unknown", "n/a", "N/A", "na", "NA", "nan", "NaN", "NAN"),
    replacement=None,
):

    return tuple(
        where(object_ in bad_objects, replacement, object_) for object_ in iterable
    )


def group_and_apply_function_on_each_group_in_iterable(
    iterable, groups, callable_=mean
):

    unique_groups_in_order = get_unique_iterable_objects_in_order(groups)

    applied_by_group = []

    for group in unique_groups_in_order:

        applied_by_group.append(callable_(asarray(iterable)[groups == group]))

    return unique_groups_in_order, applied_by_group


def get_unique_iterable_objects_in_order(iterable):

    unique_objects_in_order = []

    for object_ in iterable:

        if object_ not in unique_objects_in_order:

            unique_objects_in_order.append(object_)

    return unique_objects_in_order


def make_object_int_mapping(iterable):

    object_int = {}

    int_object = {}

    for int_, object in enumerate(get_unique_iterable_objects_in_order(iterable)):

        object_int[object] = int_

        int_object[int_] = object

    return object_int, int_object
