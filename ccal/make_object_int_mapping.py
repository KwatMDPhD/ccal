from .get_unique_iterable_objects_in_order import get_unique_iterable_objects_in_order


def make_object_int_mapping(iterable):

    object_int = {}

    int_object = {}

    for int, object in enumerate(get_unique_iterable_objects_in_order(iterable)):

        object_int[object] = int

        int_object[int] = object

    return object_int, int_object
