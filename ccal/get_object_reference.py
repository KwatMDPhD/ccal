from .cast_str_to_builtins import cast_str_to_builtins


def get_objectreference(object, namespace):

    for reference, object_ in namespace.items():

        if object is object_:

            return reference

    return cast_str_to_builtins(object)
