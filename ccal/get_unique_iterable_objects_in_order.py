def get_unique_iterable_objects_in_order(iterable):

    unique_objects_in_order = []

    for object in iterable:

        if object not in unique_objects_in_order:

            unique_objects_in_order.append(object)

    return unique_objects_in_order
