def flatten_nested_iterable(iterable, iterable_types=(list, tuple)):

    list = list(iterable)

    i = 0

    while i < len(list):

        while isinstance(list[i], iterable_types):

            if not len(list[i]):

                list.pop(i)

                i -= 1

                break

            else:

                list[i : i + 1] = list[i]

        i += 1

    return list
