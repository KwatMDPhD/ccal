def merge_dicts_with_function(dict_0, dict_1, function):

    dict = {}

    for key in dict_0.keys() | dict_1.keys():

        if key in dict_0 and key in dict_1:

            dict[key] = function(dict_0[key], dict_1[key])

        elif key in dict_0:

            dict[key] = dict_0[key]

        elif key in dict_1:

            dict[key] = dict_1[key]

        else:

            raise ValueError("dict_0 or dict_1 changed during iteration.")

    return dict
