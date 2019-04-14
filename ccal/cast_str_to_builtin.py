def cast_str_to_builtin(str):

    if str == "None":

        return None

    elif str == "True":

        return True

    elif str == "False":

        return False

    for type in (int, float):

        try:

            return type(str)

        except ValueError:

            pass

    return str
