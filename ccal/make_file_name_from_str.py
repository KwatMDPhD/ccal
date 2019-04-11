from re import sub


def make_file_name_from_str(str):

    return sub(r"(?u)[^-\w.]", "", str.strip().replace(" ", "_"))
