from re import sub


def normalize_file_name(str):

    return sub(r"(?u)[^-\w.]", "_", str.strip().replace(" ", "_"))
