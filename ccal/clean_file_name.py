from re import sub


def clean_file_name(str):

    return sub(r"(?u)[^-\w.]", "", str.strip().replace(" ", "_"))
