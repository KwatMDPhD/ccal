def is_str_version(str):

    str_split = str.split(sep=".")

    return "." in str and len(str_split) == 3 and all(i.isnumeric() for i in str_split)
