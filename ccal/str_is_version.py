def str_is_version(str):

    return (
        "." in str
        and len(str.split(sep=".")) == 3
        and all(i.isnumeric() for i in str.split(sep="."))
    )
