from re import sub


def cast_str_to_builtins(str_):

    if str_ == "None":

        return None

    elif str_ == "True":

        return True

    elif str_ == "False":

        return False

    for type_ in (int, float):

        try:

            return type_(str_)

        except ValueError:

            pass

    return str_


def title_str(str_):

    original_uppers = []

    on_upper = False

    upper_start = upper_end = None

    for i, character in enumerate(str_):

        if character.isupper():

            if on_upper:

                upper_end += 1

            else:

                on_upper = True

                upper_start = i

                upper_end = upper_start + 1

        else:

            if on_upper:

                on_upper = False

                original_uppers.append((upper_start, upper_end))

                upper_start = upper_end = None

    if upper_start:

        original_uppers.append((upper_start, upper_end))

    str_ = str_.title().replace("_", " ")

    for upper_start, upper_end in original_uppers:

        str_ = (
            str_[:upper_start] + str_[upper_start:upper_end].upper() + str_[upper_end:]
        )

    for lowercase_character in (
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "for",
        "nor",
        "on",
        "at",
        "to",
        "from",
        "of",
        "vs",
    ):

        str_ = str_.replace(
            " {} ".format(lowercase_character.title()),
            " {} ".format(lowercase_character),
        )

    return " ".join(sub_str.strip() for sub_str in str_.split())


def untitle_str(str_):

    return str_.lower().replace(" ", "_").replace("-", "_")


def split_str_ignoring_inside_quotes(str_, separator):

    splits = []

    part = ""

    for str_split in str_.split(separator):

        if '"' in str_split:

            if part:

                part += str_split

                splits.append(part)

                part = ""

            else:

                part = str_split

        else:

            if part:

                part += str_split

            else:

                splits.append(str_split)

    if part:

        splits.append(part)

    return splits


def str_is_version(str_):

    return (
        "." in str_
        and len(str_.split(sep=".")) == 3
        and all(i.isnumeric() for i in str_.split(sep="."))
    )


def make_file_name_from_str(str_):

    return sub(r"(?u)[^-\w.]", "", str_.strip().replace(" ", "_"))
