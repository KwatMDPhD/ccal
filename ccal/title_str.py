def title_str(str):

    original_uppers = []

    on_upper = False

    upper_start = None

    upper_end = None

    for i, character in enumerate(str):

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

                upper_start = None

                upper_end = None

    if upper_start:

        original_uppers.append((upper_start, upper_end))

    str = str.title().replace("_", " ")

    for upper_start, upper_end in original_uppers:

        str = str[:upper_start] + str[upper_start:upper_end].upper() + str[upper_end:]

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

        str = str.replace(
            " {} ".format(lowercase_character.title()),
            " {} ".format(lowercase_character),
        )

    return " ".join(sub_str.strip() for sub_str in str.split())
