"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""


def title_str(string):
    """
    Title a a_str.
    :param string: str;
    :return: str;
    """

    # Remember indices of original uppercase letters
    uppers = []
    start = end = None
    is_upper = False
    for i, c in enumerate(string):
        if c.isupper():
            # print('{} is UPPER.'.format(c))
            if is_upper:
                end += 1
            else:
                is_upper = True
                start = i
                end = start + 1
                # print('Start is {}.'.format(i))
        else:
            if is_upper:
                is_upper = False
                uppers.append((start, end))
                start = None
                end = start
    else:
        if start:
            uppers.append((start, end))

    # Title
    string = string.title().replace('_', ' ')

    # Upper all original uppercase letters
    for start, end in uppers:
        string = string[:start] + string[start: end].upper() + string[end:]

    # Lower some words
    for lowercase in ['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'of', 'vs', 'vs']:
        string = string.replace(' ' + lowercase.title() + ' ', ' ' + lowercase + ' ')

    return string


def untitle_str(string):
    """
    Untitle a string.
    :param string: str;
    :return: str;
    """

    string = str(string)
    return string.lower().replace(' ', '_').replace('-', '_')


def clean_str(string, illegal_chars=(' ', '\t', ',', ';', '|'), replacement_char='_'):
    """
    Return a copy of string that has all non-allowed characters replaced by a new character (default: underscore).
    :param string:
    :param illegal_chars:
    :param replacement_char:
    :return:kkkkkuu
    """

    new_string = str(string)
    for illegal_char in illegal_chars:
        new_string = new_string.replace(illegal_char, replacement_char)
    return new_string


def cast_str_to_int_float_bool_or_str(string):
    """
    Convert string into the following data types (return the first successful): int, float, bool, or str.
    :param string: str;
    :return: int, float, bool, or str;
    """

    value = string.strip()

    for var_type in [int, float]:
        try:
            converted_var = var_type(value)
            return converted_var
        except ValueError:
            pass

    if value == 'True':
        return True
    elif value == 'False':
        return False

    return str(value)


def indent_str(string, n_tabs=1):
    """
    Indent block of text by adding a n_tabs number of tabs (default 1) to the beginning of each line.
    :param string:
    :param n_tabs:
    :return:
    """

    # TODO: consider deleting
    return '\n'.join(['\t' * n_tabs + line for line in string.split('\n')])


def reset_encoding(a_str):
    """

    :param a_str: str;
    :return: str;
    """

    return a_str.replace(u'\u201c', '"').replace(u'\u201d', '"')
