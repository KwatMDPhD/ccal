"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import re


def title_str(str_):
    """
    Title a a_str.
    :param str_: str;
    :return: str;
    """

    # Remember indices of original uppercase letters
    uppers = []
    start = end = None
    is_upper = False
    for i, c in enumerate(str_):
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
            if is_upper:  # Reset
                is_upper = False
                uppers.append((start, end))
                start = None
                end = start
    else:
        if isinstance(start, int):
            uppers.append((start, end))

    # Title
    str_ = str_.title().replace('_', ' ')

    # Upper all original uppercase letters
    for start, end in uppers:
        str_ = str_[:start] + str_[start:end].upper() + str_[end:]

    # Lower some words
    for lowercase in [
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at',
            'to', 'from', 'of', 'vs', 'vs'
    ]:
        str_ = str_.replace(' ' + lowercase.title() + ' ',
                            ' ' + lowercase + ' ')

    return str_


def untitle_str(str_):
    """
    Untitle a string.
    :param str_: str;
    :return: str;
    """

    str_ = str(str_)
    return str_.lower().replace(' ', '_').replace('-', '_')


def clean_str(str_,
              illegal_chars=(' ', '\t', ',', ';', '|'),
              replacement_char='_'):
    """
    Return a copy of string that has all non-allowed characters replaced by a new character (default: underscore).
    :param str_:
    :param illegal_chars:
    :param replacement_char:
    :return: str;
    """

    new_string = str(str_)
    for illegal_char in illegal_chars:
        new_string = new_string.replace(illegal_char, replacement_char)
    return new_string


def cast_str_to_int_float_bool_or_str(str_, fmt='{:.3f}'):
    """
    Convert string into the following data types (return the first successful): int, float, bool, or str.
    :param str_: str;
    :param fmt: str; formatter for float
    :return: int, float, bool, or str;
    """

    value = str(str_).strip()

    for var_type in [int, float]:
        try:
            converted_var = var_type(value)
            if var_type == float:
                converted_var = fmt.format(converted_var)
            return converted_var
        except ValueError:
            pass

    if value == 'True':
        return True
    elif value == 'False':
        return False

    return str(value)


def remove_nested_quotes(str_):
    """

    :param str_:
    :return:
    """

    if isinstance(str_, str):
        str_ = re.sub(r'^"|"$|^\'|\'$', '', str_)
    return str_


def reset_encoding(str_):
    """

    :param str_: str;
    :return: str;
    """

    return str_.replace(u'\u201c', '"').replace(u'\u201d', '"')


def split_ignoring_inside_quotes(str_, sep):
    """
    Split str_ by sep not in quotes.
    :param str_: str;
    :param sep: str;
    :return: list; list of str
    """

    split = str_.split(sep)
    to_return = []
    temp = ''
    for s in split:
        if '"' in s:
            if temp:
                temp += s
                to_return.append(temp)
                temp = ''
            else:
                temp += s
        else:
            if temp:
                temp += s
            else:
                to_return.append(s)
    return to_return


def sort_numerically(string_iterable, reverse=False):
    """

    :param string_iterable:
    :param reverse:
    :return:
    """
    digit_parser = re.compile(r'[A-Za-z]+|\d+')

    def maybe_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    return sorted(
        string_iterable,
        key=lambda x: [maybe_int(s) for s in re.findall(digit_parser, x)],
        reverse=reverse)
