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

from operator import add, sub


def merge_dicts(*dicts):
    """
    Shallow copy and merge dicts into a new dict; precedence goes to
    key value pairs in latter dict. Only keys in both dicts will be kept.
    :param dicts: iterable of dict;
    :return: dict;
    """

    merged = dict()
    for d in dicts:
        merged.update(d)

    return merged


def dict_add(dict_1, dict_2):
    """
    Add dict_1 and dict_2.
    :param dict_1:
    :param dict_2:
    :return:
    """

    return merge_dicts_with_function(add, dict_1, dict_2)


def dict_subtract(dict_1, dict_2):
    """
    Subtract dict_2 from dict_1.
    :param dict_1:
    :param dict_2:
    :return:
    """

    return merge_dicts_with_function(sub, dict_1, dict_2)


def merge_dicts_with_function(function, dict_1, dict_2):
    """
    Apply function to values keyed by the same key in dict_1 and dict_2.
    :param function: function;
    :param dict_1: dict;
    :param dict_2: dict;
    :return: dict; merged dict
    """

    new_dict = {}
    all_keys = set(dict_1.keys()).union(dict_2.keys())
    for k in all_keys:
        if k in dict_1 and k in dict_2:
            new_dict[k] = function(dict_1[k], dict_2[k])
        elif k in dict_1:
            new_dict[k] = dict_1[k]
        else:
            new_dict[k] = dict_2[k]
    return new_dict
