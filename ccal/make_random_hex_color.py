from numpy.random import choice


def make_random_hex_color():

    return "#{}{}{}{}{}{}".format(*choice(list("0123456789abcdef"), size=6))
