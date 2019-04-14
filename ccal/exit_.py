from sys import exit

from .echo_or_print import echo_or_print


def exit_(str, exception=None):

    echo_or_print("Uh oh :( ... {}".format(str), fg="red", bg="black")

    if exception is None:

        exit()

    else:

        raise exception
