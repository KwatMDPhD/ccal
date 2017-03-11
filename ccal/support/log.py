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

from datetime import datetime
from multiprocessing import current_process

VERBOSE = True


# TODO: use logging (https://docs.python.org/3.5/howto/logging.html)
def print_log(string, print_process=False):
    """
    Print string together with logging information.
    :param string: str; message to printed
    :param print_process: bool;
    :return: None
    """

    if VERBOSE:
        to_print = '[{}] {}'.format(timestamp(time_only=False), string)
        if print_process:
            to_print = '<{}> {}'.format(current_process().name, to_print)
        print(to_print)


def timestamp(time_only=False):
    """
    Get the current time.
    :param time_only: bool; exclude year, month, and date or not
    :return: str; the current time
    """

    if time_only:
        formatter = '%H:%M:%S'
    else:
        formatter = '%Y-%m-%d %H:%M:%S'
    return datetime.now().strftime(formatter)
