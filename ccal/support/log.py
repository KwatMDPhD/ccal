"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from datetime import datetime

from .. import VERBOSE


def print_log(string):
    """
    Print string together with logging information.
    :param string: str; message to printed
    :return: None
    """

    # TODO: use logging (https://docs.python.org/3.5/howto/logging.html)
    if VERBOSE:
        print('<{}> {}'.format(timestamp(time_only=True), string))


def timestamp(time_only=False):
    """
    Get the current time.
    :param time_only: bool; exclude year, month, and date or not
    :return: str; the current time
    """

    if time_only:
        formatter = '%H%M%S'
    else:
        formatter = '%Y%m%d%H%M%S'
    return datetime.now().strftime(formatter)
