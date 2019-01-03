from datetime import datetime
from io import UnsupportedOperation
from logging import FileHandler, Formatter, StreamHandler, getLogger
from random import choice

from click import secho


def get_now(only_time=False):

    if only_time:

        formatter = "%H:%M:%S"

    else:

        formatter = "%Y-%m-%d %H:%M:%S"

    return datetime.now().strftime(formatter)


def initialize_logger(name):

    logger = getLogger(name)

    logger.setLevel(10)

    fh = FileHandler("/tmp/{}.{:%Y:%m:%d:%H:%M:%S}.log".format(name, datetime.now()))

    fh.setFormatter(Formatter("%(asctime)s|%(levelname)s: %(message)s\n", "%H%M%S"))

    logger.addHandler(fh)

    sh = StreamHandler()

    sh.setFormatter(Formatter("%(levelname)s: %(message)s\n"))

    logger.addHandler(sh)

    logger.info("Initialized {} logger.".format(name))

    return logger


def echo_or_print(
    text,
    fg=None,
    bg=None,
    bold=None,
    dim=None,
    underline=None,
    blink=None,
    reverse=None,
    reset=True,
):

    if fg == "random":

        fg = choice(("green", "yellow", "blue", "magenta", "cyan", "white"))

        bg = "black"

    try:

        secho(
            text,
            fg=fg,
            bg=bg,
            bold=bold,
            dim=dim,
            underline=underline,
            blink=blink,
            reverse=reverse,
            reset=reset,
        )

    except UnsupportedOperation:

        print(text)


def log_and_return_response(response, logger=None):

    str_ = response.get_data().decode().strip()

    if logger is None:

        print(str_)

    else:

        logger.debug(str_)

    return response
