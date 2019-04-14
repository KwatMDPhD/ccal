from os.path import abspath, expanduser


def get_absolute_path(path):

    return abspath(expanduser(path))
