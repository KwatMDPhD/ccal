from os import mkdir
from os.path import isdir, split

from .get_absolute_path import get_absolute_path


def establish_path(path, path_type):

    path = get_absolute_path(path)

    if path_type == "file":

        if path.endswith("/"):

            raise ValueError("File path {} should not end with '/'.".format(path))

    elif path_type == "directory":

        if not path.endswith("/"):

            path += "/"

    directory_path, file_name = split(path)

    missing_directory_paths = []

    while not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        directory_path, file_name = split(directory_path)

    for directory_path in reversed(missing_directory_paths):

        mkdir(directory_path)

        print("Created directory {}.".format(directory_path))
