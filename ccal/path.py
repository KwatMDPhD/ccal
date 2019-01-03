from os import listdir, mkdir, remove
from os.path import abspath, exists, expanduser, isdir, isfile, islink, split
from shutil import copy, copytree, rmtree
from warnings import warn


def establish_path(path, path_type, print_=True):

    path = clean_path(path)

    if path_type == "file":

        if path.endswith("/"):

            raise ValueError("File path {} should not end with '/'.".format(path))

    elif path_type == "directory":

        if not path.endswith("/"):

            path += "/"

    else:

        raise ValueError("Unknown path_type: {}.".format(path_type))

    directory_path, file_name = split(path)

    missing_directory_paths = []

    while not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        directory_path, file_name = split(directory_path)

    for directory_path in reversed(missing_directory_paths):

        mkdir(directory_path)

        if print_:

            print("Created directory {}.".format(directory_path))


def copy_path(from_path, to_path, overwrite=False, print_=True):

    if overwrite:

        remove_path(to_path, print_=print_)

    if isdir(from_path):

        copytree(from_path, to_path)

        copied_path = True

    elif exists(from_path):

        copy(from_path, to_path)

        copied_path = True

    else:

        warn("Could not copy {} because it does not exist.".format(from_path))

        copied_path = False

    if copied_path and print_:

        print("Copied {} =(to)=> {}.".format(from_path, to_path))


def remove_paths(directory_path, path_type, print_=True):

    for name in listdir(directory_path):

        path = "{}/{}".format(directory_path, name)

        if path_type not in ("file", "directory", "link"):

            raise ValueError("Unknown path_type: {}.".format(path_type))

        if (
            (path_type == "file" and isfile(path))
            or (path_type == "directory" and isdir(path))
            or (path_type == "link" and islink(path))
        ):

            remove_path(path, print_=print_)


def remove_path(path, print_=True):

    if isdir(path):

        rmtree(path)

        removed_path = True

    elif exists(path):

        remove(path)

        removed_path = True

    else:

        warn("Could not remove {} because it does not exist.".format(path))

        removed_path = False

    if removed_path and print_:

        print("Removed {}.".format(path))


def clean_path(path):

    return abspath(expanduser(path))


def clean_name(name):

    cleaned_name = ""

    for character in name:

        if character.isalnum():

            cleaned_name += character

        else:

            cleaned_name += "_"

    return cleaned_name
