from os.path import isdir, join


def conda_is_installed(conda_directory_path):

    return all(isdir(join(conda_directory_path, name)) for name in ("bin", "lib"))
