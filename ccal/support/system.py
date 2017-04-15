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

from os import environ
from subprocess import PIPE, Popen, run

from numpy.random import get_state, seed
from pip import get_installed_distributions, main


def run_command(command):
    """
    Execute cmd.
    :param command: str;
    :return: str;
    """

    print(command)
    output = run(command,
                 shell=True,
                 check=True,
                 stdout=PIPE,
                 universal_newlines=True)
    return output


def install_libraries(libraries_needed):
    """
    Check if libraries_needed are installed; if not, install using pip.
    :param libraries_needed: iterable; library names
    :return: None
    """

    print('Trying to install ({}) ...'.format(', '.join(libraries_needed)))

    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]

    # If any of the libraries_needed is not in the currently installed libraries,
    # then install it using pip
    for lib in libraries_needed:
        if lib not in libraries_installed:
            print('{} not found; installing it using pip ...'.format(lib))
            main(['install', lib])


def source_environment(filepath):
    """
    Update environment using source_environment.
    :param filepath:
    :return: None
    """

    print('Sourcing {} ...'.format(filepath))

    for line in Popen(
            './{}; env'.format(filepath),
            stdout=PIPE,
            universal_newlines=True,
            shell=True).stdout:
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        environ[key] = value
        print('\t{} = {}'.format(key, value))


def get_name(obj, namesapce):
    """

    :param obj: object;
    :param namesapce: dict;
    :return: str;
    """

    # TODO: print non-strings as non-strings

    for obj_name_in_namespace, obj_in_namespace in namesapce.items():
        if obj_in_namespace is obj:  # obj is a existing obj
            return obj_name_in_namespace

    # obj is a str
    return '\'{}\''.format(obj)


def print_random_state(mark='', print_process=False):
    """
    Print numpy random state.
    :param mark: str;
    :return: None
    """

    random_state = get_state()

    _, keys, pos, _, _ = random_state
    try:
        print(
            '[{}] Seed0={}\ti={}\t@={}'.format(mark, keys[0], pos, keys[pos]),
            print_process=print_process)
    except IndexError:
        print(
            '[{}] Seed0={}\ti={}'.format(mark, keys[0], pos),
            print_process=print_process)


def skip_random_states(random_seed, n_skips, skipper='', for_skipper=None):
    """

    :param random_seed: int;
    :param skipper: str;
    :param for_skipper: object;
    :return: list; list of arrays;
    """

    seed(random_seed)
    r = get_state()

    for i in range(n_skips):
        exec(skipper)
        r = get_state()

    return r
