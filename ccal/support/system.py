"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from multiprocessing import Pool
from os import environ
from subprocess import PIPE, run, Popen

from pip import get_installed_distributions, main

from .log import print_log


def install_libraries(libraries_needed):
    """
    Check if libraries_needed are installed; if not, install using pip.
    :param libraries_needed: iterable; library names
    :return: None
    """

    print_log('Checking library dependencies ...')

    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]

    # If any of the libraries_needed is not in the currently installed libraries, then install it using pip
    for lib in libraries_needed:
        if lib not in libraries_installed:
            print_log('{} not found; installing it using pip ...'.format(lib))
            main(['install', lib])


def source_environment(filepath):
    """
    Update environment using source_environment.
    :param filepath:
    :return: None
    """

    print_log('Sourcing {} ...'.format(filepath))

    for line in Popen('./{}; env'.format(filepath), stdout=PIPE, universal_newlines=True, shell=True).stdout:
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        environ[key] = value
        print_log('\t{} = {}'.format(key, value))


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


def parallelize(function, list_of_args, n_jobs=1):
    """
    Apply function on list_of_args using parallel computing across n_jobs jobs; n_jobs doesn't have to be the length of
    list_of_args.
    :param function: function;
    :param list_of_args: iterable; for each item in iterable, function(item) is executed; function won't run if empty
    :param n_jobs: int; number of allowed simultaneous jobs
    :return: list; list of outputs returned by all jobs, in the order of the list_of_args
    """

    with Pool(n_jobs) as p:
        return p.map(function, list_of_args)


def run_cmd(cmd):
    """
    Execute cmd.
    :param cmd: str;
    :return: str;
    """

    print(cmd)
    output = run(cmd, shell=True, check=True, stdout=PIPE, universal_newlines=True)
    return output
