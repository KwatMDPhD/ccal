from .subprocess_ import run_command

from .get_installed_pip_libraries import get_installed_pip_libraries


def install_python_libraries(libraries):

    libraries_installed = get_installed_pip_libraries()

    for library in libraries:

        if library not in libraries_installed:

            run_command("pip install {}".format(library), print_command=True)

        else:

            print("{} is already installed.".format(library))
