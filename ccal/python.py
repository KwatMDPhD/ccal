from .str_ import cast_str_to_builtins
from .subprocess_ import run_command


def get_installed_pip_libraries():

    return [
        line.split()[0]
        for line in run_command("pip list").stdout.strip().split(sep="\n")[2:]
    ]


def install_python_libraries(libraries):

    libraries_installed = get_installed_pip_libraries()

    for library in libraries:

        if library not in libraries_installed:

            run_command("pip install {}".format(library), print_command=True)

        else:

            print("{} is already installed.".format(library))


def get_object_reference(object_, namespace):

    for reference, object__ in namespace.items():

        if object__ is object_:

            return reference

    return cast_str_to_builtins(object_)
