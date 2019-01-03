from platform import uname
from subprocess import CalledProcessError

from .subprocess_ import run_command


def get_machine():

    uname_ = uname()

    return "{}_{}".format(uname_.system, uname_.machine)


def get_shell_environment():

    environemnt = {}

    for line in run_command("env").stdout.split(sep="\n"):

        if line and not line.strip().startswith(":"):

            key, value = line.split(sep="=", maxsplit=1)

            environemnt[key.strip()] = value.strip()

    return environemnt


def have_program(program_name):

    try:

        return bool(run_command("which {}".format(program_name)).stdout.strip())

    except CalledProcessError:

        return False


def shutdown_machine():

    run_command("sudo shutdown -h now")


def reboot_machine():

    run_command("sudo reboot")
