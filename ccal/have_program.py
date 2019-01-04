from subprocess import CalledProcessError

from .subprocess_ import run_command


def have_program(program_name):

    try:

        return bool(run_command("which {}".format(program_name)).stdout.strip())

    except CalledProcessError:

        return False
