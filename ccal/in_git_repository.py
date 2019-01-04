from subprocess import CalledProcessError

from .log import echo_or_print
from .subprocess_ import run_command


def in_git_repository():

    echo_or_print("Checking if in git repository ...")

    try:

        run_command("git status")

        return True

    except CalledProcessError:

        return False
