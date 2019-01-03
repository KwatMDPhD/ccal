from subprocess import CalledProcessError

from .log import echo_or_print
from .str_ import str_is_version
from .subprocess_ import run_command


def create_gitkeep(directory_path):

    gitkeep_file_path = "{}/.gitkeep".format(directory_path)

    open(gitkeep_file_path, mode="w").close()

    print("Created {}.".format(gitkeep_file_path))


def get_git_versions(sort=True):

    tags = run_command("git tag --list").stdout.strip().split(sep="\n")

    versions = [tag for tag in tags if str_is_version(tag)]

    if sort:

        versions.sort(key=lambda iii: tuple(int(i) for i in iii.split(sep=".")))

    return versions


def clean_git_url(git_url):

    if git_url.endswith("/"):

        git_url = git_url[:-1]

    if git_url.endswith(".git"):

        git_url = git_url[:-4]

    return git_url


def in_git_repository():

    echo_or_print("Checking if in git repository ...")

    try:

        run_command("git status")

        return True

    except CalledProcessError:

        return False
