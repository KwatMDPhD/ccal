from .str_ import str_is_version
from .subprocess_ import run_command


def get_git_versions(sort=True):

    tags = run_command("git tag --list").stdout.strip().split(sep="\n")

    versions = [tag for tag in tags if str_is_version(tag)]

    if sort:

        versions.sort(key=lambda iii: tuple(int(i) for i in iii.split(sep=".")))

    return versions
