from os.path import join
from urllib.parse import urljoin

from .add_conda_to_path import add_conda_to_path
from .conda_is_installed import conda_is_installed
from .download import download
from .get_machine import get_machine
from .run_command import run_command


def install_and_activate_conda(
    conda_directory_path, pip_installs=None, conda_installs=None
):

    if not conda_is_installed(conda_directory_path):

        conda_script_file_name_template = "Miniconda3-latest-{}-x86_64.sh"

        machine = get_machine()

        if "Darwin" in machine:

            str = "MacOSX"

        elif "Linux" in machine:

            str = "Linux"

        conda_script_file_name = conda_script_file_name_template.format(str)

        tmp_directory_path = join("/", "tmp")

        download(
            urljoin("https://repo.continuum.io/miniconda/", conda_script_file_name),
            tmp_directory_path,
        )

        run_command(
            "bash {} -b -p {}".format(
                join(tmp_directory_path, conda_script_file_name), conda_directory_path
            )
        )

    add_conda_to_path(conda_directory_path)

    if pip_installs is not None:

        run_command("pip install {}".format(" ".join(pip_installs)))

    if conda_installs is not None:

        for channel, packages in conda_installs.items():

            run_command(
                "conda install --channel {} --yes {}".format(
                    channel, " ".join(packages)
                )
            )
