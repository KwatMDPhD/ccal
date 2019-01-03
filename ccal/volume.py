from os.path import isdir

from .subprocess_ import run_command


def mount_volume(volume_name, mount_directory_path):

    if not isdir(mount_directory_path):

        raise ValueError(
            "{0} does not exist. Make it by\n$ sudo mkdir -pv {0}".format(
                mount_directory_path
            )
        )

    run_command("sudo mount {} {}".format(volume_name, mount_directory_path))


def unmount_volume(volume_name_or_mount_directory_path):

    run_command("sudo umount {}".format(volume_name_or_mount_directory_path))


def get_volume_name(volume_label):

    volume_dict = make_volume_dict()

    for volume_name, dict_ in volume_dict.items():

        if dict_.get("LABEL") == volume_label:

            return volume_name


def make_volume_dict():

    volume_dict = {}

    for line in run_command("sudo blkid").stdout.strip("\n").split(sep="\n"):

        line = line.split()

        volume_name = line[0][:-1]

        volume_dict[volume_name] = {}

        for field_value in line[1:]:

            field, value = field_value.replace('"', "").split(sep="=")

            volume_dict[volume_name][field] = value

    return volume_dict
