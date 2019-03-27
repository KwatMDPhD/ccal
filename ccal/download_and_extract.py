from os import remove
from os.path import basename, join, splitext
from shutil import unpack_archive

from .download import download


def download_and_extract(url, directory_path):

    file_path = download(url, directory_path)

    file_name = basename(file_path)

    unpack_archive(file_path, extract_dir=join(directory_path, splitext(file_name)[0]))

    remove(file_path)
