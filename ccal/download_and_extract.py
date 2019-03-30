from os import remove
from os.path import splitext
from shutil import unpack_archive

from .download import download


def download_and_extract(url, directory_path):

    file_path = download(url, directory_path)

    unpack_archive(file_path, extract_dir=splitext(file_path)[0])

    remove(file_path)
