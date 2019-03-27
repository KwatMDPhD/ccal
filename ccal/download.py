import ssl
from os.path import basename, join
from urllib.parse import urlsplit
from urllib.request import urlretrieve


def download(url, directory_path):

    file_name = basename(urlsplit(url).path)

    file_path = join(directory_path, file_name)

    ssl._create_default_https_context = ssl._create_unverified_context

    urlretrieve(url, file_path)

    return file_path
