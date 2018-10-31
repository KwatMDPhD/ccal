from os import walk

from ccal import VERSION
from setuptools import setup

name = 'ccal'

url = 'https://github.com/UCSD-CCAL/ccal'

strs_to_skip = (
    '.git',
    '__pycache__',
)

packages = []

for directory_path, directory_names, file_names in walk(name):

    if not any(str_ in directory_path for str_ in strs_to_skip):

        if '/data/' not in directory_path:

            packages.append(directory_path)

package_data = []

for directory_path, directory_names, file_names in walk(
        '{}/sequencing_process/data'.format(name)):

    if not any(str_ in directory_path for str_ in strs_to_skip):

        for file_name in file_names:

            package_data.append('{}/{}'.format(
                directory_path.split(
                    sep='/',
                    maxsplit=1,
                )[1],
                file_name,
            ))

for directory_path, directory_names, file_names in walk(
        '{}/gene/data'.format(name)):

    if not any(str_ in directory_path for str_ in strs_to_skip):

        for file_name in file_names:

            package_data.append('{}/{}'.format(
                directory_path.split(
                    sep='/',
                    maxsplit=1,
                )[1],
                file_name,
            ))

setup(
    name=name,
    version=VERSION,
    description='Computational Cancer Analysis Library',
    long_description='See {} to learn more.'.format(url),
    url=url,
    author='(Kwat) Huwate Yeerna',
    author_email='kwatme8@gmail.com',
    license='LICENSE',
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ),
    packages=packages,
    python_requires='>=3.6',
    install_requires=(),
    package_data={'ccal': package_data},
    include_package_data=True,
)
