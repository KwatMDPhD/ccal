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

        if 'sequencing_process/resource' not in directory_path:

            packages.append(directory_path)

package_data = []

for directory_path, directory_names, file_names in walk(
        '{}/sequencing_process/resource'.format(name)):

    if not any(str_ in directory_path for str_ in strs_to_skip):

        for file_name in file_names:

            package_data.append('{}/{}'.format(
                directory_path.split(sep='/', maxsplit=1)[1],
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
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ),
    keywords='Data Craft',
    packages=packages,
    python_requires='>=3.5',
    install_requires=(
        'biopython>=1.70.0',
        'click>=6.7.0',
        'geoparse>=1.0.5',
        'matplotlib>=2.1.1',
        'numpy>=1.12.1',
        'pandas>=0.23.0',
        'pefile>=2017.8.1',
        'plotly>=2.5.1',
        'pycrypto>=2.6.1',
        'pyfaidx>=0.5.4.1',
        'pytabix>=0.0.2',
        'scikit-learn>=0.19.1',
        'scipy>=1.1.0',
        'statsmodels>=0.8.0',
    ),
    package_data={'ccal': package_data},
    include_package_data=True,
)
