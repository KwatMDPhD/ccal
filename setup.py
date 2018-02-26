from os import walk
from os.path import join

from setuptools import setup

name = 'ccal'
url = 'https://github.com/UCSD-CCAL/ccal'

directory_names_to_skip = (
    '.git',
    '__pycache__', )

packages = []
for dp, dns, fns in walk(name):
    if dp.split('/')[-1] not in directory_names_to_skip:
        packages.append(dp)

package_data = []
for dp, dns, fns in walk(join(name, 'sequencing_process', 'resources')):
    if dp.split(sep='/')[-1] not in directory_names_to_skip:
        for fn in fns:
            package_data.append(join(dp.split(sep='/', maxsplit=1)[1], fn))

setup(
    name=name,
    version='0.4.2',
    description=
    'Computational Cancer Analysis Library: bioinformatics library for hunting cancers',
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
        'Topic :: Scientific/Engineering :: Bio-Informatics', ),
    keywords='Computational Cancer Analysis',
    packages=packages,
    python_requires='>=3.3',
    install_requires=(
        'biopython>=1.70.0',
        'click>=6.7.0',
        'geoparse>=1.0.5',
        'matplotlib>=2.1.1',
        'numpy>=1.12.1',
        'pandas>=0.22.0',
        'pycrypto>=2.6.1',
        'scikit-learn>=0.19.1',
        'scipy>=1.0.0',
        'seaborn>=0.8.1',
        'statsmodels>=0.8.0', ),
    # And must install manually: $ conda install -c conda-forge rpy2 r-mass
    package_data={
        'ccal': package_data,
    },
    include_package_data=True)
