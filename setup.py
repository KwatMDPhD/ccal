from os import walk
from os.path import abspath, dirname, isdir

from setuptools import setup

NAME = 'ccal'
here = abspath(dirname(__file__))

packages = [NAME]
for location in walk(NAME):

    p = location[0]

    if any([bad_fn in p for bad_fn in ['.git', '__pycache__']]):
        continue

    if isdir(p):
        packages.append(p)

setup(
    name=NAME,
    version='0.0.2',
    description='Library for hunting cancers',
    long_description='See https://github.com/ucsd-ccal/ccal for documentation.',
    url='https://github.com/ucsd-ccal/ccal',
    author='Huwate (Kwat) Yeerna',
    author_email='kwatme8@gmail.com',
    license='LICENSE',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords='Computational Cancer Analysis',
    packages=packages,
    python_requires='>=3.6.0',
    install_requires=[
        'bcrypt>=3.1.0, <3.2.0',
        'biopython>=1.70.0, <1.71.0',
        'matplotlib>=2.0.0, <2.1.0',
        'pandas>=0.20.0, <0.21.0',
        'pycrypto>=2.6.0, <2.7.0',
        'pyfaidx>=0.5.0, <0.6.0',
        'pytabix>=0.0.2, <0.1.0',
        'scikit-learn>=0.19.0, <0.19.2',
        'scipy>=0.19.0, <0.20.0',
        'seaborn>=0.8.0, <0.9.0',
        'statsmodels>=0.8.0, <0.9.0',
    ],
    # conda install -c r rpy2 r-mass
    include_package_data=True)
