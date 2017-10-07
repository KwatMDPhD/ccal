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
    name='ccal',
    version='0.9.5',
    description='Library for hunting cancers',
    long_description='',
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
    python_requires='>=3',
    install_requires=[
        # 'bcrypt>=3.1.3',
        # 'biopython>=1.7.0',
        # 'matplotlib>=2.0.2',
        # 'pycrypto>=2.6.1',
        # 'pyfaidx>=0.5.0',
        # 'pytabix>=0.0.2',
        # 'rpy2>=2.7.8',
        # 'scipy>=0.19.1',
        # 'seaborn>=0.8.1',
    ],
    include_package_data=True)
