from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ccal',
    version='0.9.1',
    description='Library for hunting cancers',
    long_description=long_description,
    url='https://github.com/ucsd-ccal/ccal',
    author='Huwate (Kwat) Yeerna',
    author_email='kwatme8@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords='Computational Cancer Analysis',
    packages=find_packages(exclude=[
        'contrib',
        'docs',
        'tests',
    ]),
    python_requires='>=3',
    install_requires=[
        'bcrypt>=3.1.3',
        'biopython>=1.7.0',
        'matplotlib>=2.0.2',
        'pycrypto>=2.6.1',
        'pyfaidx>=0.5.0',
        'pytabix>=0.0.2',
        'rpy2>=2.7.8',
    ])
