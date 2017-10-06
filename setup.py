from os import listdir
from os.path import abspath, dirname, isdir, join

from setuptools import setup

NAME = 'ccal'
here = abspath(dirname(__file__))

packages = [NAME]
for fn in listdir(NAME):

    if fn not in ['__pycache__'] and isdir(join(here, NAME, fn)):
        packages += ['{0}/{1}/{1}'.format(NAME, fn)]

setup(
    name='ccal',
    version='0.8.8',
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
        'bcrypt>=3.1.3',
        'biopython>=1.7.0',
        'matplotlib>=2.0.2',
        'pycrypto>=2.6.1',
        'pyfaidx>=0.5.0',
        'pytabix>=0.0.2',
    ],
    include_package_data=True)
