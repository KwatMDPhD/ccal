from os import walk

from ccal import VERSION
from setuptools import find_packages, setup

setup(
    name="ccal",
    version=VERSION,
    url="https://github.com/KwatME/ccal",
    author="Kwat Medetgul-Ernar (Huwate Yeerna)",
    author_email="kwatme8@gmail.com",
    packages=find_packages(),
    # python_requires=,
    # install_requires=,
    package_data={"ccal": ["data"]},
)
