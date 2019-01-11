from os import walk

from ccal import VERSION
from setuptools import setup

package_data = []

for directory_path, directory_names, file_names in walk("data"):

    for file_name in file_names:

        package_data.append("{}/{}".format(directory_path, file_name))

setup(
    name="ccal",
    version=VERSION,
    description="Computational Cancer Analysis Library",
    url="https://github.com/KwatME/ccal",
    author="Kwat Medetgul-Ernar (Huwate Yeerna)",
    author_email="kwatme8@gmail.com",
    license="LICENSE",
    classifiers=(
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ),
    python_requires=">=3.6",
    install_requires=(),
    include_package_data=True,
    package_data={"ccal": package_data},
)
