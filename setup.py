from setuptools import find_packages, setup

na = "kwat"

setup(
    name=na,
    version="0.5.0",
    url="https://github.com/KwatME/kwat.py",
    python_requires=">=3.6.0",
    install_requires=[
        "numpy",
        "pandas",
        "xlrd",
        "openpyxl",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "KDEpy",
        "plotly",
        "click",
        "requests",
    ],
    packages=find_packages(),
    package_data={na: ["data/*"]},
    entry_points={
        "console_scripts": ["{0}={0}.{1}:{1}".format(na, "cli")],
    },
)
