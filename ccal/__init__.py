"""
Computational Cancer Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov


Description:
Check dependencies and install missing ones.
"""
from . import support
from . import visualize
from . import information
from . import analyze

print('=' * 79)
print('=' * 20 + ' Computational Cancer Analysis Library ' + '=' * 20)
print('=' * 79)

support.install_libraries(['rpy2', 'numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'matplotlib', 'seaborn'])
support.plant_seed()
