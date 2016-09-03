"""
Computational Cancer Analysis Library v0.1

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov
"""
from . import support
from . import information
from . import analyze
from . import visualize
from . import onco_gps
from . import onco_match

print('=' * 79)
print('=' * 20 + ' Computational Cancer Analysis Library ' + '=' * 20)
print('=' * 79)

support.install_libraries(['rpy2', 'numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'matplotlib', 'seaborn'])
support.plant_seed()
