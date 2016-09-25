"""
Computational Cancer Analysis Library

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis Laboratory, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis Laboratory, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov

Import missing dependencies. And import functional modules.
"""
# TODO: Optimize import

from . import support

# Install libraries that are not in Anaconda3 distribution
support.install_libraries(['rpy2', 'seaborn'])

from . import visualize
from . import onco_gps
from . import association

print('=' * 80)
print('=' * 17 + ' Computational Cancer Analysis Library (CCAL) ' + '=' * 17)
print('=' * 80)

support.plant_seed()
