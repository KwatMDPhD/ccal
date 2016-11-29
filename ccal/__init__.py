"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

# TODO: import limited functions
from .support import *
from .oncogps import *
from .association import *

VERBOSE = True

RANDOM_SEED = 20121020
print('Planted a random seed: {}.'.format(RANDOM_SEED))
