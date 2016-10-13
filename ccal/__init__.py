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

from random import seed

SEED = 20121020
seed(SEED)

VERBOSE = True

from .support import read_gct, write_gct
from .oncogps import define_components, project_w, define_states, make_oncogps_map
from .association import plot_association_summary_panel, make_association_panel, make_association_panels, \
    make_comparison_matrix

print('=' * 80)
print('=' * 17 + ' Computational Cancer Analysis Library (CCAL) ' + '=' * 17)
print('=' * 80)
print('Planted a random seed: {}.'.format(SEED))
