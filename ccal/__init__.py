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
from .support import read_gct, write_gct, simulate_dataframe_or_series, plot_clustermap, plot_nmf_result, plot_x_vs_y
from .onco_gps import define_components, define_states, make_map
from .association import catalogue, make_match_panel, compare

print('=' * 80)
print('=' * 17 + ' Computational Cancer Analysis Library (CCAL) ' + '=' * 17)
print('=' * 80)
