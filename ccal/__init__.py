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

SEED = 20121020
print('Planted a random seed: {}.'.format(SEED))

VERBOSE = True

from .support import read_gct, write_gct, write_rnk
from .oncogps import define_components, get_w_or_h_matrix, solve_for_components, define_states, get_state_labels, \
    make_oncogps_map
from .association import compute_association, make_association_panel, make_association_panels, \
    plot_association_summary_panel, make_comparison_matrix
