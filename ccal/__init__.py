"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo - the author of the version implemented in R
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from .support import read_gct, write_gct, simulate_dataframe_or_series
from .onco_gps import define_components, define_states, make_map
from .association import catalogue, match, compare
from .visualize import plot_clustermap, plot_clusterings, plot_nmf_result, plot_clustering_scores

print('=' * 80)
print('=' * 17 + ' Computational Cancer Analysis Library (CCAL) ' + '=' * 17)
print('=' * 80)

support.plant_seed()
