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

VERBOSE = True
RANDOM_SEED = 20121020

import sys
sys.setrecursionlimit(10000)

from .computational_cancer_biology import association, imv, oncogps
from .support.plot import plot_heatmap, plot_clustermap, plot_points, plot_distribution, plot_violin_or_box
from .support.file import read_gct, write_gct, read_gmt, read_gmts, write_rnk, load_ccle_bundle
