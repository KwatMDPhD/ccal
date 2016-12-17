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

VERBOSE = False
RANDOM_SEED = 20121020

from .computational_cancer_biology import association
from .computational_cancer_biology import essentiality
from .computational_cancer_biology import oncogps
from .support.file import read_gct, write_gct, read_gmt, read_gmts_and_collapse, write_rnk
from .support.plot import plot_heatmap, plot_clustermap, plot_x_vs_y
