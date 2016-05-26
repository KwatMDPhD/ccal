"""
Cancer Computational Biology Analysis Library v0.1

Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

James Jensen
Email
Affiliation
"""
import os
import numpy as np
import pandas as pd
from library.support import *
from library.visualize import *
from library.information import *



## Define Global variable
TESTING = False
# Path to CCBA dicrectory (repository)
PATH_CCBA = '/Users/Kwat/binf/ccba/'
# Path to testing data directory
PATH_TEST_DATA = os.path.join(PATH_CCBA, 'data', 'test')


def make_heatmap_panel(reference, dataframe, annotation_columns, title=None):
    """
    """
    if 'IC' in annotation_columns:
        dataframe.ix[:, 'IC'] = pd.Series([compute_information_coefficient(np.array(row[1]), reference) for row in dataframe.iterrows()], index=dataframe.index)
    dataframe.sort(['IC'], inplace=True)
    plot_heatmap_panel(reference, dataframe, annotation_columns, title=title)  