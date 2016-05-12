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



## Define Global variable
TESTING = False
# Path to CCBA dicrectory (repository)
PATH_CCBA = '/Users/Kwat/binf/ccba/'
# Path to testing data directory
PATH_TEST_DATA = os.path.join(PATH_CCBA, 'data', 'test')