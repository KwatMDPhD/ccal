"""
Cancer Computational Biology Analysis Library v0.1

Authors:
1) Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

2) Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center
"""
import os
from libary.support import *
from libary.visualize import *



## Define Global variable
TESTING = False
# Path to CCBA dicrectory (repository)
PATH_CCBA = '/Users/Kwat/binf/ccba/'
# Path to testing data directory
PATH_TEST_DATA = os.path.join(PATH_CCBA, 'data', 'test')