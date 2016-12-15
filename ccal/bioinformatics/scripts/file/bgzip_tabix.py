"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import argparse

from ....support.file import bgzip_tabix


parser = argparse.ArgumentParser()
parser.add_argument('input_fname', help='Input file')
args = parser.parse_args()

bgzip_tabix(args.input_fname)
