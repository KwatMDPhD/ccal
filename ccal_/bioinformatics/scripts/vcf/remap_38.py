"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import argparse

from ...vcf import remap_38


parser = argparse.ArgumentParser()
parser.add_argument('input_fname', help='Input file')
parser.add_argument('source_assembly', help='Source genomic assembly {grch37, hg19}.')
args = parser.parse_args()

remap_38(args.input_fname, args.source_assembly)
