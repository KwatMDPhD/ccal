"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import argparse

from ...vcf import extract_chr


parser = argparse.ArgumentParser()
parser.add_argument('input_fname', help='Input VCF file {.vcf, .vcf.gz}')
parser.add_argument('chromosome_format', help='Chromosome format {chr#, #}')
args = parser.parse_args()

extract_chr(args.input_fname, args.chromosome_format)
