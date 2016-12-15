"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import argparse

from ...vcf import isec


parser = argparse.ArgumentParser()
parser.add_argument('vcf_1', help='1st Input VCF file {.vcf, .vcf.gz}')
parser.add_argument('vcf_2', help='2nd Input VCF file {.vcf, .vcf.gz}')
parser.add_argument('output_directory', help='Output directory')
args = parser.parse_args()

isec(args.vcf_1, args.vcf_2, args.output_directory)
