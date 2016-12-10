import argparse

from ...vcf import rename_chr_sort

parser = argparse.ArgumentParser()
parser.add_argument('input_fname', help='Input file')
args = parser.parse_args()

rename_chr_sort(args.input_fname)
