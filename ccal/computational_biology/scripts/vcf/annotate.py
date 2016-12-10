import argparse

from ...vcf import annotate

parser = argparse.ArgumentParser()
parser.add_argument('input_fname', help='Input VCF file {.vcf, .vcf.gz}')
parser.add_argument('genomic_assembly', help='Genomic assembly {grch37, grch38}.')
parser.add_argument('pipeline', help='Annotation pipeline {snpeff, snpeff-clinvar, dbsnp-snpeff-clinvar}')
args = parser.parse_args()

annotate(args.input_fname, args.genomic_assembly, args.pipeline)
