from os import environ
from os.path import join, isfile, isdir

from ..support.log import print_log

# ======================================================================================================================
# Parameters
# ======================================================================================================================
PATH_HOME = environ['HOME']

PATH_TOOLS = join(PATH_HOME, 'tools')
PATH_DATA = join(PATH_HOME, 'data')

# Java
PICARD = 'java -Xmx{}g -jar {}'.format(12, join(PATH_TOOLS, 'picard.jar'))
SNPEFF = 'java -Xmx{}g -jar {}'.format(12, join(PATH_TOOLS, 'snpEff', 'snpEff.jar'))
SNPSIFT = 'java -Xmx{}g -jar {}'.format(12, join(PATH_TOOLS, 'snpEff', 'SnpSift.jar'))

# Reference genome assemblies
PATH_GRCH38 = join(PATH_DATA, 'grch', '38', 'sequence', 'primary_assembly',
                   'Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz')
PATH_HG38 = join(PATH_DATA, 'grch', '38', 'sequence', 'hg', 'hg38.unmasked.fa.gz')

PATH_GENOME = PATH_GRCH38

# Genome assembly chains
PATH_CHAIN_GRCH37_TO_GRCH38 = join(PATH_DATA, 'grch', 'genomic_assembly_chain', 'GRCh37_to_GRCh38.chain.gz')
PATH_CHAIN_HG19_TO_HG38 = join(PATH_DATA, 'grch', 'genomic_assembly_chain', 'hg19ToHg38.over.chain.gz')

# DBSNP
PATH_DBSNP = join(PATH_DATA, 'grch', '38', 'variant', '00-All.vcf.gz')

# ClinVar
PATH_CLINVAR = join(PATH_DATA, 'grch', '38', 'variant', 'clinvar.vcf.gz')

# Chromosomes
PATH_CHROMOSOME_MAP = join(PATH_DATA, 'grch', 'grch_chromosome_map.txt')
CHROMOSOMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', 'M', 'MT', 'X', 'Y']
CHROMOSOMES_CHR = ['chr{}'.format(c) for c in CHROMOSOMES]

for p in [PATH_HOME, PATH_TOOLS, PATH_DATA, PATH_HG38, PATH_GRCH38, PATH_CHAIN_HG19_TO_HG38,
          PATH_CHAIN_GRCH37_TO_GRCH38, PATH_CLINVAR, PATH_DBSNP, PATH_CHROMOSOME_MAP]:
    if not (isdir(p) or isfile(p)):
        print_log('Warning: file {} doesn\'t exists.'.format(p))

CODON_TO_AMINO_ACID = {'GUC': 'V', 'ACC': 'T', 'GUA': 'V', 'GUG': 'V', 'GUU': 'V', 'AAC': 'N', 'CCU': 'P', 'UGG': 'W',
                       'AGC': 'S', 'AUC': 'I', 'CAU': 'H', 'AAU': 'N', 'AGU': 'S', 'ACU': 'T', 'CAC': 'H', 'ACG': 'T',
                       'CCG': 'P', 'CCA': 'P', 'ACA': 'T', 'CCC': 'P', 'GGU': 'G', 'UCU': 'S', 'GCG': 'A', 'UGC': 'C',
                       'CAG': 'Q', 'GAU': 'D', 'UAU': 'Y', 'CGG': 'R', 'UCG': 'S', 'AGG': 'R', 'GGG': 'G', 'UCC': 'S',
                       'UCA': 'S', 'GAG': 'E', 'GGA': 'G', 'UAC': 'Y', 'GAC': 'D', 'GAA': 'E', 'AUA': 'I', 'GCA': 'A',
                       'CUU': 'L', 'GGC': 'G', 'AUG': 'M', 'CUG': 'L', 'CUC': 'L', 'AGA': 'R', 'CUA': 'L', 'GCC': 'A',
                       'AAA': 'K', 'AAG': 'K', 'CAA': 'Q', 'UUU': 'F', 'CGU': 'R', 'CGA': 'R', 'GCU': 'A', 'UGU': 'C',
                       'AUU': 'I', 'UUG': 'L', 'UUA': 'L', 'CGC': 'R', 'UUC': 'F', 'UAA': 'X', 'UAG': 'X', 'UGA': 'X'}
