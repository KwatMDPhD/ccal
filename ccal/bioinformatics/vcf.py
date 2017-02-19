"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import re
from pprint import pprint

from pandas import read_csv
import tabix
from Bio import bgzf

from . import PATH_GRCH38, PATH_HG38, PATH_CHAIN_GRCH37_TO_GRCH38, PATH_CHAIN_HG19_TO_HG38, \
    PATH_CHROMOSOME_MAP, CHROMOSOMES, CHROMOSOMES_CHR, \
    PATH_DBSNP, PATH_CLINVAR, \
    PICARD, SNPEFF, SNPSIFT
from ..support.str_ import split_ignoring_inside_quotes, remove_nested_quotes
from ..support.file import bgzip_tabix, mark_filename
from ..support.system import run_cmd

CASTER = {
    'POS': int,
    'QUAL': float,
    'GT': lambda x: re.split('[|/]', x),
    'AD': lambda x: x.split(','),
    'VQSLOD': float,
    'CLNSIG': lambda x: max([int(s) for s in re.split('[,|]', x)]),
}


# ======================================================================================================================
# Work with VCF as DataFrame
# ======================================================================================================================
def read_vcf(filepath, verbose=False):
    """
    Read a VCF.
    :param filepath: str;
    :param verbose: bool;
    :return: dict;
    """

    vcf = {
        'meta_information': {'INFO': {},
                             'FILTER': {},
                             'FORMAT': {},
                             'reference': {}
                             },
        'header': [],
        'samples': [],
        'data': None
    }

    # Open VCF
    try:
        f = open(filepath)
        f.readline()
        f.seek(0)
        bgzipped = False
    except UnicodeDecodeError:
        f = bgzf.open(filepath)
        bgzipped = True

    for line in f:
        if bgzipped:
            line = line.decode()
        line = line.strip()

        if line.startswith('##'):  # Meta-information
            # Remove '##' prefix
            line = line[2:]

            # Find the 1st '='
            ei = line.find('=')

            # Get field name and field line
            fn, fl = line[:ei], line[ei + 1:]

            if fl.startswith('<') and fl.endswith('>'):
                # Strip '<' and '>'
                fl = fl[1:-1]

                # Split field line
                fl_split = split_ignoring_inside_quotes(fl, ',')

                # Get ID
                id_ = fl_split[0].split('=')[1]

                # Parse field line
                fd_v = {}
                for s in fl_split[1:]:
                    ei = s.find('=')
                    k, v = s[:ei], s[ei + 1:]
                    fd_v[k] = remove_nested_quotes(v)

                # Save
                if fn in vcf['meta_information']:
                    if id_ in vcf['meta_information'][fn]:
                        raise ValueError('Duplicated ID {}.'.format(id_))
                    else:
                        vcf['meta_information'][fn][id_] = fd_v
                else:
                    vcf['meta_information'][fn] = {id_: fd_v}
            else:
                print('Didn\'t parse: {}.'.format(fl))

        elif line.startswith('#CHROM'):  # Header
            # Remove '#' prefix
            line = line[1:]

            # Get header line number
            vcf['header'] = line.split('\t')
            vcf['samples'] = vcf['header'][9:]
        else:
            break

    # Close VCF
    f.close()

    # Read data
    vcf['data'] = read_csv(filepath, sep='\t', comment='#')

    if verbose:
        print('********* VCF dict (without data) *********')
        for k, v in vcf.items():
            if k != 'data':
                pprint({k: v}, compact=True, width=110)
        print('*******************************************')
    return vcf


# TODO: keep only 1 logic
def get_allelic_frequency(vcf_row):
    """

    :param vcf_row: Series; a VCF row
    :return: list; list of lists, which contain allelic frequencies for a sample
    """

    try:
        to_return = []
        for sample in vcf_row[9:]:
            sample_split = sample.split(':')

            dp = int(sample_split[2])
            to_return.append([round(ad / dp, 3) for ad in [int(i) for i in sample_split[1].split(',')]])
    except ValueError:
        pass


# ======================================================================================================================
# Work with VCF as file
# ======================================================================================================================
def get_variants_by_tabix(contig, start, end, sample_vcf, reference_vcf=None):
    """

    :param contig: str;
    :param start: int;
    :param end: int;
    :param sample_vcf: str or pytabix handler;
    :param reference_vcf: str or pytabix handler;
    :return: list; list of dict
    """

    if isinstance(sample_vcf, str):  # Open sample VCF
        sample_vcf = tabix.open(sample_vcf)

    records = sample_vcf.query(contig, start - 1, end)

    if reference_vcf and len(list(records)) == 0:  # If sample does not have the record, query reference if given

        if isinstance(reference_vcf, str):  # Open reference VCF
            reference_vcf = tabix.open(reference_vcf)

        records = reference_vcf.query(contig, start - 1, end)

    return [parse_variant(r) for r in records]


def parse_variant(vcf_row, verbose=False):
    """

    :param vcf_row:
    :return: dict
    """

    variant = {
        'CHROM': _cast('CHROM', vcf_row[0]),
        'POS': _cast('POS', vcf_row[1]),
        'REF': _cast('REF', vcf_row[3]),
        'FILTER': _cast('FILTER', vcf_row[6]),
    }

    # ID
    rsid = vcf_row[2]
    if rsid and rsid != '.':
        variant['ID'] = _cast('ID', rsid)

    # ALT
    alt = vcf_row[4]
    if alt and alt != '.':
        variant['ALT'] = _cast('ALT', alt)

    # QUAL
    qual = vcf_row[5]
    if qual and qual != '.':
        variant['QUAL'] = _cast('QUAL', qual)

    # Samples
    format_ = vcf_row[8].split(':')
    for i, s in enumerate(vcf_row[9:]):
        s_d = {}

        # Sample
        for k, v in zip(format_, s.split(':')):
            s_d[k] = _cast(k, v)

        # Genotype
        if 'ALT' in variant:
            ref_alts = [variant['REF']] + variant['ALT'].split(',')
            s_d['genotype'] = [ref_alts[int(gt)] for gt in s_d['GT']]
        else:
            s_d['genotype'] = [variant['REF']] * 2

        # Allelic frequency
        s_d['allelic_frequency'] = [round(int(ad) / int(s_d['DP']), 3) for ad in s_d['AD']]

        variant['sample_{}'.format(i + 1)] = s_d

    info_split = vcf_row[7].split(';')
    for i_s in info_split:
        if i_s.startswith('ANN='):
            anns = {}
            for i, a in enumerate(i_s.split(',')):
                a_split = a.split('|')

                anns[i] = {
                    'effect': a_split[1],
                    'putative_impact': a_split[2],
                    'gene_name': a_split[3],
                    'gene_id': a_split[4],
                    'feature_type': a_split[5],
                    'feature_id': a_split[6],
                    'transcript_biotype': a_split[7],
                    'rank': a_split[8],
                    'hgvsc': a_split[9],
                    'hgvsp': a_split[10],
                    'cdna_position': a_split[11],
                    'cds_position': a_split[12],
                    'protein_position': a_split[13],
                    'distance_to_feature': a_split[14],
                }
            variant['ANN'] = anns
        else:
            try:
                k, v = i_s.split('=')
                if v and v != '.':
                    # TODO: decode properly
                    variant[k] = _cast(k, v)
            except ValueError:
                print('INFO error: {} (not key=value)'.format(i_s))

    if verbose:
        print('********* Variant *********')
        pprint(variant, compact=True, width=110)
        print('***************************')
    return variant


def _cast(k, v, caster=CASTER):
    """

    :param k: str;
    :param v: str;
    :param caster: dict;
    :return:
    """

    if k in caster:
        return caster[k](v)
    else:
        return v


# ======================================================================================================================
# Operate
# ======================================================================================================================
def concat_snp_indel(snp_filepath, indel_filepath, output_fname):
    """
    Concatenate SNP and InDel using BCFTools concat and sort.
    :param snp_filepath: str;
    :param indel_filepath: str;
    :param output_fname: str;
    :return: str;
    """

    return concat_sort([snp_filepath, indel_filepath], output_fname)


def concat_sort(fnames, output_fname):
    """
    Concatenate VCFs <fnames> using BCFTools concat and sort.
    :param fnames:
    :param output_fname: str;
    :return: str;
    """

    output_fname = mark_filename(output_fname, 'concat_sort', '.vcf')
    fnames = [bgzip_tabix(fn) for fn in fnames]

    cmd = 'bcftools concat -a ' + ' '.join(fnames) + ' > {}'.format(output_fname)

    run_cmd(cmd)

    return bgzip_tabix(output_fname)


def isec(fname1, fname2, output_directory):
    """

    :param fname1:
    :param fname2:
    :param output_directory:
    :return: None
    """

    fname1 = bgzip_tabix(fname1)
    fname2 = bgzip_tabix(fname2)

    cmd = 'bcftools isec -O z -p {} {} {}'.format(output_directory, fname1, fname2)

    run_cmd(cmd)


def remap_38(fname, source_assembly):
    """
    Re-map genomic coordinates of <fname> based on GRCh38 using picard LiftoverVcf.
    :param fname: str;
    :param source_assembly: str; {hg19, grch37}
    :return: str;
    """

    if source_assembly == 'hg19':
        chain = PATH_CHAIN_HG19_TO_HG38
        path_target_assembly = PATH_HG38
    elif source_assembly == 'grch37':
        chain = PATH_CHAIN_GRCH37_TO_GRCH38
        path_target_assembly = PATH_GRCH38
    else:
        raise ValueError('Unknown assembly_from {}.'.format(source_assembly))

    mark = '{}grch38'.format(source_assembly)

    output_fname = mark_filename(fname, mark, '.vcf')
    reject_fname = mark_filename(fname, '{}_rejected'.format(mark), '.vcf')

    cmd = '{} LiftoverVcf INPUT={} OUTPUT={} REJECT={} CHAIN={} REFERENCE_SEQUENCE={}'.format(PICARD,
                                                                                              fname,
                                                                                              output_fname,
                                                                                              reject_fname,
                                                                                              chain,
                                                                                              path_target_assembly)

    run_cmd(cmd)

    bgzip_tabix(reject_fname)
    return bgzip_tabix(output_fname)


def rename_chr_sort(fname):
    """
    Rename chromosomes.
    :param fname:
    :return: str;
    """

    output_fname = mark_filename(fname, 'rename_chr_sort', '.vcf')
    fname = bgzip_tabix(fname)

    cmd = 'bcftools annotate --rename-chrs {} {} -o {}'.format(PATH_CHROMOSOME_MAP, fname, output_fname)

    run_cmd(cmd)

    return bgzip_tabix(output_fname)


def extract_chr(fname, chromosome_format):
    """
    Extract chromosome 1 to 22, X, Y, and MT from <fname>, and bgzip and tabix the extracted VCF.
    :param fname:
    :param chromosome_format:
    :return: str;
    """

    output_fname = mark_filename(fname, 'extract_chr', '.vcf')
    fname = bgzip_tabix(fname)

    cmd_template = 'bcftools view -r {} {} -o {}'
    if chromosome_format == 'chr#':
        cmd = cmd_template.format(','.join(CHROMOSOMES_CHR), fname, output_fname)
    elif chromosome_format == '#':
        cmd = cmd_template.format(','.join(CHROMOSOMES), fname, output_fname)
    else:
        raise ValueError('Chromosome format {} not found in (chr#, #)'.format(chromosome_format))

    run_cmd(cmd)

    return bgzip_tabix(output_fname)


def snpeff(fname, genomic_assembly):
    """
    Annotate VCF <fname> using SNPEff.
    :param fname:
    :param genomic_assembly: str;
    :return: str;
    """

    if genomic_assembly == 'grch37':
        genomic_assembly = 'GRCh37.75'
    elif genomic_assembly == 'grch38':
        genomic_assembly = 'GRCh38.82'
    else:
        raise ValueError('Unknown genomic_assembly {}; choose from (grch37, grch38).'.format(genomic_assembly))

    output_fname = mark_filename(fname, 'snpeff', '.vcf')

    fname = bgzip_tabix(fname)

    cmd = '{} -noDownload -v -noLog -s {}.html {} {} > {}'.format(SNPEFF,
                                                                  output_fname[:-len('.vcf')],
                                                                  genomic_assembly,
                                                                  fname,
                                                                  output_fname)

    run_cmd(cmd)

    return bgzip_tabix(output_fname)


def snpsift(fname, annotation):
    """
    Annotate VCF <fname> using SNPSift.
    :param fname:
    :param annotation: str; {dpsnp, clinvar}
    :return: str;
    """

    if annotation == 'dbsnp':
        mark = 'dbsnp'
        path_annotation = PATH_DBSNP
        flag = '-noInfo'

    elif annotation == 'clinvar':
        mark = 'clinvar'
        path_annotation = PATH_CLINVAR
        flag = ''

    else:
        raise ValueError('annotation has to be one of {dbsnp, clinvar}.')

    output_fname = mark_filename(fname, mark, '.vcf')

    fname = bgzip_tabix(fname)

    cmd = '{} annotate -noDownload -v -noLog {} {} {} > {}'.format(SNPSIFT, flag, path_annotation, fname, output_fname)

    run_cmd(cmd)

    return bgzip_tabix(output_fname)


def annotate(fname, genomic_assembly, pipeline):
    """
    Annotate <fname> VCF with SNPEff and ClinVar (SNPSift).
    :param fname: str;
    :param genomic_assembly: str;
    :param pipeline: str;
    :return: str;
    """

    if pipeline == 'snpeff':
        print('\n**************************************************************************')
        output_fname = snpeff(fname, genomic_assembly)

    elif pipeline == 'snpeff-clinvar':
        print('\n**************************************************************************')
        output_fname = snpeff(fname, genomic_assembly)
        print('\n**************************************************************************')
        output_fname = snpsift(output_fname, 'clinvar')

    elif pipeline == 'dbsnp-snpeff-clinvar':
        print('\n**************************************************************************')
        output_fname = snpsift(fname, 'dbsnp')
        print('\n**************************************************************************')
        output_fname = snpeff(output_fname, genomic_assembly)
        print('\n**************************************************************************')
        output_fname = snpsift(output_fname, 'clinvar')

    else:
        raise ValueError(
            'Unknown pipeline {}; choose from (snpeff, snpeff-clinvar, dbsnp-snpeff-clinvar).'.format(pipeline))

    return bgzip_tabix(output_fname)
