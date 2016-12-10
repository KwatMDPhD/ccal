import re

import tabix
from pandas import read_csv

from . import PATH_GRCH38, PATH_HG38, PATH_CHAIN_GRCH37_TO_GRCH38, PATH_CHAIN_HG19_TO_HG38, \
    PATH_CHROMOSOME_MAP, CHROMOSOMES, CHROMOSOMES_CHR, \
    PATH_DBSNP, PATH_CLINVAR, \
    PICARD, SNPEFF, SNPSIFT
from ..support.system import run_cmd
from ..support.file import bgzip_tabix, mark_filename


# ======================================================================================================================
# Query
# ======================================================================================================================
def read_vcf(filepath, verbose=False):
    """

    :param filepath:
    :param verbose:
    :return:
    """

    meta_information = []
    header_line = None
    header = None

    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.startswith('##'):
                meta_information.append(line.strip())
            elif line.startswith('#CHROM'):
                header_line = i
                print('header starts at line {}\n'.format(header_line))
                header = line.strip()
            else:
                break
    vcf_data = read_csv(filepath, sep='\t', skiprows=header_line)

    if verbose:
        print('Meta-information\n{}\n'.format(meta_information))
        print('header\n{}\n'.format(header))
        print('VCF\n{}\n'.format(vcf_data.head()))
    return meta_information, header_line, header, vcf_data


def read_sample_and_reference_vcfs(sample_vcf, reference_vcf):
    """
    Read sample and reference (dbSNP) VCF.GZs, and return as pytabix handlers.
    :param sample_vcf:
    :param reference_vcf:
    :return:
    """

    return tabix.open(sample_vcf), tabix.open(reference_vcf)


def get_variant_by_tabix(contig, start_position, stop_position, sample_variants, reference_variants):
    """

    :param contig:
    :param start_position:
    :param stop_position:
    :param sample_variants: pytabix handler;
    :param reference_variants: pytabix handler;
    :return:
    """

    records = sample_variants.query(contig, start_position - 1, stop_position)
    sample_or_reference = 'sample'

    if len(list(records)) > 1:  # Have more than 1 record per variant ID
        raise ValueError(
            'More than 1 record found when querying variant at {}:()-()'.format(contig, start_position, stop_position))

    elif len(list(records)) == 0:  # If sample does not have the genotype, query reference
        records = reference_variants.query(contig, start_position - 1, stop_position)
        sample_or_reference = 'reference'

    # Parse query output
    for r in records:
        variant_result = parse_vcf_information(r, sample_or_reference)
        return variant_result


def get_gene_variants_by_tabix(contig, start_position, stop_position, sample_variants):
    """

    :param contig:
    :param start_position:
    :param stop_position:
    :param sample_variants:
    :return:
    """

    records = sample_variants.query(contig, start_position - 1, stop_position)
    sample_or_reference = 'sample'

    gene_results = {}

    for r in records:
        # Get only variant with variant ID
        if r[2] != '.':
            # Store variant information
            gene_results[r[2]] = parse_vcf_information(r, sample_or_reference)
    return gene_results


def parse_vcf_information(record, sample_or_reference):
    """

    :param record: tabix record;
    :param sample_or_reference: str;
    :return: dict;
    """

    contig, start_position, quality, information = record[0], record[1], record[5], record[7].split(';')

    result = {'contig': contig,
              'start_position': start_position,
              'nucleotide_pair': get_vcf_nucleotide_pair(record, sample_or_reference)}
    if quality != '.':
        result['quality'] = quality

    for info in information:
        try:
            key, value = info.split('=')
        except ValueError:
            pass
            # if verbose: print('Error parsing {} (not key=value entry in INFO)'.format(info))

        fields = {'VQSLOD': lambda score: float(score),
                  'ANN': lambda string: string,
                  'CLNSIG': lambda scores: max([int(s) for s in re.split('[,|]', scores)])}
        if key in fields:
            if key == 'ANN':
                for k, v in parse_vcf_annotation(value).items():
                    result[k] = v
            else:
                result[key] = fields[key](value)

    return result


def get_vcf_nucleotide_pair(record, sample_or_reference):
    """

    :param record: tabix record;
    :param sample_or_reference: {'sample', 'reference'}
    :return: str;
    """

    if sample_or_reference == 'sample':
        zygosity = get_vcf_zygosity(record)
        reference_nucleotide, alternate_nucleotides = record[3], record[4].split(',')
        return tuple([([reference_nucleotide] + alternate_nucleotides)[x] for x in zygosity])

    elif sample_or_reference == 'reference':
        reference_nucleotide = record[3]
        return tuple([reference_nucleotide] * 2)

    else:
        raise ValueError('Unknown sample_or_reference: {}'.format(sample_or_reference))


def get_vcf_zygosity(record):
    """

    :param record: tabix record
    :return:
    """
    zygosity = [int(code) for code in re.split('[|/]', record[9].split(':')[0])]
    return zygosity


def parse_vcf_annotation(annotations):
    """

    :param annotations: annotations of ANN=annotation INFO entry in a VCF
    :return: dict
    """

    first_annotation = annotations.split(',')[0].split('|')
    return {'effect': first_annotation[1], 'impact': first_annotation[2]}


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
