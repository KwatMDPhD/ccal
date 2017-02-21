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

from numpy import argmin
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

VCF_FIELD_CASTER = {
    'POS': int,
    'QUAL': float,
    'GT': lambda x: re.split('[|/]', x),
    'AD': lambda x: x.split(','),
    'VQSLOD': float,
    'CLNSIG': lambda x: max([int(s) for s in re.split('[,|]', x)]),
}

ANN_FIELDS = ['ALT', 'effect', 'impact', 'gene_name', 'gene_id', 'feature_type', 'feature_id', 'transcript_biotype',
              'rank', 'hgvsc', 'hgvsp', 'cdna_position', 'cds_position', 'protein_position', 'distance_to_feature',
              'error']

# Prioritize Sequence Ontology terms in order of severity, as estimated by Ensembl and others:
# http://useast.ensembl.org/info/genome/variation/predicted_data.html#consequences
# http://snpeff.sourceforge.net/VCFannotationformat_v1.0.pdf
# vcf2maf.pl
ANN_EFFECT_RANKING = [
    # Loss of transcript or exon
    'transcript_ablation',
    'exon_loss_variant',
    # Altered splicing 1
    'splice_acceptor_variant',
    'splice_donor_variant',
    # Nonsense mutation
    'stop_gained',
    # Frameshift
    'frameshift_variant',
    # Nonstop mutation 1
    'stop_lost',
    # Nonstart mutation
    'start_lost',
    'initiator_codon_variant',
    # Altered transcript
    'transcript_amplification',
    'transcript_variant',
    # InDel
    'disruptive_inframe_insertion',
    'disruptive_inframe_deletion',
    'inframe_insertion',
    'inframe_deletion',
    # Missense mutation
    'conservative_missense_variant',
    'rare_amino_acid_variant',
    'missense_variant',
    'protein_altering_variant',
    # Altered splicing 2
    'splice_region_variant',
    # Nonstop mutation 2
    'incomplete_terminal_codon_variant',
    # Silent mutation
    'start_retained_variant',
    'stop_retained_variant',
    'synonymous_variant',
    # Mutation
    'coding_sequence_variant',
    'exon_variant',
    # Altered miRNA
    'mature_miRNA_variant',
    # Altered 5'UTR
    '5_prime_UTR_variant',
    '5_prime_UTR_premature_start_codon_gain_variant',
    # Altered 3'UTR
    '3_prime_UTR_variant',
    # Altered non-coding exon region
    'non_coding_exon_variant',
    'non_coding_transcript_exon_variant',
    # Altered intragenic region
    'intragenic_variant',
    'conserved_intron_variant',
    'intron_variant',
    'INTRAGENIC',
    # Altered nonsense-mediated-decay-target region
    'NMD_transcript_variant',
    # Altered non-coding region
    'non_coding_transcript_variant',
    'nc_transcript_variant',
    # Altered 5'flank site
    'upstream_gene_variant',
    # Altered 3'flank site
    'downstream_gene_variant',
    # Altered transcription-factor-binding region
    'TF_binsing_site_ablation',
    'TFBS_ablation',
    'TF_binding_site_amplification',
    'TFBS_amplification',
    'TF_binding_site_variant',
    'TFBS_variant',
    # Altered regulatory region
    'regulatory_region_ablation',
    'regulatory_region_amplification',
    'regulatory_region_variant',
    'regulatory_region',
    'feature_elongation',
    'feature_truncation',
    # Altered intergenic region
    'conserved_intergenic_variant',
    'intergenic_variant',
    'intergenic_region',
]


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

    for row in f:

        if bgzipped:
            row = row.decode()
        row = row.strip()

        if row.startswith('##'):  # Meta-information
            # Remove '##' prefix
            row = row[2:]

            # Find the 1st '='
            ei = row.find('=')

            # Get field name and field line
            fn, fl = row[:ei], row[ei + 1:]

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

        elif row.startswith('#CHROM'):  # Header
            # Remove '#' prefix
            row = row[1:]

            # Get header line number
            vcf['header'] = row.split('\t')
            vcf['samples'] = vcf['header'][9:]
        else:
            break

    # Close VCF
    f.close()

    # Read data
    vcf['data'] = read_csv(filepath, sep='\t', comment='#', header=None, names=vcf['header'])

    if verbose:
        print('********* VCF dict (without data) *********')
        for k, v in vcf.items():
            if k != 'data':
                pprint({k: v}, compact=True, width=110)
        print('*******************************************')
    return vcf


def get_start_end_positions(vcf_data):
    """

    :param vcf_data: DataFrame;
    :return: list; list of lists which contain
    """

    def f(vcf_row):
        pos, ref, alt = vcf_row.iloc[[1, 3, 4]]
        return _get_variant_start_end_positions(pos, ref, alt)

    s = vcf_data.apply(f, axis=1)
    s.name = 'start_end_positions'
    return s


def get_variant_type(vcf_data):
    """

    :param vcf_data: DataFrame;
    :return: list; list of lists which contain
    """

    def f(vcf_row):
        ref, alt = vcf_row.iloc[[3, 4]]
        return _get_variant_type(ref, alt)

    s = vcf_data.apply(f, axis=1)
    s.name = 'variant_type'
    return s


def get_allelic_frequencies(vcf_data, sample_iloc=9):
    """

    :param vcf_data: DataFrame;
    :param sample_iloc: int;
    :return: list; list of lists which contain allelic frequencies for a sample
    """

    def f(vcf_row):
        s_split = vcf_row.iloc[sample_iloc].split(':')
        try:
            dp = int(s_split[2])
            return tuple(['{:0.2f}'.format(ad / dp) for ad in [int(i) for i in s_split[1].split(',')]])
        except ValueError:
            return None
        except ZeroDivisionError:
            return None

    s = vcf_data.apply(f, axis=1)
    s.name = 'allelic_frequency'
    return s


def get_ann(vcf_data, field, n_ann=1):
    """

    :param vcf_data: DataFrame;
    :param field: str;
    :param n_ann: int;
    :return: list; list of lists which contain
    """

    i = ANN_FIELDS.index(field)

    def f(vcf_row):
        for i_s in vcf_row.iloc[7].split(';'):  # For each INFO

            if i_s.startswith('ANN='):  # ANN
                i_s = i_s[len('ANN='):]

                to_return = []
                anns = i_s.split(',')

                for a in anns[:min(len(anns), n_ann)]:  # For each ANN
                    to_return.append(a.split('|')[i])

                if len(to_return) == 1:
                    return to_return[0]
                else:
                    return to_return

    s = vcf_data.apply(f, axis=1)
    s.name = field
    return s


def get_maf_variant_classification(vcf_data, n_ann=1):
    """

    :param vcf_data: DataFrame;
    :param n_ann: int;
    :return: list; list of lists which contain
    """

    def f(vcf_row):
        for i_s in vcf_row.iloc[7].split(';'):  # For each INFO

            if i_s.startswith('ANN='):  # ANN
                i_s = i_s[len('ANN='):]

                anns = i_s.split(',')

                for a in anns[:min(len(anns), n_ann)]:  # For each ANN
                    effect = a.split('|')[1]
                    return _get_maf_variant_classification(effect)

    s = vcf_data.apply(f, axis=1)
    s.name = 'effect'
    return s


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


# ======================================================================================================================
# Parse
# ======================================================================================================================
def parse_variant(vcf_row, n_anns=1, verbose=False):
    """

    :param vcf_row: iterable;
    :param n_anns: int;
    :param verbose: bool;
    :return: dict;
    """

    variant = {
        'CHROM': _cast('CHROM', vcf_row[0]),
        'POS': _cast('POS', vcf_row[1]),
        'REF': _cast('REF', vcf_row[3]),
        'FILTER': _cast('FILTER', vcf_row[6]),
    }
    ref = variant['REF']

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

    # Variant type
    if alt:
        vt = _get_variant_type(ref, alt)
        variant['variant_type'] = vt

    # Samples
    variant['samples'] = []
    format_ = vcf_row[8].split(':')
    for i, s in enumerate(vcf_row[9:]):
        s_d = {'sample_id': i + 1}

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
        if 'DP' in s_d and int(s_d['DP']):
            s_d['allelic_frequency'] = [round(int(ad) / int(s_d['DP']), 3) for ad in s_d['AD']]

        variant['samples'].append(s_d)

    info_split = vcf_row[7].split(';')
    for i_s in info_split:
        if i_s.startswith('ANN='):
            anns = {}
            for i, a in enumerate(i_s.split(',')[:n_anns]):
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
                pass
                # print('INFO error: {} (not key=value)'.format(i_s))

    if verbose:
        print('********* Variant *********')
        pprint(variant, compact=True, width=110)
        print('***************************')
    return variant


def _cast(k, v, caster=VCF_FIELD_CASTER):
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


def _get_variant_type(ref, alt):
    """

    :param ref: str;
    :param alt: str;
    :return: str;
    """

    if len(ref) == len(alt):
        if len(ref) == 1:
            vt = 'SNP'
        elif len(ref) == 2:
            vt = 'DNP'
        elif len(ref) == 3:
            vt = 'TNP'
        else:  # 4 <= len(ref)
            vt = 'ONP'

    elif len(ref) < len(alt):
        vt = 'INS'

    else:  # len(alt) < len(ref)
        vt = 'DEL'

    return vt


def _get_variant_start_end_positions(pos, ref, alt):
    """

    :param ref: str;
    :param alt: str;
    :return: (str, str);
    """

    if len(ref) == len(alt):
        s, e = pos, pos + len(alt) - 1

    elif len(ref) < len(alt):
        s, e = pos, pos + 1

    else:  # len(alt) < len(ref)
        s, e = pos + 1, pos + len(ref) - len(alt)

    return s, e


def _get_maf_variant_classification(effect):
    """

    :param e: str; effect or effects concatenated by '&'
    :return: str; MAF variant classification
    """

    es = effect.split('&')
    vc = _convert_ann_effect_to_maf_variant_classificaton(es[argmin([ANN_EFFECT_RANKING.index(e) for e in es])])
    return vc


def _convert_ann_effect_to_maf_variant_classificaton(e):
    """

    :param e: str; effect
    :return: str; MAF variant classification
    """

    if e in ('transcript_ablation',
             'exon_loss_variant',
             'splice_acceptor_variant',
             'splice_donor_variant',
             'splice_region_variant'):
        vc = 'Splice_Site'

    elif e in ('stop_gained'):
        vc = 'Nonsense_Mutation'

    elif e in ('frameshift_variant OR INS'):
        vc = 'Frame_Shift_Ins'

    elif e in ('frameshift_variant OR DEL'):
        vc = 'Frame_Shift_Del'

    elif e in ('stop_lost'):
        vc = 'Nonstop_Mutation'

    elif e in ('start_lost',
               'initiator_codon_variant'):
        vc = 'Translation_Start_Site'

    elif e in ('disruptive_inframe_insertion',
               'inframe_insertion'):
        vc = 'In_Frame_Ins'

    elif e in ('disruptive_inframe_deletion',
               'inframe_deletion'):
        vc = 'In_Frame_Del'

    elif e in ('transcript_variant',
               'conservative_missense_variant',
               'rare_amino_acid_variant',
               'missense_variant',
               'protein_altering_variant',
               'coding_sequence_variant'):
        vc = 'Missense_Mutation'

    elif e in ('transcript_amplification',
               'intragenic_variant',
               'conserved_intron_variant',
               'intron_variant',
               'INTRAGENIC',
               'NMD_transcript_variant',
               'TF_binding_site_ablation',
               'TFBS_ablation',
               'TF_binding_site_amplification',
               'TFBS_amplification',
               'TF_binding_site_variant',
               'TFBS_variant',
               'regulatory_region_ablation',
               'regulatory_region_amplification',
               'regulatory_region_variant',
               'regulatory_region'):
        vc = 'Intron'

    elif e in ('incomplete_terminal_codon_variant',
               'start_retained_variant',
               'stop_retained_variant',
               'synonymous_variant'):
        vc = 'Silent'

    elif e in ('exon_variant',
               'mature_miRNA_variant',
               'non_coding_exon_variant',
               'non_coding_transcript_exon_variant',
               'non_coding_transcript_variant',
               'nc_transcript_variant'):
        vc = 'RNA'

    elif e in ('5_prime_UTR_variant',
               '5_prime_UTR_premature_start_codon_gain_variant'):
        vc = '%\'UTR'

    elif e in ('3_prime_UTR_variant'):
        vc = '3\'UTR'

    elif e in ('feature_elongation',
               'feature_truncation',
               'conserved_intergenic_variant',
               'intergenic_variant',
               'intergenic_region'):
        vc = 'IGR'

    elif e in ('upstream_gene_variant'):
        vc = '5\'Flank'

    elif e in ('downstream_gene_variant'):
        vc = '3\'Flank'

    else:
        print('Unknown effect: {}.'.format(e))
        vc = 'Targeted_Region'
    return vc


# ======================================================================================================================
# Use executables
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
