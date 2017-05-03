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

import copy
import re
from pprint import pprint

import pyfaidx
from pandas import DataFrame, read_csv


def get_apobec_mutational_signature_enrichment(mutation_filepath,
                                               fasta_filepath):
    """
    """

    # If only 1 filepath is passed, put it in a list
    if isinstance(mutation_filepath, str):
        mutation_filepath = [mutation_filepath]

    # Get file type
    for e in ['vcf', 'vcf.gz', 'maf']:
        if mutation_filepath[0].endswith(e):
            filetype = e

    # Load reference genome
    fasta = pyfaidx.Fasta(
        fasta_filepath,
        filt_function=lambda c: '_' not in c,  # Load only 1-22, X, Y, and M
        sequence_always_upper=True)
    print('Loaded reference genome: {}.'.format(list(fasta.keys())))

    span = 20

    # Set up mutational signature
    ss = [
    'tCa ==> tGa',
    'tCa ==> tTa',
    'tCt ==> tGt',
    'tCt ==> tTt',
    'tGa ==> tCa',
    'tGa ==> tAa',
    'aGa ==> aCa',
    'aGa ==> aAa',
    ]

    # Identigy what to count
    signature_mutations,\
        control_mutations,\
        signature_b_motifs,\
        control_b_motifs = _identify_what_to_count(ss)

    # Count
    samples = {}
    for i, fp in enumerate(mutation_filepath):

        # Get sample ID
        id_ = fp.split('/')[-1].split('.')[0]
        if id_ in samples:
            raise ValueError('{} duplicated'.format(id_))
        print('({}) {} ...'.format(i, id_))

        # Count
        samples[id_] = count(fp, filetype, fasta, span, signature_mutations,
                             control_mutations, signature_b_motifs,
                             control_b_motifs)

    # Tabulate results
    df = DataFrame(samples)
    df.ix['APOBEC Mutational Signature Enrichment'] = (
        df.ix[list(signature_mutations.keys())].sum() /
        df.ix[list(control_mutations.keys())].sum()) / (
            df.ix[list(signature_b_motifs.keys())].sum() /
            df.ix[list(control_b_motifs.keys())].sum())

    return df


def _identify_what_to_count(signature_mutations):
    """
    """

    # Signature mutations
    s_mutations = {}
    for m in signature_mutations:

        # Get before & after motifs, which must be the same length
        m_split = m.split('==>')
        b_m, a_m = [m_.strip() for m_ in m_split]
        if len(b_m) != len(a_m):
            raise ValueError(
                'Before ({}) & after ({}) motifs differ in length.'.format(
                    b_m, a_m))

        s_mutations[m] = {
            'before': b_m.upper(),  # Before motif
            'after': a_m.upper(),  # After motif
            'n': 0,  # Mutation count
            'change_start_i': min([
                m.start() for m in re.finditer('[A-Z]+', b_m)
            ]),  # Chaning-motif start index
            'change_end_i': max([m.end() for m in re.finditer('[A-Z]+', b_m)
                                 ]),  # Changing-motif end index
        }
    print('s_mutations:')
    pprint(s_mutations)

    # Control mutations
    c_mutations = {}
    for d in s_mutations.values():

        # Get before & after motifs
        b_m = d.get('before')
        a_m = d.get('after')

        # Get changing-before & -after motifs
        c_s_i, c_e_i = d.get('change_start_i'), d.get('change_end_i')
        c_b_m = b_m[c_s_i:c_e_i]
        c_a_m = a_m[c_s_i:c_e_i]

        c_mutations['{} ==> {}'.format(c_b_m, c_a_m)] = {
            'before': c_b_m,
            'after': c_a_m,
            'n': 0
        }
    print('\nc_mutations:')
    pprint(c_mutations)

    # Signature before-motifs
    s_b_motifs = {d.get('before').lower(): 0 for m, d in s_mutations.items()}
    print('\ns_b_motifs:')
    pprint(s_b_motifs)

    # Control before-motifs
    c_b_motifs = {d.get('before').lower(): 0 for m, d in c_mutations.items()}
    print('\nc_b_motifs:')
    pprint(c_b_motifs)
    print()

    return s_mutations, c_mutations, s_b_motifs, c_b_motifs


def count(filepath, filetype, fasta, span, signature_mutations,
          control_mutations, signature_b_motifs, control_b_motifs):
    """
    """

    # Load mutation file
    if filetype in ('vcf', 'vcf.gz'):
        df = read_csv(
            filepath, sep='\t', comment='#',
            encoding='ISO-8859-1').iloc[:, [0, 1, 3, 4]]
    elif filetype == 'maf':
        df = read_csv(
            filepath, sep='\t', comment='#',
            encoding='ISO-8859-1').iloc[:, [4, 5, 10, 12]]

    # Identify what to count
    s_mutations = copy.deepcopy(signature_mutations)
    c_mutations = copy.deepcopy(control_mutations)
    s_b_motifs = copy.deepcopy(signature_b_motifs)
    c_b_motifs = copy.deepcopy(control_b_motifs)

    # Evaluate each row
    n_spanning_bases = 0
    n_mutations = 0
    for i, (chr_, pos, ref, alt) in df.iterrows():
        chr_ = 'chr{}'.format(chr_)

        pos = int(pos) - 1

        # Skip if there is no reference information
        if chr_ not in fasta.keys():
            continue

        # Skip if variant is not a SNP
        if not (1 == len(ref) == len(alt)) or ref == '-' or alt == '-':
            continue

        assert ref == fasta[chr_][pos].seq, '{} in {}'.format(
            ref, fasta[chr_][pos - 1:pos + 2].seq)

        n_mutations += 1

        # Check if this mutation matches any signature mutation
        for m, d in s_mutations.items():

            # Get before & after signature motifs
            b_m = d.get('before')
            a_m = d.get('after')

            # Get changing-before & -after motifs
            c_s_i, c_e_i = d.get('change_start_i'), d.get('change_end_i')
            c_b_m = b_m[c_s_i:c_e_i]
            c_a_m = a_m[c_s_i:c_e_i]

            # Check if the chaning-before motif matches the ref
            if c_b_m == ref:

                # Check if the surrounding sequences are the same
                if b_m == fasta[chr_][pos - c_s_i:pos + len(b_m) - c_e_i +
                                      1].seq:

                    # Check if the changing-after motif matches the alt
                    if c_a_m == alt:

                        # Matched a signature mutation, so increment count
                        d['n'] += 1

        # Check if this mutation matches any control mutation
        for m, d in c_mutations.items():

            # Get before & after control motifs
            b_m = d.get('before')
            a_m = d.get('after')

            # Check if the before motif matches the ref
            if b_m == ref:

                # Check if the after motif matches the alt
                if a_m == alt:

                    # Matched a control mutation, so increment count
                    d['n'] += 1

        # Get mutation-spanning sequences
        span_seq = fasta[chr_][pos - span:pos + span + 1].seq
        if re.findall('[^ACGT]', span_seq):
            print(span_seq)

        n_spanning_bases += len(span_seq)

        # Count signature's changing-before motifs in the spanning sequences
        for m in s_b_motifs:
            s_b_motifs[m] += span_seq.count(m.upper())
        # Count control's changing-before motifs in the spanning sequences
        for m in c_b_motifs:
            c_b_motifs[m] += span_seq.count(m.upper())

    counts = {
        'N Mutations': i + 1,
        'N Mutations Analyzed': n_mutations,
        'N Spanning Bases': n_spanning_bases,
    }
    counts.update({m: d['n'] for m, d in s_mutations.items()})
    counts.update({m: d['n'] for m, d in c_mutations.items()})
    counts.update(s_b_motifs)
    counts.update(c_b_motifs)

    return counts
