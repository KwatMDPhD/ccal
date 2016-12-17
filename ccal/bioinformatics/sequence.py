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

from . import CODON_TO_AMINO_ACID


def dna_to_rna(dna_sequence):
    """

    :param dna_sequence: str;
    :return: str;
    """

    return dna_sequence.replace('T', 'U')


def rna_to_dna(rna_sequence):
    """

    :param rna_sequence: str;
    :return: str;
    """

    return rna_sequence.replace('U', 'T')


def dna_to_reverse_complement(dna_sequence):
    complements = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N', '': ''}
    return ''.join([complements[x] for x in dna_sequence[::-1]])


def translate_nucleotides(nucleotide_seq, nucleotide_type='dna', reading_frame_offset=0, reading_frame_direction=1):
    """

    :param nucleotide_seq:
    :param nucleotide_type:
    :param reading_frame_offset:
    :param reading_frame_direction:
    :return:
    """

    assert reading_frame_offset in (0, 1, 2)
    assert reading_frame_direction in (-1, 1)

    if reading_frame_direction == -1:
        nucleotide_seq = dna_to_reverse_complement(nucleotide_seq)

    if nucleotide_type == 'dna':
        nucleotide_seq = dna_to_rna(nucleotide_seq)

    return [CODON_TO_AMINO_ACID[codon] for codon in split_codons(nucleotide_sequence=nucleotide_seq,
                                                                 reading_frame_offset=reading_frame_offset,
                                                                 reading_frame_direction=reading_frame_direction)]


def split_codons(nucleotide_sequence, reading_frame_offset=0, reading_frame_direction=1):
    """
    Return a list of 3-character strings representing codons extracted from a nucleotide sequence
    at a specified offset (0, 1, 2) and direction (-1 or 1).

    :param nucleotide_sequence:
    :param reading_frame_offset:
    :param reading_frame_direction:
    :return:
    """

    assert reading_frame_offset in (0, 1, 2)
    assert reading_frame_direction in (-1, 1)

    codons = []
    num_codons = int((len(nucleotide_sequence) - reading_frame_offset) / 3)
    for i in range(num_codons):
        codons.append(nucleotide_sequence[i * 3 + reading_frame_offset:(i + 1) * 3 + reading_frame_offset])

    return codons
