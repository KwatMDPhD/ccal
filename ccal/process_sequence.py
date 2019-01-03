CODON_TO_AMINO_ACID = {
    "GUC": "V",
    "ACC": "T",
    "GUA": "V",
    "GUG": "V",
    "GUU": "V",
    "AAC": "N",
    "CCU": "P",
    "UGG": "W",
    "AGC": "S",
    "AUC": "I",
    "CAU": "H",
    "AAU": "N",
    "AGU": "S",
    "ACU": "T",
    "CAC": "H",
    "ACG": "T",
    "CCG": "P",
    "CCA": "P",
    "ACA": "T",
    "CCC": "P",
    "GGU": "G",
    "UCU": "S",
    "GCG": "A",
    "UGC": "C",
    "CAG": "Q",
    "GAU": "D",
    "UAU": "Y",
    "CGG": "R",
    "UCG": "S",
    "AGG": "R",
    "GGG": "G",
    "UCC": "S",
    "UCA": "S",
    "GAG": "E",
    "GGA": "G",
    "UAC": "Y",
    "GAC": "D",
    "GAA": "E",
    "AUA": "I",
    "GCA": "A",
    "CUU": "L",
    "GGC": "G",
    "AUG": "M",
    "CUG": "L",
    "CUC": "L",
    "AGA": "R",
    "CUA": "L",
    "GCC": "A",
    "AAA": "K",
    "AAG": "K",
    "CAA": "Q",
    "UUU": "F",
    "CGU": "R",
    "CGA": "R",
    "GCU": "A",
    "UGU": "C",
    "AUU": "I",
    "UUG": "L",
    "UUA": "L",
    "CGC": "R",
    "UUC": "F",
    "UAA": "X",
    "UAG": "X",
    "UGA": "X",
}


def transcribe_dna_sequence(dna_sequence):

    return dna_sequence.replace("T", "U")


def reverse_transcribe_rna_sequence(rna_sequence):

    return rna_sequence.replace("U", "T")


def reverse_complement_dna_sequence(dna_sequence):

    dna_to_complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

    return "".join(dna_to_complement[dna] for dna in reversed(dna_sequence))


def translate_nucleotide_sequence(
    nucleotide_sequence,
    nucleotide_type,
    reading_frame_offset=0,
    reading_frame_direction=1,
):

    if nucleotide_type == "DNA":

        if reading_frame_direction == -1:

            nucleotide_sequence = reverse_complement_dna_sequence(nucleotide_sequence)

        nucleotide_sequence = transcribe_dna_sequence(nucleotide_sequence)

    return "".join(
        CODON_TO_AMINO_ACID[codon]
        for codon in split_codons(
            nucleotide_sequence, reading_frame_offset=reading_frame_offset
        )
    )


def split_codons(nucleotide_sequence, reading_frame_offset=0):

    codons = []

    for i in range(int((len(nucleotide_sequence) - reading_frame_offset) / 3)):

        codons.append(
            nucleotide_sequence[
                i * 3 + reading_frame_offset : (i + 1) * 3 + reading_frame_offset
            ]
        )

    return codons
