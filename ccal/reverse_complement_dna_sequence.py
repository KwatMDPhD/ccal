from .DNA_COMPLEMENT import DNA_COMPLEMENT


def reverse_complement_dna_sequence(dna_sequence):

    return "".join(DNA_COMPLEMENT[dna] for dna in reversed(dna_sequence))
