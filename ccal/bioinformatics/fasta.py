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

from ..support.system import run_command


def get_sequence(filepath, chromosome, start, end):
    """
    Return genomic sequences from region specified by chromosome:start-stop in filepath.
    :param filepath: str;
    :param chromosome: int or str; chromosome
    :param start: int or str; start position
    :param end: int or str; end position; must be greater than or equal to start position
    :return: str; genomic sequences from region specified by chromosome:start-stop in filepath
    """

    if start >= end:
        raise ValueError('Starting genomic position must be greater than the ending genomic position.')

    cmd = 'samtools faidx {} {}:{}-{}'.format(filepath, chromosome, start, end)

    stdout = run_command(cmd).stdout

    s = ''.join(stdout.split('\n')[1:])

    return s
