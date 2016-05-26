"""
Cancer Computational Biology Analysis Supporting Library v0.1

Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

James Jensen
Email
Affiliation
"""


import os
import numpy as np
import pandas as pd


def read_gct(filename, fill_na=0):
    """
    Read <filename> (.gct) and convert it into a matrix.
    """
    data = pd.read_csv(filename, skiprows=2, sep='\t')
    if fill_na != None:
        data.fillna(fill_na, inplace=True)
    column1, column2 = data.columns[:2]
    assert column1 == 'Name', 'Column 1 != "Name"'
    assert column2 == 'Description', 'Column 2 != "Description"'

    #
    data.set_index('Name', inplace=True)
    data.index.name = None

    #
    description = data['Description']
    data.drop('Description', axis=1, inplace=True)

    return data


def write_gct(matrix, filename, description=None):
    """
    Write a matrix to <filename> (.gct).
    """
    # Set output filename
    if not filename.endswith('.gct'):
        filename += '.gct'

    matrix.index.name = 'Name'

    if description:
        assert len(description) == matrix.size[1]
    else:
        description = matrix.index
    matrix.insert(0, 'Description', description)

    with open(filename, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(*matrix.shape))
        matrix.to_csv(f, sep='\t')