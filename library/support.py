"""
Cancer Computational Biology Analysis Supporting Library v0.1

Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center
"""

import pandas as pd


# ======================================================================================================================
# File operations
# ======================================================================================================================
def read_gct(filename, fill_na=0, verbose=False):
    """
    Read `filename` (.gct) and convert it into a DataFrame.
    """
    dataframe = pd.read_csv(filename, skiprows=2, sep='\t')
    if fill_na != None:
        dataframe.fillna(fill_na, inplace=True)
    column1, column2 = dataframe.columns[:2]
    assert column1 == 'Name', 'Column 1 != "Name"'
    assert column2 == 'Description', 'Column 2 != "Description"'

    #
    dataframe.set_index('Name', inplace=True)
    dataframe.index.name = None

    #
    description = dataframe['Description']
    dataframe.drop('Description', axis=1, inplace=True)

    return dataframe, description


def write_gct(dataframe, filename, description=None, index_column=None, verbose=False):
    """
    Write a `dataframe` to `filename` as .gct.
    """
    
    # Set output filename
    if not filename.endswith('.gct'):
        filename += '.gct'
        
    # Set index (Name)
    if index_column:
        dataframe.set_index(index_column, inplace=True)
    dataframe.index.name = 'Name'
    
    n_rows, n_cols = dataframe.shape[0], dataframe.shape[1]
    
    # Set Description
    if description:
        assert len(description) == n_rows
    else:
        description = dataframe.index
    dataframe.insert(0, 'Description', description)

    with open(filename, 'w') as f:
        f.writelines('#1.2\n{}\t{}\n'.format(n_rows, n_cols))
        dataframe.to_csv(f, sep='\t')


# ======================================================================================================================
# Data structure operations
# ======================================================================================================================
def filter_features_and_samples(dataframe, features, samples, verbose=False):
    """
    Filter a subset of features and samples.
    """
