"""
Computational Cancer Biology Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Biology, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Biology, UCSD Cancer Center


Description:
"""
import datetime
import pandas as pd

VERBOSE = True


# ======================================================================================================================
# Utilities
# ======================================================================================================================
def verbose_print(string):
    """
    Print `string`.
    :param string: str, message to be printed
    :return:
    """
    if VERBOSE:
        print('{} {}'.format(datetime.datetime.now().time(), string))


# ======================================================================================================================
# File operations
# ======================================================================================================================
def read_gct(filename, fill_na=None):
    """
    Read `filename` (.gct) and convert it into a DataFrame.
    :param filename:
    :param fill_na:
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


def write_gct(dataframe, filename, description=None, index_column=None):
    """
    Write a `dataframe` to `filename` as .gct.
    :param dataframe:
    :param filename:
    :param description:
    :param index_column:
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
