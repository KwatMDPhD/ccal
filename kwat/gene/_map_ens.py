from pandas import read_csv

from ..constant import DATA_DIRECTORY
from ..dataframe import map_to


def _map_ens():

    return map_to(
        read_csv("{}ens.tsv.gz".format(DATA_DIRECTORY), sep="\t"), "Gene name"
    )