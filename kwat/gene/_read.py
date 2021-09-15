from numpy import array, full
from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH


def _read(co_se):

    hg = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        low_memory=False,
    )

    se_ = full(hg.shape[0], True)

    for co, se in co_se.items():

        print("Selecting by {}: {}".format(co, se))

        se_ &= array([an in se for an in hg.loc[:, co].values])

        print("{}/{}".format(se_.sum(), se_.size))

    return hg.loc[se_, :]
