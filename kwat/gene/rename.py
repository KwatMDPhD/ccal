from os.path import join
from re import search

from numpy import logical_not
from pandas import read_csv, read_excel

from ..constant import DATA_DIRECTORY_PATH
from ..dataframe import map_to
from ..dictionary import clean, rename as dictionary_rename
from ..string import split_and_get
from ._read import _read


def _split(an):

    if isinstance(an, str):

        return an.split(sep="|")

    else:

        return []


def _update(da, di):

    og_, sy_ = da.values.T

    hu_ = og_ == "human"

    if hu_.sum() == 1:

        geh = sy_[hu_][0]

        for gem in sy_[logical_not(hu_)]:

            di[gem] = geh

    else:

        return

        print("-" * 80)

        print("\n".join("({}) {}".format(og, sy) for og, sy in zip(og_, sy_)))


def rename(na_, **ke_ar):

    na_re = {}

    read_csv(
        join(DATA_DIRECTORY_PATH, "HOM_MouseHumanSequence.rpt.txt.gz"),
        sep="\t",
        usecols=[0, 1, 3],
        index_col=0,
    ).groupby(level=0).apply(_update, na_re)

    for cg_re in [
        read_excel(
            join(DATA_DIRECTORY_PATH, "illumina_humanmethylation27_content.xlsx"),
            usecols=[0, 10],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            join(DATA_DIRECTORY_PATH, "HumanMethylation450_15017482_v1-2.csv.gz"),
            skiprows=7,
            usecols=[0, 21],
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            join(
                DATA_DIRECTORY_PATH,
                "infinium-methylationepic-v-1-0-b5-manifest-file.csv.gz",
            ),
            skiprows=7,
            usecols=[0, 15],
            index_col=0,
            squeeze=True,
        ),
    ]:

        for cg, re in cg_re.dropna().iteritems():

            na_re[cg] = split_and_get(re, ";", 0)

    na_re.update(
        map_to(read_csv(join(DATA_DIRECTORY_PATH, "ens.tsv.gz"), sep="\t"), "Gene name")
    )

    na_re.update(
        map_to(
            _read({}).drop(
                labels=[
                    "locus_group",
                    "locus_type",
                    "status",
                    "location",
                    "location_sortable",
                    "gene_family",
                    "gene_family_id",
                    "date_approved_reserved",
                    "date_symbol_changed",
                    "date_name_changed",
                    "date_modified",
                    "pubmed_id",
                    "lsdb",
                ],
                axis=1,
            ),
            "symbol",
            fu=_split,
        )
    )

    nap_ = []

    for na in na_:

        if search(r"^ENS[TG]", na):

            na = split_and_get(na, ".", 0)

        nap_.append(na)

    return dictionary_rename(nap_, clean(na_re), **ke_ar)
