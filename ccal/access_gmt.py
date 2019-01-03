from warnings import warn

from pandas import DataFrame, concat


def read_gmts(gmt_file_paths, sets=None, drop_description=True, collapse=False):

    dfs = []

    for gmt_file_path in gmt_file_paths:

        dfs.append(read_gmt(gmt_file_path, drop_description=drop_description))

    df = concat(dfs, sort=True)

    if sets is not None:

        df = df.loc[(df.index & sets)].dropna(axis=1, how="all")

    if collapse:

        return df.unstack().dropna().sort_values().unique()

    else:

        return df


def read_gmt(gmt_file_path, drop_description=True):

    lines = []

    with open(gmt_file_path) as gmt_file:

        for line in gmt_file:

            split = line.strip().split(sep="\t")

            lines.append(split[:2] + [gene for gene in set(split[2:]) if gene])

    df = DataFrame(lines)

    df.set_index(0, inplace=True)

    df.index.name = "Gene Set"

    if drop_description:

        df.drop(1, axis=1, inplace=True)

        df.columns = tuple("Gene {}".format(i) for i in range(0, df.shape[1]))

    else:

        df.columns = ("Description",) + tuple(
            "Gene {}".format(i) for i in range(0, df.shape[1] - 1)
        )

    return df


def write_gmt(df, gmt_file_path, descriptions=None):

    df = df.copy()

    if df.columns[0] != "Description":

        if descriptions is not None:

            df.insert(0, "Description", descriptions)

        else:

            df.insert(0, "Description", df.index)

    if not gmt_file_path.endswith(".gmt"):

        warn("Adding .gmt to {} ...".format(gmt_file_path))

        gmt_file_path += ".gmt"

    df.to_csv(gmt_file_path, header=None, sep="\t")
