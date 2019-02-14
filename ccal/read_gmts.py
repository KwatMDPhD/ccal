from pandas import concat

from .read_gmt import read_gmt


def read_gmts(gmt_file_paths, collapse=False):

    df = concat(
        tuple(read_gmt(gmt_file_path) for gmt_file_path in gmt_file_paths), sort=True
    )

    if collapse:

        return sorted(set(df.unstack().dropna()))

    else:

        return df
