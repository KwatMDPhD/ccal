from ccal import normalize_nd_array
from pandas import DataFrame


def normalize_df(df, axis, method, raise_for_bad=True):

    return DataFrame(
        normalize_nd_array(df.values, axis, method, raise_for_bad=raise_for_bad),
        index=df.index,
        columns=df.columns,
    )
