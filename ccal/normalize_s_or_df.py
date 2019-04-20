from pandas import DataFrame, Series

from .normalize_nd_array import normalize_nd_array


def normalize_s_or_df(s_or_df, axis, method, rank_method="average"):

    s_or_df_normalized = type(s_or_df)(
        normalize_nd_array(
            s_or_df.values, axis, method, rank_method=rank_method, raise_for_bad=False
        )
    )

    if isinstance(s_or_df, Series):

        s_or_df_normalized.name = s_or_df.name

        s_or_df_normalized.index = s_or_df.index

    elif isinstance(s_or_df, DataFrame):

        s_or_df_normalized.index = s_or_df.index

        s_or_df_normalized.columns = s_or_df.columns

    return s_or_df_normalized
