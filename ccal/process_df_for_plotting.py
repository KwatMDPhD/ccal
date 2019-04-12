from numpy import asarray, sort
from pandas import DataFrame

from .check_nd_array_for_bad import check_nd_array_for_bad
from .cluster_2d_array_slices import cluster_2d_array_slices
from .normalize_nd_array import normalize_nd_array


def process_df_for_plotting(
    df,
    normalization_axis=None,
    normalization_method=None,
    row_annotation=None,
    column_annotation=None,
    cluster_axis=None,
    cluster_distance_function="euclidean",
    cluster_linkage_method="ward",
    sort_axis=None,
):

    if normalization_method:

        df = DataFrame(
            normalize_nd_array(
                df.values, normalization_axis, normalization_method, raise_for_bad=False
            ),
            index=df.index,
            columns=df.columns,
        )

    if row_annotation is not None or column_annotation is not None:

        if row_annotation is not None:

            row_indices = asarray(row_annotation).argsort()

            row_annotation = row_annotation[row_indices]

            df = df.iloc[row_annotation]

        if column_annotation is not None:

            column_indices = asarray(column_annotation).argsort()

            column_annotation = column_annotation[column_indices]

            df = df.iloc[:, column_indices]

    elif cluster_axis is not None:

        if not check_nd_array_for_bad(df.values, raise_for_bad=False).any():

            if cluster_axis in (None, 0):

                df = df.iloc[
                    cluster_2d_array_slices(
                        df.values,
                        0,
                        distance_function=cluster_distance_function,
                        linkage_method=cluster_linkage_method,
                    )
                ]

            if cluster_axis in (None, 1):

                df = df.iloc[
                    :,
                    cluster_2d_array_slices(
                        df.values,
                        1,
                        distance_function=cluster_distance_function,
                        linkage_method=cluster_linkage_method,
                    ),
                ]

    elif sort_axis in (0, 1):

        df = DataFrame(sort(df.values, axis=sort_axis), index=None, columns=None)

    return df
