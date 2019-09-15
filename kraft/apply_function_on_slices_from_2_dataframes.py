from pandas import DataFrame

from .apply_function_on_slices_from_2_matrices import (
    apply_function_on_slices_from_2_matrices,
)
from .cluster_matrix import cluster_matrix
from .plot_heat_map import plot_heat_map


def apply_function_on_slices_from_2_dataframes(
    dataframe_0, dataframe_1, axis, function, layout=None, file_path_prefix=None
):

    comparison = apply_function_on_slices_from_2_matrices(
        dataframe_0.values, dataframe_1.values, axis, function
    )

    if axis == 0:

        index_and_columns = dataframe_0.columns

    elif axis == 1:

        index_and_columns = dataframe_0.index

    comparison = DataFrame(
        comparison, index=index_and_columns, columns=index_and_columns
    )

    if file_path_prefix is None:

        html_file_path = None

    else:

        comparison.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

        html_file_path = "{}.html".format(file_path_prefix)

    plot_heat_map(
        comparison.iloc[
            cluster_matrix(comparison.values, 0), cluster_matrix(comparison.values, 1)
        ],
        layout=layout,
        html_file_path=html_file_path,
    )

    return comparison
