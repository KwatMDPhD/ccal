from pandas import DataFrame

from .apply_function_on_2_2d_arrays_slices import apply_function_on_2_2d_arrays_slices
from .compute_information_coefficient_between_2_1d_arrays import (
    compute_information_coefficient_between_2_1d_arrays,
)
from .plot_heat_map import plot_heat_map
from .cluster_2d_array import cluster_2d_array


def apply_function_on_2_dfs_slices(
    df_0,
    df_1,
    axis,
    function=compute_information_coefficient_between_2_1d_arrays,
    title=None,
    name_0=None,
    name_1=None,
    file_path_prefix=None,
):

    comparison = apply_function_on_2_2d_arrays_slices(
        df_0.values, df_1.values, axis, function
    )

    if axis == 0:

        comparison = DataFrame(comparison, index=df_0.index, columns=df_1.index)

    elif axis == 1:

        comparison = DataFrame(comparison, index=df_0.columns, columns=df_1.columns)

    if file_path_prefix is None:

        html_file_path = None

    else:

        comparison.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

        html_file_path = "{}.html".format(file_path_prefix)

    plot_heat_map(
        comparison.iloc[
            cluster_2d_array(comparison.values, 0),
            cluster_2d_array(comparison.values, 1),
        ],
        title=title,
        xaxis_title=name_1,
        yaxis_title=name_0,
        html_file_path=html_file_path,
    )

    return comparison
