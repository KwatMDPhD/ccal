from pandas import Series

from ccal import plot_points


def select_series_low_and_high_index(
    values,
    low_margin_factor=1,
    high_margin_factor=1,
    title=None,
    value_name="Value",
    html_file_path=None,
):

    values = values.sort_values()

    margin = values.std() / 2

    low_index = values.index[values < values.mean() - margin * low_margin_factor]

    high_index = values.index[values.mean() + margin * high_margin_factor < values]

    rank = Series(range(values.size), index=values.index)

    if values.size < 1e3:

        mode = "markers"

    else:

        mode = "lines"

    plot_points(
        (rank, rank[high_index], rank[low_index]),
        (values, values[high_index], values[low_index]),
        names=("All", "High", "Low"),
        modes=(mode,) * 3,
        texts=(values.index, high_index, low_index),
        title=title,
        xaxis_title="Rank",
        yaxis_title=value_name,
        html_file_path=html_file_path,
    )

    return low_index, high_index
