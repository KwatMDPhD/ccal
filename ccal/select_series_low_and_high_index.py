from ccal import plot_points
from pandas import Series


def select_series_low_and_high_index(
        values,
        low_margin_factor=1,
        high_margin_factor=1,
        title=None,
        value_name=None,
        file_path_prefix=None,
):

    values = values.sort_values()

    margin = values.std() / 2

    low_index = values.index[
        values < values.mean() - margin * low_margin_factor]

    high_index = values.index[
        values.mean() + margin * high_margin_factor < values]

    rank = Series(
        range(values.size),
        index=values.index,
    )

    if file_path_prefix is not None:

        Series(
            low_index,
            name='Low Index',
        ).to_csv(
            '{}.low.tsv'.format(file_path_prefix),
            sep='\t',
        )

        Series(
            high_index,
            name='High Index',
        ).to_csv(
            '{}.high.tsv'.format(file_path_prefix),
            sep='\t',
        )

        html_file_path = '{}.low_and_high.html'.format(file_path_prefix)

    else:

        html_file_path = None

    if value_name is None:

        value_name = 'Value'

    plot_points(
        (
            rank,
            rank[high_index],
            rank[low_index],
        ),
        (
            values,
            values[high_index],
            values[low_index],
        ),
        names=(
            'All',
            'High',
            'Low',
        ),
        modes=('lines', ) * 3,
        texts=(
            values.index,
            high_index,
            low_index,
        ),
        title=title,
        xaxis_title='Rank',
        yaxis_title=value_name,
        html_file_path=html_file_path,
    )

    return low_index, high_index
