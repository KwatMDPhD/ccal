from numpy.random import choice
from pandas import Series

from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram


def summarize_feature_x_sample(
    feature_x_sample,
    feature_x_sample_alias="Feature-x-Sample",
    feature_x_sample_value_name="Value",
    plot=True,
    plot_heat_map_max_size=640000,
    plot_histogram_max_size=64000,
    plot_rug_max_size=6400,
):

    print("Shape: {}".format(feature_x_sample.shape))

    print("Min: {}".format(feature_x_sample.values.min()))

    print("Max: {}".format(feature_x_sample.values.max()))

    for axis in (0, 1):

        n_unique_value_count = feature_x_sample.apply(
            lambda series: series.unique().size, axis=axis
        ).value_counts()

        n_unique_value_count.index.name = "N Unique"

        n_unique_value_count.name = "Count"

        n_unique_value_count = n_unique_value_count.to_frame()

        n_extreme = 8

        if n_extreme * 2 < n_unique_value_count.shape[0]:

            print(
                "Axis {} Top and Bottom {} Number of Unique Values:".format(
                    axis, n_extreme
                )
            )

            print(
                n_unique_value_count.iloc[
                    list(range(n_extreme)) + list(range(-n_extreme, 0))
                ]
            )

        else:

            print("Axis {} Number of Unique Values:".format(axis))

            print(n_unique_value_count)

    if plot:

        if feature_x_sample.size < plot_heat_map_max_size:

            plot_heat_map(
                feature_x_sample,
                title=feature_x_sample_alias,
                xaxis_title=feature_x_sample.columns.name,
                yaxis_title=feature_x_sample.index.name,
            )

        feature_x_sample_not_na_values = feature_x_sample.unstack().dropna()

        if plot_histogram_max_size < feature_x_sample_not_na_values.size:

            print("Sampling random {:,} values ...".format(plot_histogram_max_size))

            feature_x_sample_not_na_values = choice(
                feature_x_sample_not_na_values,
                size=plot_histogram_max_size,
                replace=False,
            )

        value_name = "Not-NA Value"

        plot_histogram(
            (Series(feature_x_sample_not_na_values),),
            plot_rug=feature_x_sample_not_na_values.size < plot_rug_max_size,
            title="{}<br>Histogram of {}".format(feature_x_sample_alias, value_name),
            xaxis_title=value_name,
        )

    isna__feature_x_sample = feature_x_sample.isna()

    n_na = isna__feature_x_sample.values.sum()

    print("N NA: {} ({:.2f}%)".format(n_na, n_na / feature_x_sample.size * 100))

    if n_na and plot and isna__feature_x_sample.size < plot_histogram_max_size:

        value_name = "N NA"

        plot_histogram(
            (isna__feature_x_sample.sum(axis=1), isna__feature_x_sample.sum()),
            plot_rug=max(isna__feature_x_sample.shape) < plot_rug_max_size,
            title="{}<br>Histogram of {} ".format(feature_x_sample_alias, value_name),
            xaxis_title=value_name,
        )
