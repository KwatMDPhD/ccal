from numpy import nanmax, nanmin, unique
from pandas import DataFrame, Series

from .nd_array.nd_array.normalize_nd_array import normalize_nd_array
from .plot.plot.make_colorscale import make_colorscale
from .plot.plot.style import (
    BINARY_COLORS_WHITE_BLACK,
    CATEGORICAL_COLORS,
    CONTINUOUS_COLORSCALE_FOR_MATCH,
)


def _process_target_or_features_for_plotting(target_or_features, type_, plot_std):

    if type_ == "continuous":

        if isinstance(target_or_features, Series):

            target_or_features = Series(
                normalize_nd_array(
                    target_or_features.values, None, "-0-", raise_for_bad=False
                ),
                name=target_or_features.name,
                index=target_or_features.index,
            )

        elif isinstance(target_or_features, DataFrame):

            target_or_features = DataFrame(
                normalize_nd_array(
                    target_or_features.values, 1, "-0-", raise_for_bad=False
                ),
                index=target_or_features.index,
                columns=target_or_features.columns,
            )

        target_or_features_nanmin = nanmin(target_or_features.values)

        target_or_features_nanmax = nanmax(target_or_features.values)

        if plot_std is None:

            plot_min = target_or_features_nanmin

            plot_max = target_or_features_nanmax

        else:

            plot_min = -plot_std

            plot_max = plot_std

        colorscale = CONTINUOUS_COLORSCALE_FOR_MATCH

    else:

        plot_min = None

        plot_max = None

        if type_ == "categorical":

            n_color = unique(target_or_features).size

            colorscale = make_colorscale(
                colors=CATEGORICAL_COLORS[:n_color], plot=False
            )

        elif type_ == "binary":

            colorscale = make_colorscale(colors=BINARY_COLORS_WHITE_BLACK, plot=False)

    return target_or_features, plot_min, plot_max, colorscale
