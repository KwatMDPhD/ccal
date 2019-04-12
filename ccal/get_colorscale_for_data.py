from numpy import unique

from .COLOR_CATEGORICAL import COLOR_CATEGORICAL
from .COLOR_WHITE_BLACK import COLOR_WHITE_BLACK
from .get_colormap_colors import get_colormap_colors
from .make_colorscale_from_colors import make_colorscale_from_colors


def get_colorscale_for_data(data, data_type):

    if data_type == "continuous":

        colorscale = make_colorscale_from_colors(get_colormap_colors("bwr"))

    elif data_type == "categorical":

        colorscale = make_colorscale_from_colors(COLOR_CATEGORICAL[: unique(data).size])

    elif data_type == "binary":

        colorscale = make_colorscale_from_colors(COLOR_WHITE_BLACK)

    return colorscale
