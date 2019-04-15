from numpy import unique

from .COLORS import COLORS
from .get_colormap_colors import get_colormap_colors


def match_colors_to_data(data, data_type):

    if data_type == "continuous":

        return tuple(get_colormap_colors("bwr"))

    elif data_type == "categorical":

        return COLORS["curated"][: unique(data).size]

    elif data_type == "binary":

        return COLORS["white_black"]
