from numpy import unique

from .check_nd_array_for_bad import check_nd_array_for_bad
from .COLORS import COLORS
from .get_colormap_colors import get_colormap_colors


def pick_nd_array_colors(nd_array, data_type):

    is_good = ~check_nd_array_for_bad(nd_array, raise_for_bad=False)

    if data_type == "continuous":

        return tuple(get_colormap_colors("bwr"))

    elif data_type == "categorical":

        return COLORS["curated"][: unique(nd_array[is_good]).size]

    elif data_type == "binary":

        return COLORS["white_black"]
