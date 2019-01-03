from matplotlib.colors import LinearSegmentedColormap

from .make_colorscale import make_colorscale

BAD_COLOR = "#f5deb3"

CATEGORICAL_COLORS = (
    "#20d9ba",
    "#9017e6",
    "#ff1968",
    "#ffe119",
    "#3cb44b",
    "#4e41d8",
    "#ffa400",
    "#aaffc3",
    "#800000",
    "#e6beff",
    "#fffac8",
    "#0082c8",
    "#e6194b",
    "#006442",
    "#46f0f0",
    "#bda928",
    "#c91f37",
    "#fabebe",
    "#d2f53c",
    "#aa6e28",
    "#ff0000",
    "#808000",
    "#003171",
    "#ff4e20",
    "#a4345d",
    "#ffd8b1",
    "#bb7796",
    "#f032e6",
)

BINARY_COLORS_WHITE_BLACK = ("#ebf6f7", "#171412")

BINARY_COLORS_WHITE_BROWN = ("#ebf6f7", "#181b26")

BINARY_COLORS_RUBY_EMERALD = ("#ff1968", "#20d9ba")

_rb = (0.26, 0.26, 0.26, 0.39, 0.69, 1, 1, 1, 1, 1, 1)

_g = (0.26, 0.16, 0.09, 0.26, 0.69)

CONTINUOUS_COLORSCALE_FOR_MATCH = make_colorscale(
    colormap=LinearSegmentedColormap(
        "association",
        dict(
            red=tuple((0.1 * i, j, j) for i, j in enumerate(_rb)),
            green=tuple((0.1 * i, j, j) for i, j in enumerate(_g + (1,) + _g[::-1])),
            blue=tuple((0.1 * i, j, j) for i, j in enumerate(_rb[::-1])),
        ),
    ),
    plot=False,
)
