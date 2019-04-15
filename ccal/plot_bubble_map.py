from numpy import linspace, meshgrid

from .make_colorscale_from_colors import make_colorscale_from_colors
from .normalize_nd_array import normalize_nd_array
from .pick_nd_array_colors import pick_nd_array_colors
from .plot_and_save import plot_and_save


def plot_bubble_map(
    df_size,
    df_color=None,
    marker_size_max=32,
    data_type="continuous",
    showscale=None,
    colorbar_x=None,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    html_file_path=None,
):

    if df_color is None:

        df_color = df_size

    axis_template = {"zeroline": False}

    x, y = meshgrid(
        linspace(0, df_size.shape[1] - 1, df_size.shape[1]),
        linspace(0, df_size.shape[0] - 1, df_size.shape[0]),
    )

    plot_and_save(
        {
            "layout": {
                "width": max(640, marker_size_max * 2 * df_size.shape[1]),
                "height": max(640, marker_size_max * 2 * df_size.shape[0]),
                "title": {"text": title},
                "xaxis": {
                    "tickvals": tuple(range(df_size.shape[1])),
                    "ticktext": df_size.columns,
                    "title": "{} ({})".format(xaxis_title, df_size.shape[1]),
                    **axis_template,
                },
                "yaxis": {
                    "tickvals": tuple(range(df_size.shape[0])),
                    "ticktext": df_size.index[::-1],
                    "title": "{} ({})".format(yaxis_title, df_size.shape[0]),
                    **axis_template,
                },
            },
            "data": [
                {
                    "type": "scatter",
                    "x": x.ravel(),
                    "y": y.ravel()[::-1],
                    "text": df_size.values.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize_nd_array(df_size.values, None, "0-1").ravel()
                        * marker_size_max,
                        "color": df_color.values.ravel(),
                        "colorscale": make_colorscale_from_colors(
                            pick_nd_array_colors(df_color.values, data_type)
                        ),
                        "showscale": True,
                        "colorbar": {
                            "x": colorbar_x,
                            "len": 0.64,
                            "thickness": marker_size_max / 3,
                        },
                        "line": {"color": "#000000"},
                    },
                }
            ],
        },
        html_file_path,
    )
