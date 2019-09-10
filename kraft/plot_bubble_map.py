from numpy import arange, meshgrid

from .COLORBAR import COLORBAR
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .normalize_array import normalize_array
from .plot_plotly_figure import plot_plotly_figure


def plot_bubble_map(
    dataframe_size,
    dataframe_color=None,
    marker_size_max=20,
    colorscale=None,
    layout=None,
    html_file_path=None,
):

    x_grid = arange(dataframe_size.shape[1])

    y_grid = arange(dataframe_size.shape[0])[::-1]

    layout_template = {
        "xaxis": {
            "title": dataframe_size.columns.name,
            "tickvals": x_grid,
            "ticktext": dataframe_size.columns,
        },
        "yaxis": {
            "title": dataframe_size.index.name,
            "tickvals": y_grid,
            "ticktext": dataframe_size.index,
        },
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    if dataframe_color is None:

        dataframe_color = dataframe_size

    mesh_grid_x, mesh_grid_y = meshgrid(x_grid, y_grid)

    plot_plotly_figure(
        {
            "layout": layout,
            "data": [
                {
                    "type": "scatter",
                    "x": mesh_grid_x.ravel(),
                    "y": mesh_grid_y.ravel(),
                    "text": dataframe_size.values.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize_array(dataframe_size.values, "0-1").ravel()
                        * marker_size_max,
                        "color": dataframe_color.values.ravel(),
                        "colorscale": colorscale,
                        "colorbar": COLORBAR,
                    },
                }
            ],
        },
        html_file_path,
    )
