from numpy import asarray, nonzero, unique

from .make_colorscale_from_colors import make_colorscale_from_colors
from .match_colors_to_data import match_colors_to_data
from .plot_and_save import plot_and_save


def plot_heat_map(
    df,
    data_type="continuous",
    showscale=None,
    colorbar_x=None,
    layout_width=800,
    layout_height=800,
    heat_map_axis_domain=(0, 0.9),
    annotation_axis_domain=(0.92, 1),
    row_annotation=None,
    row_annotation_str=None,
    row_annotation_keyword_arguments=None,
    column_annotation=None,
    column_annotation_str=None,
    column_annotation_keyword_arguments=None,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    x_ticks=None,
    y_ticks=None,
    html_file_path=None,
):

    heat_map_axis_template = {
        "domain": heat_map_axis_domain,
        "showgrid": False,
        "zeroline": False,
    }

    annotation_axis_template = {"zeroline": False, "ticks": "", "showticklabels": False}

    layout = {
        "width": layout_width,
        "height": layout_height,
        "title": {"text": title},
        "xaxis": {
            "title": "{} ({})".format(xaxis_title, df.shape[1]),
            "ticks": x_ticks,
            "showticklabels": x_ticks,
            **heat_map_axis_template,
        },
        "xaxis2": {"domain": annotation_axis_domain, **annotation_axis_template},
        "yaxis": {
            "title": "{} ({})".format(yaxis_title, df.shape[0]),
            "ticks": y_ticks,
            "showticklabels": y_ticks,
            **heat_map_axis_template,
        },
        "yaxis2": {"domain": annotation_axis_domain, **annotation_axis_template},
    }

    colorbar_template = {"len": 0.64, "thickness": layout_width / 64}

    if row_annotation is not None or column_annotation is not None:

        colorbar_template["y"] = (heat_map_axis_domain[1] - heat_map_axis_domain[0]) / 2

    data = [
        {
            "type": "heatmap",
            "z": df.values[::-1],
            "x": df.columns,
            "y": df.index[::-1],
            "colorscale": make_colorscale_from_colors(
                match_colors_to_data(df.values, data_type)
            ),
            "showscale": showscale,
            "colorbar": {"x": colorbar_x, **colorbar_template},
        }
    ]

    if row_annotation is not None or column_annotation is not None:

        layout["annotations"] = []

        annotation_keyword_arguments = {"showarrow": False, "borderpad": 0}

        if row_annotation is not None:

            row_annotation = row_annotation[::-1]

            data.append(
                {
                    "xaxis": "x2",
                    "type": "heatmap",
                    "z": tuple((i,) for i in row_annotation),
                    "colorscale": make_colorscale_from_colors(
                        match_colors_to_data(asarray(row_annotation), "categorical")
                    ),
                    "showscale": False,
                    "hoverinfo": "y+z",
                }
            )

            if row_annotation_str is not None:

                if row_annotation_keyword_arguments is None:

                    row_annotation_keyword_arguments = {}

                for a in unique(row_annotation):

                    indices = nonzero(row_annotation == a)[0]

                    index_0 = indices[0]

                    layout["annotations"].append(
                        {
                            "xref": "x2",
                            "x": 0,
                            "y": index_0 + (indices[-1] - index_0) / 2,
                            "text": "<b>{}</b>".format(row_annotation_str[a]),
                            **annotation_keyword_arguments,
                            **row_annotation_keyword_arguments,
                        }
                    )

        if column_annotation is not None:

            data.append(
                {
                    "yaxis": "y2",
                    "type": "heatmap",
                    "z": tuple((i,) for i in column_annotation),
                    "transpose": True,
                    "colorscale": make_colorscale_from_colors(
                        match_colors_to_data(asarray(column_annotation), "categorical")
                    ),
                    "showscale": False,
                    "hoverinfo": "x+z",
                }
            )

            if column_annotation_str is not None:

                if column_annotation_keyword_arguments is None:

                    column_annotation_keyword_arguments = {"textangle": -90}

                for a in unique(column_annotation):

                    indices = nonzero(column_annotation == a)[0]

                    index_0 = indices[0]

                    layout["annotations"].append(
                        {
                            "yref": "y2",
                            "x": index_0 + (indices[-1] - index_0) / 2,
                            "y": 0,
                            "text": "<b>{}</b>".format(column_annotation_str[a]),
                            **annotation_keyword_arguments,
                            **column_annotation_keyword_arguments,
                        }
                    )

    plot_and_save({"layout": layout, "data": data}, html_file_path)
