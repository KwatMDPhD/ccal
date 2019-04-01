from .COLOR_CATEGORICAL import COLOR_CATEGORICAL
from .plot_and_save import plot_and_save


def plot_bar(
    xs,
    ys,
    names=None,
    colors=None,
    position_labels=None,
    position_label_colors=None,
    position_label_size=0.16,
    orientation="v",
    layout_width=None,
    layout_height=None,
    title=None,
    xaxis_title=None,
    xaxis_dtick=None,
    yaxis_title=None,
    yaxis_dtick=None,
    barmode=None,
    html_file_path=None,
    plotly_html_file_path=None,
):

    layout = {
        "width": layout_width,
        "height": layout_height,
        "title": title,
        "barmode": barmode,
    }

    if position_labels is None:

        position_label_size = 0

    else:

        position_label_size = position_label_size

    if orientation == "v":

        axes = ("xaxis", "yaxis", "yaxis2")

        anchor = "y2"

    elif orientation == "h":

        axes = ("yaxis", "xaxis", "xaxis2")

        anchor = "x2"

    axis_dicts = (
        {"ticks": "outside", "anchor": anchor},
        {
            "domain": (0, position_label_size),
            "showgrid": False,
            "zeroline": False,
            "ticks": "",
            "showticklabels": False,
        },
        {"domain": (position_label_size, 1)},
    )

    for axis, axis_dict in zip(axes, axis_dicts):

        layout[axis] = axis_dict

    if orientation == "v":

        layout["xaxis"].update({"title": xaxis_title, "dtick": xaxis_dtick})

        layout["yaxis2"].update({"title": yaxis_title, "dtick": yaxis_dtick})

    elif orientation == "h":

        layout["xaxis2"].update({"title": xaxis_title, "dtick": xaxis_dtick})

        layout["yaxis"].update({"title": yaxis_title, "dtick": yaxis_dtick})

    data = []

    for i, (x, y) in enumerate(zip(xs, ys)):

        if names is None:

            name = None

        else:

            name = names[i]

        if colors is None:

            color = COLOR_CATEGORICAL[i]

        else:

            color = colors[i]

        trace = {
            "type": "bar",
            "name": name,
            "x": x,
            "y": y,
            "orientation": orientation,
            "marker": {"color": color},
        }

        if orientation == "v":

            trace.update({"yaxis": "y2"})

        elif orientation == "h":

            trace.update({"xaxis": "x2"})

        data.append(trace)

    if position_labels is not None:

        layout_annotations = []

        for i, label in enumerate(position_labels):

            if position_label_colors is None:

                color = None

            else:

                color = position_label_colors[i]

            annotation = {
                "xref": "x",
                "yref": "y",
                "text": "<b>{}</b>".format(label),
                "font": {"color": color},
                "bordercolor": "#ebf6f7",
                "showarrow": False,
            }

            if orientation == "v":

                annotation.update({"x": i, "y": 0, "textangle": -90})

            elif orientation == "h":

                annotation.update({"x": 0, "y": i, "textangle": 0})

            layout_annotations.append(annotation)

        layout.update({"annotations": layout_annotations})

    plot_and_save(
        {"layout": layout, "data": data}, html_file_path, plotly_html_file_path
    )
