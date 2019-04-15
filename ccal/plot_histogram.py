from pandas import Series

from .plot_and_save import plot_and_save


def plot_histogram(
    xs,
    names=None,
    texts=None,
    histnorm="",
    plot_rug=True,
    layout_width=None,
    layout_height=None,
    title=None,
    xaxis_title=None,
    html_file_path=None,
):

    if plot_rug:

        yaxis_max = 0.16

        yaxis2_min = yaxis_max + 0.08

    else:

        yaxis_max = 0

        yaxis2_min = 0

    data = []

    for i, x in enumerate(xs):

        if names is None:

            name = None

        else:

            name = names[i]

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "name": name,
                "legendgroup": i,
                "x": x,
                "histnorm": histnorm,
                "opacity": 0.8,
            }
        )

        if plot_rug:

            if texts is None:

                if isinstance(x, Series):

                    text = x.index

                else:

                    text = None

            else:

                text = texts[i]

            data.append(
                {
                    "type": "scatter",
                    "legendgroup": i,
                    "showlegend": False,
                    "x": x,
                    "y": (i,) * len(x),
                    "text": text,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                }
            )

    plot_and_save(
        {
            "layout": {
                "width": layout_width,
                "height": layout_height,
                "title": {"text": title},
                "xaxis": {"anchor": "y", "title": xaxis_title},
                "yaxis": {
                    "domain": (0, yaxis_max),
                    "dtick": 1,
                    "zeroline": False,
                    "showticklabels": False,
                },
                "yaxis2": {"domain": (yaxis2_min, 1), "title": histnorm.title()},
                "barmode": "overlay",
            },
            "data": data,
        },
        html_file_path,
    )
