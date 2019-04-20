from .COLORS import COLORS
from .plot_and_save import plot_and_save


def plot_histogram(
    serieses,
    histnorm="",
    plot_rug=True,
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

    for i, series in enumerate(serieses):

        color = COLORS["curated"][i]

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "name": series.name,
                "legendgroup": series.name,
                "x": series,
                "histnorm": histnorm,
                "marker": {"color": color},
            }
        )

        if plot_rug:

            data.append(
                {
                    "type": "scatter",
                    "legendgroup": series.name,
                    "showlegend": False,
                    "x": series,
                    "y": (i,) * series.size,
                    "text": series.index,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                }
            )

    plot_and_save(
        {
            "layout": {
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
