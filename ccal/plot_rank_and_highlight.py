from numpy import nonzero

from .plot_and_save import plot_and_save


def plot_rank_and_highlight(
    element_value,
    highlights,
    title=None,
    yaxis=None,
    html_file_path=None,
    plotly_html_file_path=None,
):

    element_value = element_value.sort_values()

    negative_indices = nonzero(element_value < 0)[0]

    positive_indices = nonzero(0 < element_value)[0]

    opacity = 0.16

    if highlights is None:

        highlight_indices = ()

    else:

        highlight_indices = nonzero(
            [gene_set in highlights for gene_set in element_value.index]
        )[0]

    highlights_marker_size = 16

    plot_and_save(
        dict(
            layout=dict(
                title=title,
                xaxis=dict(title="Rank"),
                yaxis=yaxis,
                annotations=[
                    dict(
                        x=x,
                        y=abs(element_value[x]),
                        text=element_value.index[x],
                        font=dict(size=10),
                        arrowhead=2,
                        arrowsize=0.8,
                        clicktoshow="onoff",
                    )
                    for x in highlight_indices
                ],
            ),
            data=[
                dict(
                    type="scatter",
                    name="-",
                    x=negative_indices,
                    y=-element_value[negative_indices],
                    text=element_value[negative_indices].index,
                    mode="markers",
                    marker=dict(color="#0088ff", opacity=opacity),
                ),
                dict(
                    type="scatter",
                    name="+",
                    x=positive_indices,
                    y=element_value[positive_indices],
                    text=element_value[positive_indices].index,
                    mode="markers",
                    marker=dict(color="#ff1968", opacity=opacity),
                ),
                dict(
                    type="scatter",
                    name="highlights",
                    x=highlight_indices,
                    y=abs(element_value[highlight_indices]),
                    text=element_value[highlight_indices].index,
                    mode="markers",
                    marker=dict(
                        color="#20d9ba",
                        size=highlights_marker_size,
                        line=dict(width=highlights_marker_size / 8, color="#ebf6f7"),
                    ),
                ),
            ],
        ),
        html_file_path,
        plotly_html_file_path,
    )
