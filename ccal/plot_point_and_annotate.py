from pandas import Series

from .plot_and_save import plot_and_save


def plot_point_and_annotate(
    x,
    y,
    abs_dimension,
    annotation=(),
    opacity=0.64,
    annotation_font_size=10,
    title=None,
    html_file_path=None,
    plotly_html_file_path=None,
):

    if x is None:

        x = Series(range(y.size), name="Rank", index=y.index)

    if abs_dimension == "x":

        is_negative = x < 0

    elif abs_dimension == "y":

        is_negative = y < 0

    data = [
        dict(
            type="scatter",
            name="-",
            x=x[is_negative],
            y=y[is_negative].abs(),
            text=y.index[is_negative],
            mode="markers",
            marker=dict(color="#0088ff", opacity=opacity),
        ),
        dict(
            type="scatter",
            name="+",
            x=x[~is_negative],
            y=y[~is_negative],
            text=y.index[~is_negative],
            mode="markers",
            marker=dict(color="#ff1968", opacity=opacity),
        ),
    ]

    annotations = []

    for group_name, elements, size, color in annotation:

        group_elements = y.index & elements

        group_x = x[group_elements]

        group_y = y[group_elements]

        data.append(
            dict(
                type="scatter",
                name=group_name,
                x=group_x,
                y=group_y.abs(),
                text=group_elements,
                mode="markers",
                marker=dict(
                    size=size, color=color, line=dict(width=1, color="#ebf6f7")
                ),
            )
        )

        annotations += [
            dict(
                x=x_,
                y=abs(y_),
                text=element,
                font=dict(size=annotation_font_size),
                arrowhead=2,
                arrowsize=0.8,
                clicktoshow="onoff",
            )
            for element, x_, y_ in zip(group_elements, group_x, group_y)
        ]

    plot_and_save(
        dict(
            layout=dict(
                title=title,
                xaxis=dict(title=x.name),
                yaxis=dict(title=y.name),
                annotations=annotations,
            ),
            data=data,
        ),
        html_file_path,
        plotly_html_file_path,
    )
