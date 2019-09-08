from numpy import absolute, nanmax
from pandas import Series

from .compute_vector_context import compute_vector_context
from .plot_plotly_figure import plot_plotly_figure


def plot_context(
    _vector_or_series,
    text=None,
    n_data=None,
    location=None,
    scale=None,
    degree_of_freedom=None,
    shape=None,
    fit_initial_location=None,
    fit_initial_scale=None,
    n_grid=1e3,
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
    y_max_is_pdf_max=False,
    n_bin=None,
    plot_rug=None,
    layout=None,
    xaxis=None,
    html_file_path=None,
):

    if isinstance(_vector_or_series, Series):

        if title_text is None:

            title_text = _vector_or_series.name

        if xaxis_title_text is None:

            xaxis_title_text = "Value"

        if text is None:

            text = _vector_or_series.index

        vector = _vector_or_series.values

    else:

        vector = _vector_or_series

    context_dict = compute_vector_context(
        vector,
        n_data=n_data,
        location=location,
        scale=scale,
        degree_of_freedom=degree_of_freedom,
        shape=shape,
        fit_initial_location=fit_initial_location,
        fit_initial_scale=fit_initial_scale,
        n_grid=n_grid,
        degree_of_freedom_for_tail_reduction=degree_of_freedom_for_tail_reduction,
        multiply_distance_from_reference_argmax=multiply_distance_from_reference_argmax,
        global_location=global_location,
        global_scale=global_scale,
        global_degree_of_freedom=global_degree_of_freedom,
        global_shape=global_shape,
    )

    pdf_max = context_dict["pdf"].max()

    context = context_dict["context"]

    absolute_context = absolute(context)

    absolute_context_max = nanmax(absolute_context)

    if y_max_is_pdf_max:

        y_max = pdf_max

        if y_max < absolute_context_max:

            absolute_context = absolute_context / absolute_context_max * y_max

    else:

        y_max = max(pdf_max, absolute_context_max)

    if plot_rug is None:

        plot_rug = _vector_or_series.size < 1e3

    if plot_rug:

        yaxis_max = 0.16

        yaxis2_min = yaxis_max + 0.08

    else:

        yaxis_max = 0

        yaxis2_min = 0

    layout = {
        "xaxis": {"anchor": "y", "title": {"text": xaxis_title_text}},
        "yaxis": {
            "domain": (0, yaxis_max),
            "dtick": 1,
            "zeroline": False,
            "showticklabels": False,
        },
        "yaxis2": {"domain": (yaxis2_min, 1)},
        "legend": {"orientation": "h", "xanchor": "center", "x": 0.5, "y": -0.2},
        **layout,
    }

    annotations = []

    for i, (template, fit_parameter) in enumerate(
        zip(
            (
                "N = {:.0f}",
                "Location = {:.2f}",
                "Scale = {:.2f}",
                "DF = {:.2f}",
                "Shape = {:.2f}",
            ),
            context_dict["fit"],
        )
    ):

        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": (i + 1) / (5 + 1),
                "y": 1.064,
                "xanchor": "center",
                "text": template.format(fit_parameter),
                "showarrow": False,
            }
        )

    layout.update({"annotations": annotations})

    data = []

    data.append(
        {
            "yaxis": "y2",
            "type": "histogram",
            "name": "Data",
            "legendgroup": "Data",
            "x": vector,
            "marker": {"color": "#20d9ba"},
            "histnorm": "probability density",
            "hoverinfo": "x+y",
        }
    )

    if n_bin is not None:

        _vector_min = vector.min()

        _vector_max = vector.max()

        data[-1]["xbins"] = {
            "start": _vector_min,
            "end": _vector_max,
            "size": (_vector_max - _vector_min) / n_bin,
        }

    if plot_rug:

        data.append(
            {
                "type": "scatter",
                "legendgroup": "Data",
                "showlegend": False,
                "x": vector,
                "y": (0,) * vector.size,
                "text": text,
                "mode": "markers",
                "marker": {"symbol": "line-ns-open", "color": "#20d9ba"},
                "hoverinfo": "x+text",
            }
        )

    grid = context_dict["grid"]

    line_width = 3.2

    pdf = context_dict["pdf"]

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "PDF",
            "x": grid,
            "y": pdf,
            "line": {"width": line_width, "color": "#24e7c0"},
        }
    )

    shape_pdf_reference = context_dict["shape_pdf_reference"]

    shape_pdf_reference[pdf <= shape_pdf_reference] = None

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Shape Reference",
            "x": grid,
            "y": shape_pdf_reference,
            "line": {"width": line_width, "color": "#9017e6"},
        }
    )

    location_pdf_reference = context_dict["location_pdf_reference"]

    if location_pdf_reference is not None:

        location_pdf_reference[pdf <= location_pdf_reference] = None

        data.append(
            {
                "yaxis": "y2",
                "type": "scatter",
                "name": "Location Reference",
                "x": grid,
                "y": location_pdf_reference,
                "line": {"width": line_width, "color": "#4e40d8"},
            }
        )

    is_negative = context_dict["context"] < 0

    is_positive = 0 < context_dict["context"]

    for name, indices, color in (
        ("- Context", is_negative, "#0088ff"),
        ("+ Context", is_positive, "#ff1968"),
    ):

        data.append(
            {
                "yaxis": "y2",
                "type": "scatter",
                "name": name,
                "x": grid[indices],
                "y": absolute_context[indices],
                "line": {"width": line_width, "color": color},
                "fill": "tozeroy",
            }
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
