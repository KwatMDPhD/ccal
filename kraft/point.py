from numpy import apply_along_axis, full, isnan, nan, unique, where
from pandas import DataFrame
from sklearn.manifold import MDS

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .geometry import get_convex_hull, get_triangulation
from .plot import COLORBAR, get_color, make_colorscale, plot_plotly
from .support import merge_2_dicts


def pull_point(node_x_dimension, point_x_node):

    point_x_dimension = full((point_x_node.shape[0], node_x_dimension.shape[1]), nan)

    for point_index in range(point_x_node.shape[0]):

        pulls = point_x_node[point_index, :]

        for dimension_index in range(node_x_dimension.shape[1]):

            point_x_dimension[point_index, dimension_index] = (
                pulls * node_x_dimension[:, dimension_index]
            ).sum() / pulls.sum()

    return point_x_dimension


def map_point(
    point_x_point_distance,
    n_dimension,
    metric=True,
    n_init=int(1e3),
    max_iter=int(1e3),
    verbose=0,
    eps=1e-3,
    n_job=1,
    random_seed=RANDOM_SEED,
):

    point_x_dimension = MDS(
        n_components=n_dimension,
        metric=metric,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        n_jobs=n_job,
        random_state=random_seed,
        dissimilarity="precomputed",
    ).fit_transform(point_x_point_distance)

    return apply_along_axis(normalize, 0, point_x_dimension, "0-1")


def plot_node_point(
    node_x_dimension,
    point_x_dimension,
    show_node_text=True,
    point_trace=None,
    point_group=None,
    group_colorscale=None,
    grid=None,
    # TODO: rename
    grid_point_probability=None,
    # TODO: rename
    grid_point_group=None,
    point_value=None,
    value_colorscale=None,
    value_opacity=0.8,
    value_na_opacity=0.2,
    points_to_highlight=(),
    html_file_path=None,
):

    node_x_y = DataFrame(
        {
            "x": node_x_dimension.iloc[:, 1].values,
            "y": 1 - node_x_dimension.iloc[:, 0].values,
        },
        index=node_x_dimension.index,
    )

    point_x_y = DataFrame(
        {
            "x": point_x_dimension.iloc[:, 1].values,
            "y": 1 - point_x_dimension.iloc[:, 0].values,
        },
        index=point_x_dimension.index,
    )

    title_text = "{} {} and {} {}".format(
        node_x_y.index.size,
        node_x_y.index.name,
        point_x_y.index.size,
        point_x_y.index.name,
    )

    if point_value is not None:

        title_text = "{}<br>{}".format(title_text, point_value.name)

    axis = {"showgrid": False, "zeroline": False, "showticklabels": False}

    layout = {
        "height": 880,
        "width": 880,
        "title": {
            "x": 0.5,
            "text": "<b>{}</b>".format(title_text),
            "font": {
                "size": 24,
                "color": "#4e40d8",
                "family": "Times New Roman, sans-serif",
            },
        },
        "xaxis": axis,
        "yaxis": axis,
        "annotations": [],
    }

    data = []

    triangulation_xs, triangulation_ys = get_triangulation(node_x_y)

    convex_hull_xs, convex_hull_ys = get_convex_hull(node_x_y)

    data.append(
        {
            "name": "Line",
            "x": triangulation_xs + convex_hull_xs,
            "y": triangulation_ys + convex_hull_ys,
            "mode": "lines",
            "line": {"color": "#171412"},
        }
    )

    data.append(
        {
            "name": node_x_y.index.name,
            "x": node_x_y["x"],
            "y": node_x_y["y"],
            "text": node_x_y.index,
            "mode": "markers",
            "marker": {
                "size": 20,
                "color": "#23191e",
                "line": {"width": 1, "color": "#ebf6f7"},
            },
            "hoverinfo": "text",
        }
    )

    if show_node_text:

        arrowwidth = 1.6

        arrowcolor = "#ebf6f7"

        layout["annotations"] += [
            {
                "x": x,
                "y": y,
                "text": "<b>{}</b>".format(node),
                "font": {
                    "size": 16,
                    "color": "#23191e",
                    "family": "Gravitas One, monospace",
                },
                "borderpad": 2,
                "borderwidth": arrowwidth,
                "bordercolor": arrowcolor,
                "bgcolor": "#ffffff",
                "arrowwidth": arrowwidth,
                "arrowcolor": arrowcolor,
            }
            for node, (x, y) in node_x_y.iterrows()
        ]

    if grid_point_group is not None:

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": grid,
                "y": 1 - grid,
                "z": grid_point_probability,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
            }
        )

        unique_groups = unique(grid_point_group[~isnan(grid_point_group)])

        group_color = {
            group: get_color(group_colorscale, group, unique_groups.size)
            for group in unique_groups
        }

        for group in unique_groups:

            z = grid_point_probability.copy()

            z[grid_point_group != group] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": grid,
                    "y": 1 - grid,
                    "z": z,
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", group_color[group])
                    ),
                    "showscale": False,
                    "hoverinfo": "none",
                }
            )

    point_trace_template = {
        "name": point_x_y.index.name,
        "mode": "markers",
        "marker": {
            "size": 16,
            "color": "#20d9ba",
            "line": {"width": 0.8, "color": "#ebf6f7"},
        },
        "hoverinfo": "text",
    }

    if point_trace is not None:

        point_trace_template = merge_2_dicts(point_trace_template, point_trace)

    if point_value is not None:

        point_value = point_value.reindex(index=point_x_y.index)

        point_value = point_value[
            point_value.abs().sort_values(na_position="first").index
        ]

        if value_na_opacity == 0:

            point_value.dropna(inplace=True)

        else:

            value_opacity = where(point_value.isna(), value_na_opacity, value_opacity)

        point_x_y = point_x_y.loc[point_value.index]

        if point_value.astype(float).map(float.is_integer).all():

            tickvals = point_value.unique()

            ticktext_format = "{:.0f}".format

        else:

            tickvals = point_value.describe()[
                ["min", "25%", "50%", "mean", "75%", "max"]
            ].values

            ticktext_format = "{:.2e}".format

        data.append(
            merge_2_dicts(
                point_trace_template,
                {
                    "x": point_x_y["x"],
                    "y": point_x_y["y"],
                    "text": tuple(
                        "{}<br>{:.2e}".format(point, value)
                        for point, value in point_value.items()
                    ),
                    "marker": {
                        "color": point_value,
                        "colorscale": value_colorscale,
                        "colorbar": {
                            **COLORBAR,
                            "tickmode": "array",
                            "tickvals": tickvals,
                            "ticktext": tuple(
                                ticktext_format(tickval) for tickval in tickvals
                            ),
                        },
                        "opacity": value_opacity,
                    },
                },
            )
        )

    elif point_group is not None:

        for group in unique_groups:

            name = "Group {:.0f}".format(group)

            is_group = point_group == group

            data.append(
                merge_2_dicts(
                    point_trace_template,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_y["x"][is_group],
                        "y": point_x_y["y"][is_group],
                        "text": point_x_y.index[is_group],
                        "marker": {"color": group_color[group]},
                    },
                )
            )
    else:

        data.append(
            {
                **point_trace_template,
                "x": point_x_y["x"],
                "y": point_x_y["y"],
                "text": point_x_y.index,
            }
        )

    layout["annotations"] += [
        {
            "x": point_x_y.loc[point_name, "x"],
            "y": point_x_y.loc[point_name, "y"],
            "text": "<b>{}</b>".format(point_name),
            "arrowhead": 2,
            "arrowwidth": 2,
            "arrowcolor": "#c93756",
            "standoff": None,
            "clicktoshow": "onoff",
        }
        for point_name in points_to_highlight
    ]

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
