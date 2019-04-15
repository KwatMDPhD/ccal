from matplotlib.colors import LinearSegmentedColormap, to_hex
from numpy import asarray, cos, isnan, linspace, nan, pi, sin, unique, where
from pandas import Series, isna

from .clip_nd_array_by_standard_deviation import clip_nd_array_by_standard_deviation
from .get_colormap_colors import get_colormap_colors
from .get_triangulation_edges_from_point_x_dimension import (
    get_triangulation_edges_from_point_x_dimension,
)
from .make_colorscale_from_colors import make_colorscale_from_colors
from .match_colors_to_data import match_colors_to_data
from .normalize_nd_array import normalize_nd_array
from .plot_and_save import plot_and_save


def plot_gps_map(
    nodes,
    node_name,
    node_x_dimension,
    elements,
    element_name,
    element_x_dimension,
    element_marker_size,
    element_labels,
    grid_values,
    grid_labels,
    label_colors,
    grid_label_opacity,
    annotation_x_element,
    annotation_types,
    annotation_std_maxs,
    annotation_ranges,
    annotation_colors,
    layout_size,
    title,
    html_file_path,
):

    axis_template = {
        "showgrid": False,
        "zeroline": False,
        "ticks": "",
        "showticklabels": False,
    }

    layout = {
        "width": layout_size,
        "height": layout_size,
        "title": title,
        "titlefont": {"size": 32, "color": "#4c221b"},
        "xaxis": axis_template,
        "yaxis": axis_template,
    }

    data = []

    node_opacity = 0.88

    edge_xs, edge_ys = get_triangulation_edges_from_point_x_dimension(node_x_dimension)

    data.append(
        {
            "type": "scatter",
            "name": "Simplex",
            "legendgroup": node_name,
            "showlegend": False,
            "x": edge_xs,
            "y": edge_ys,
            "mode": "lines",
            "line": {"color": "#23191e"},
            "opacity": node_opacity,
            "hoverinfo": None,
        }
    )

    data.append(
        {
            "type": "scatter",
            "name": node_name,
            "legendgroup": node_name,
            "x": node_x_dimension[:, 0],
            "y": node_x_dimension[:, 1],
            "text": nodes,
            "mode": "markers+text",
            "marker": {
                "size": 32,
                "color": "#2e211b",
                "line": {"width": 2.4, "color": "#ffffff"},
            },
            "textposition": "middle center",
            "textfont": {"size": 12, "color": "#ebf6f7"},
            "opacity": node_opacity,
            "hoverinfo": "text",
        }
    )

    if element_labels is not None:

        x = linspace(0, 1, grid_values.shape[1])

        y = linspace(0, 1, grid_values.shape[0])

        data.append(
            {
                "type": "contour",
                "name": "Contour",
                "showlegend": False,
                "z": grid_values[::-1],
                "x": x,
                "y": y,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
                "showscale": False,
                "opacity": node_opacity,
            }
        )

        grid_labels_unique = unique(grid_labels[~isnan(grid_labels)]).astype(int)

        for label in grid_labels_unique:

            z = grid_values.copy()

            z[grid_labels != label] = nan

            data.append(
                {
                    "type": "heatmap",
                    "z": z[::-1],
                    "x": x,
                    "y": y,
                    "colorscale": make_colorscale_from_colors(
                        get_colormap_colors(
                            LinearSegmentedColormap.from_list(
                                None, ("#ffffff", label_colors[label])
                            )
                        )
                    ),
                    "showscale": False,
                    "opacity": grid_label_opacity,
                }
            )

    element_marker_line = {"width": 1, "color": "#000000"}

    element_opacity = 0.88

    if annotation_x_element is not None:

        if 1 < annotation_x_element.shape[0]:

            shapes = []

            marker_size_factor = element_marker_size / layout_size

        for annotation_index, (annotation_name, annotation_series) in enumerate(
            annotation_x_element.iterrows()
        ):

            if annotation_types is None:

                annotation_type = "continuous"

            else:

                annotation_type = annotation_types[annotation_index]

            if annotation_type == "continuous":

                if annotation_std_maxs is not None:

                    std_max = annotation_std_maxs[annotation_index]

                    annotation_series = Series(
                        clip_nd_array_by_standard_deviation(
                            normalize_nd_array(
                                annotation_series.values,
                                None,
                                "-0-",
                                raise_for_bad=False,
                            ),
                            std_max,
                            raise_for_bad=False,
                        ),
                        name=annotation_series.name,
                        index=annotation_series.index,
                    )

                if annotation_ranges is not None:

                    min_, max_ = annotation_ranges[annotation_index]

                else:

                    min_ = annotation_series.min()

                    max_ = annotation_series.max()

            elif annotation_type == "categorical":

                min_, max_ = 0, annotation_series.dropna().unique().size - 1

            elif annotation_type == "binary":

                min_, max_ = 0, 1

            if annotation_x_element.shape[0] == 1:

                sorted_indices = [
                    annotation_series.index.tolist().index(i)
                    for i in annotation_series.abs().sort_values().index
                ]

                annotation_series = annotation_series[sorted_indices]

                element_x_dimension = element_x_dimension[sorted_indices]

            colorscale = make_colorscale_from_colors(
                match_colors_to_data(annotation_series.values, annotation_type)
            )

            if 1 == annotation_x_element.shape[0]:

                is_na = annotation_series.isna()

                annotation_series = annotation_series[~is_na]

                data.append(
                    {
                        "type": "scatter",
                        "name": element_name,
                        "showlegend": False,
                        "x": element_x_dimension[~is_na, 0],
                        "y": element_x_dimension[~is_na, 1],
                        "text": annotation_series.index,
                        "mode": "markers",
                        "marker": {
                            "size": element_marker_size,
                            "color": annotation_series.values,
                            "cmin": min_,
                            "cmax": max_,
                            "colorscale": colorscale,
                            "showscale": annotation_type == "continuous",
                            "colorbar": {"len": 0.64, "thickness": layout_size / 80},
                            "line": element_marker_line,
                        },
                        "opacity": element_opacity,
                        "hoverinfo": "text",
                    }
                )

            else:

                for element_name, value in annotation_series.items():

                    x, y = element_x_dimension[elements.index(element_name)]

                    if isna(value):

                        continue

                    color = to_hex(
                        colorscale[
                            int(len(colorscale) * (value - min_) / (max_ - min_))
                        ][1]
                    )

                    sector_radian = pi * 2 / annotation_x_element.shape[0]

                    sector_radians = linspace(
                        sector_radian * annotation_index,
                        sector_radian * (annotation_index + 1),
                        16,
                    )

                    path = "M {} {}".format(x, y)

                    for x_, y_ in zip(cos(sector_radians), sin(sector_radians)):

                        path += " L {} {}".format(
                            x + x_ * marker_size_factor, y + y_ * marker_size_factor
                        )

                    path += " Z"

                    shapes.append(
                        {
                            "type": "path",
                            "x0": x,
                            "y0": y,
                            "path": path,
                            "fillcolor": color,
                            "line": element_marker_line,
                            "opacity": element_opacity,
                        }
                    )

        if 1 < annotation_x_element.shape[0]:

            layout["shapes"] = shapes

    elif element_labels is not None:

        for label in grid_labels_unique:

            element_indices = where(element_labels == label)

            label_str = str(label)

            data.append(
                {
                    "type": "scatter",
                    "name": label_str,
                    "legendgroup": label_str,
                    "x": element_x_dimension[element_indices, 0][0],
                    "y": element_x_dimension[element_indices, 1][0],
                    "text": asarray(elements)[element_indices],
                    "mode": "markers",
                    "marker": {
                        "size": element_marker_size,
                        "color": label_colors[label],
                        "line": element_marker_line,
                    },
                    "opacity": element_opacity,
                    "hoverinfo": "text",
                }
            )

    else:

        data.append(
            {
                "type": "scatter",
                "name": element_name,
                "x": element_x_dimension[:, 0],
                "y": element_x_dimension[:, 1],
                "text": elements,
                "mode": "markers",
                "marker": {
                    "size": element_marker_size,
                    "color": "#20d9ba",
                    "line": element_marker_line,
                },
                "opacity": element_opacity,
                "hoverinfo": "text",
            }
        )

    plot_and_save({"layout": layout, "data": data}, html_file_path)
