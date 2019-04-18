from pandas import DataFrame, Series

from .ALMOST_ZERO import ALMOST_ZERO
from .make_colorscale_from_colors import make_colorscale_from_colors
from .make_match_panel_annotations import make_match_panel_annotations
from .normalize_nd_array import normalize_nd_array
from .pick_nd_array_colors import pick_nd_array_colors
from .plot_and_save import plot_and_save


def make_summary_match_panel(
    target,
    data_dicts,
    score_moe_p_value_fdr_dicts,
    plot_only_columns_shared_by_target_and_all_data=False,
    target_ascending=True,
    target_type="continuous",
    score_ascending=False,
    plot_std=None,
    title=None,
    layout_side_margin=196,
    annotation_font_size=8.8,
    xaxis_kwargs=None,
    html_file_path=None,
):

    if plot_only_columns_shared_by_target_and_all_data:

        for data_dict in data_dicts.values():

            target = target.loc[target.index & data_dict["df"].columns]

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    target_to_plot = target.copy()

    if target_type == "continuous":

        target_to_plot = Series(
            normalize_nd_array(target_to_plot.values, None, "-0-", raise_for_bad=False),
            name=target_to_plot.name,
            index=target_to_plot.index,
        )

        if plot_std is not None:

            target_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

    target_colorscale = make_colorscale_from_colors(
        pick_nd_array_colors(target_to_plot.values, target_type)
    )

    n_row = 1 + len(data_dicts)

    for data_dict in data_dicts.values():

        n_row += data_dict["df"].shape[0]

    layout = {
        "height": max(640, 32 * n_row),
        "margin": {"l": layout_side_margin, "r": layout_side_margin},
        "title": {"text": title},
        "xaxis": {"anchor": "y"},
        "annotations": [],
    }

    row_fraction = 1 / n_row

    yaxis_name = "yaxis{}".format(len(data_dicts) + 1)

    domain_end = 1

    domain_start = domain_end - row_fraction

    if abs(domain_start) <= ALMOST_ZERO:

        domain_start = 0

    annotation_font = {"size": annotation_font_size}

    layout[yaxis_name] = {
        "domain": (domain_start, domain_end),
        "tickmode": "array",
        "tickvals": (0,),
        "ticktext": (target_to_plot.name,),
        "tickfont": annotation_font,
    }

    figure_data = [
        {
            "yaxis": yaxis_name.replace("axis", ""),
            "type": "heatmap",
            "z": target_to_plot.to_frame().T,
            "colorscale": target_colorscale,
            "showscale": False,
        }
    ]

    for data_index, (data_name, data_dict) in enumerate(data_dicts.items()):

        print("Making match panel for {} ...".format(data_name))

        data_to_plot = data_dict["df"].reindex(columns=target_to_plot.index)

        score_moe_p_value_fdr_to_plot = score_moe_p_value_fdr_dicts[data_name].loc[
            data_to_plot.index
        ]

        score_moe_p_value_fdr_to_plot.sort_values(
            "Score", ascending=score_ascending, inplace=True
        )

        data_to_plot = data_to_plot.loc[score_moe_p_value_fdr_to_plot.index]

        if data_dict["type"] == "continuous":

            data_to_plot = DataFrame(
                normalize_nd_array(data_to_plot.values, 1, "-0-", raise_for_bad=False),
                index=data_to_plot.index,
                columns=data_to_plot.columns,
            )

            if plot_std is not None:

                data_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

        data_colorscale = make_colorscale_from_colors(
            pick_nd_array_colors(data_to_plot.values, data_dict["type"])
        )

        yaxis_name = "yaxis{}".format(len(data_dicts) - data_index)

        domain_end = domain_start - row_fraction

        if abs(domain_end) <= ALMOST_ZERO:

            domain_end = 0

        domain_start = domain_end - data_to_plot.shape[0] * row_fraction

        if abs(domain_start) <= ALMOST_ZERO:

            domain_start = 0

        layout[yaxis_name] = {
            "domain": (domain_start, domain_end),
            "dtick": 1,
            "tickfont": annotation_font,
        }

        figure_data.append(
            {
                "yaxis": yaxis_name.replace("axis", ""),
                "type": "heatmap",
                "z": data_to_plot.values[::-1],
                "x": data_to_plot.columns,
                "y": data_to_plot.index[::-1],
                "colorscale": data_colorscale,
                "showscale": False,
            }
        )

        layout_annotation_template = {
            "xref": "paper",
            "yref": "paper",
            "yanchor": "middle",
            "font": annotation_font,
            "showarrow": False,
        }

        layout["annotations"].append(
            {
                "xanchor": "center",
                "x": 0.5,
                "y": domain_end + (row_fraction / 2),
                "text": "<b>{}</b>".format(data_name),
                **layout_annotation_template,
            }
        )

        layout_annotation_template.update({"xanchor": "left", "width": 64})

        for i, (annotation, strs) in enumerate(
            make_match_panel_annotations(score_moe_p_value_fdr_to_plot).items()
        ):

            x = 1.0016 + i / 10

            if data_index == 0:

                layout["annotations"].append(
                    {
                        "x": x,
                        "y": 1 - (row_fraction / 2),
                        "text": "<b>{}</b>".format(annotation),
                        **layout_annotation_template,
                    }
                )

            y = domain_end - (row_fraction / 2)

            for str_ in strs:

                layout["annotations"].append(
                    {
                        "x": x,
                        "y": y,
                        "text": "<b>{}</b>".format(str_),
                        **layout_annotation_template,
                    }
                )

                y -= row_fraction

    plot_and_save({"layout": layout, "data": figure_data}, html_file_path)
