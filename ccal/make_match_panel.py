from pandas import DataFrame, Series

from .cluster_2d_array import cluster_2d_array
from .compute_information_coefficient_between_2_1d_arrays import (
    compute_information_coefficient_between_2_1d_arrays,
)
from .is_sorted_nd_array import is_sorted_nd_array
from .make_colorscale_from_colors import make_colorscale_from_colors
from .make_match_panel_annotations import make_match_panel_annotations
from .match_target_and_data_and_compute_statistics import (
    match_target_and_data_and_compute_statistics,
)
from .normalize_nd_array import normalize_nd_array
from .pick_nd_array_colors import pick_nd_array_colors
from .plot_and_save import plot_and_save
from .RANDOM_SEED import RANDOM_SEED
from .select_series_indices import select_series_indices


def make_match_panel(
    target,
    data,
    target_ascending=True,
    score_moe_p_value_fdr=None,
    n_job=1,
    match_function=compute_information_coefficient_between_2_1d_arrays,
    n_required_for_match_function=2,
    raise_for_n_less_than_required=False,
    n_extreme=8,
    fraction_extreme=None,
    random_seed=RANDOM_SEED,
    n_sampling=0,
    n_permutation=0,
    score_ascending=False,
    plot=True,
    plot_only_sign=None,
    target_type="continuous",
    cluster_within_category=True,
    data_type="continuous",
    plot_std=None,
    title=None,
    layout_width=880,
    row_height=64,
    layout_side_margin=196,
    annotation_font_size=8.8,
    file_path_prefix=None,
):

    common_indices = target.index & data.columns

    print(
        "target.index ({}) & data.columns ({}) have {} in common.".format(
            target.index.size, data.columns.size, len(common_indices)
        )
    )

    target = target[common_indices].dropna()

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    data = data[target.index]

    if score_moe_p_value_fdr is None:

        score_moe_p_value_fdr = match_target_and_data_and_compute_statistics(
            target.values,
            data.values,
            n_job,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
            n_extreme,
            fraction_extreme,
            random_seed,
            n_sampling,
            n_permutation,
        )

        score_moe_p_value_fdr.index = data.index

    else:

        score_moe_p_value_fdr = score_moe_p_value_fdr.reindex(index=data.index)

    if score_moe_p_value_fdr.isna().values.all():

        print("score_moe_p_value_fdr has only na.")

        return score_moe_p_value_fdr

    score_moe_p_value_fdr.sort_values("Score", ascending=score_ascending, inplace=True)

    if file_path_prefix is not None:

        score_moe_p_value_fdr.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    if not plot:

        return score_moe_p_value_fdr

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.statistics.html".format(file_path_prefix)

    if score_moe_p_value_fdr.shape[0] < 1e3:

        mode = "lines+markers"

    else:

        mode = "lines"

    plot_and_save(
        {
            "layout": {
                "title": {"text": "Statistics"},
                "xaxis": {"title": "Rank"},
                "yaxis": {"title": "Score<br>{}".format(match_function.__name__)},
            },
            "data": [
                {
                    "type": "scatter",
                    "name": name,
                    "x": score_moe_p_value_fdr.index,
                    "y": score_moe_p_value_fdr[name].values,
                    "mode": mode,
                    "marker": {"color": color},
                }
                for name, color in zip(
                    score_moe_p_value_fdr.columns,
                    ("#20d9ba", "#ff1968", "#4e40d8", "#9017e6"),
                )
            ],
        },
        html_file_path,
    )

    scores_to_plot = score_moe_p_value_fdr.copy()

    if n_extreme is not None or fraction_extreme is not None:

        scores_to_plot = scores_to_plot.loc[
            select_series_indices(
                scores_to_plot["Score"],
                "<>",
                n=n_extreme,
                fraction=fraction_extreme,
                plot=False,
            )
        ]

    if plot_only_sign is not None:

        if plot_only_sign == "-":

            scores_to_plot = scores_to_plot.loc[scores_to_plot["Score"] < 0]

        elif plot_only_sign == "+":

            scores_to_plot = scores_to_plot.loc[0 < scores_to_plot["Score"]]

    data_to_plot = data.loc[scores_to_plot.index]

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

    if (
        cluster_within_category
        and target_type in ("binary", "categorical")
        and is_sorted_nd_array(target_to_plot.values)
        and (1 < target_to_plot.value_counts()).all()
        and not data_to_plot.isna().values.all()
    ):

        print("Clustering within category ...")

        data_to_plot = data_to_plot.iloc[
            :,
            cluster_2d_array(
                data_to_plot.values,
                1,
                groups=target_to_plot.values,
                raise_for_bad=False,
            ),
        ]

        target_to_plot = target_to_plot[data_to_plot.columns]

    if data_type == "continuous":

        data_to_plot = DataFrame(
            normalize_nd_array(data_to_plot.values, 1, "-0-", raise_for_bad=False),
            index=data_to_plot.index,
            columns=data_to_plot.columns,
        )

        if plot_std is not None:

            data_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

    data_colorscale = make_colorscale_from_colors(
        pick_nd_array_colors(data_to_plot.values, data_type)
    )

    target_row_fraction = max(0.01, 1 / (data_to_plot.shape[0] + 2))

    target_yaxis_domain = (1 - target_row_fraction, 1)

    data_yaxis_domain = (0, 1 - target_row_fraction * 2)

    data_row_fraction = (
        data_yaxis_domain[1] - data_yaxis_domain[0]
    ) / data_to_plot.shape[0]

    annotation_font = {"size": annotation_font_size}

    layout = {
        "width": layout_width,
        "height": row_height * max(8, (data_to_plot.shape[0] + 2) ** 0.8),
        "margin": {"l": layout_side_margin, "r": layout_side_margin},
        "title": {"text": title},
        "xaxis": {"anchor": "y", "tickfont": {"size": annotation_font_size}},
        "yaxis": {"domain": data_yaxis_domain, "dtick": 1, "tickfont": annotation_font},
        "yaxis2": {
            "domain": target_yaxis_domain,
            "tickmode": "array",
            "tickvals": (0,),
            "ticktext": (target.name,),
            "tickfont": annotation_font,
        },
        "annotations": [],
    }

    figure_data = [
        {
            "yaxis": "y2",
            "type": "heatmap",
            "z": target_to_plot.to_frame().T.values,
            "colorscale": target_colorscale,
            "showscale": False,
        },
        {
            "yaxis": "y",
            "type": "heatmap",
            "z": data_to_plot.values[::-1],
            "x": data_to_plot.columns,
            "y": data_to_plot.index[::-1],
            "colorscale": data_colorscale,
            "showscale": False,
        },
    ]

    layout_annotation_template = {
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "middle",
        "font": annotation_font,
        "width": 64,
        "showarrow": False,
    }

    for i, (annotation, strs) in enumerate(
        make_match_panel_annotations(scores_to_plot).items()
    ):

        x = 1.0016 + i / 10

        layout["annotations"].append(
            {
                "x": x,
                "y": target_yaxis_domain[1] - (target_row_fraction / 2),
                "text": "<b>{}</b>".format(annotation),
                **layout_annotation_template,
            }
        )

        y = data_yaxis_domain[1] - (data_row_fraction / 2)

        for str_ in strs:

            layout["annotations"].append(
                {
                    "x": x,
                    "y": y,
                    "text": "<b>{}</b>".format(str_),
                    **layout_annotation_template,
                }
            )

            y -= data_row_fraction

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.html".format(file_path_prefix)

    plot_and_save({"layout": layout, "data": figure_data}, html_file_path)

    return score_moe_p_value_fdr
