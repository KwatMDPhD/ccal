from ._make_annotations import _make_annotations
from ._match import _match
from ._process_target_or_features_for_plotting import (
    _process_target_or_features_for_plotting,
)
from ._style import (
    ANNOTATION_FONT_SIZE,
    ANNOTATION_WIDTH,
    LAYOUT_SIDE_MARGIN,
    LAYOUT_WIDTH,
    ROW_HEIGHT,
)
from .cluster_2d_array_slices import cluster_2d_array_slices
from .compute_information_coefficient import compute_information_coefficient
from .iterable import make_object_int_mapping
from .nd_array_is_sorted import nd_array_is_sorted
from .plot_and_save import plot_and_save
from .series import get_extreme_series_indices


def make_match_panel(
    target,
    features,
    target_ascending=True,
    score_moe_p_value_fdr=None,
    n_job=1,
    match_function=compute_information_coefficient,
    n_required_for_match_function=2,
    raise_for_n_less_than_required=False,
    extreme_feature_threshold=8,
    random_seed=20_121_020,
    n_sampling=0,
    n_permutation=0,
    score_ascending=False,
    plot_only_sign=None,
    target_type="continuous",
    cluster_within_category=True,
    features_type="continuous",
    plot_std=None,
    title="Match Panel",
    layout_width=LAYOUT_WIDTH,
    row_height=ROW_HEIGHT,
    layout_side_margin=LAYOUT_SIDE_MARGIN,
    annotation_font_size=ANNOTATION_FONT_SIZE,
    file_path_prefix=None,
    plotly_html_file_path_prefix=None,
):

    if target.name is None:

        target_name = "Target"

    else:

        target_name = target.name

    common_indices = target.index & features.columns

    print(
        "target.index ({}) & features.columns ({}) have {} in common.".format(
            target.index.size, features.columns.size, len(common_indices)
        )
    )

    target = target[common_indices]

    if target.dtype == "O":

        target = target.map(make_object_int_mapping(target)[0])

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    features = features[target.index]

    if score_moe_p_value_fdr is None:

        score_moe_p_value_fdr = _match(
            target.values,
            features.values,
            n_job,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
            extreme_feature_threshold,
            random_seed,
            n_sampling,
            n_permutation,
        )

        if score_moe_p_value_fdr.isna().values.all():

            return score_moe_p_value_fdr

        score_moe_p_value_fdr.index = features.index

        score_moe_p_value_fdr.sort_values(
            "Score", ascending=score_ascending, inplace=True
        )

        if file_path_prefix is not None:

            score_moe_p_value_fdr.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    else:

        score_moe_p_value_fdr = score_moe_p_value_fdr.reindex(index=features.index)

    indices = get_extreme_series_indices(
        score_moe_p_value_fdr["Score"],
        extreme_feature_threshold,
        ascending=score_ascending,
    )

    scores_to_plot = score_moe_p_value_fdr.loc[indices]

    features_to_plot = features.loc[scores_to_plot.index]

    if plot_only_sign is not None:

        if plot_only_sign == "-":

            indices = scores_to_plot["Score"] <= 0

        elif plot_only_sign == "+":

            indices = 0 <= scores_to_plot["Score"]

        scores_to_plot = scores_to_plot.loc[indices]

        features_to_plot = features_to_plot.loc[scores_to_plot.index]

    annotations = _make_annotations(scores_to_plot.dropna(axis=1, how="all"))

    target, target_plot_min, target_plot_max, target_colorscale = _process_target_or_features_for_plotting(
        target, target_type, plot_std
    )

    if (
        cluster_within_category
        and target_type in ("binary", "categorical")
        and 1 < target.value_counts().min()
        and nd_array_is_sorted(target.values)
        and not features_to_plot.isna().all().any()
    ):

        print("Clustering heat map within category ...")

        clustered_indices = cluster_2d_array_slices(
            features_to_plot.values, 1, groups=target.values, raise_for_bad=False
        )

        target = target.iloc[clustered_indices]

        features_to_plot = features_to_plot.iloc[:, clustered_indices]

    features_to_plot, features_plot_min, features_plot_max, features_colorscale = _process_target_or_features_for_plotting(
        features_to_plot, features_type, plot_std
    )

    target_row_fraction = max(0.01, 1 / (features_to_plot.shape[0] + 2))

    target_yaxis_domain = (1 - target_row_fraction, 1)

    features_yaxis_domain = (0, 1 - target_row_fraction * 2)

    feature_row_fraction = (
        features_yaxis_domain[1] - features_yaxis_domain[0]
    ) / features_to_plot.shape[0]

    layout = dict(
        width=layout_width,
        height=row_height * max(8, (features_to_plot.shape[0] + 2) ** 0.8),
        margin=dict(l=layout_side_margin, r=layout_side_margin),
        xaxis=dict(anchor="y", tickfont=dict(size=annotation_font_size)),
        yaxis=dict(
            domain=features_yaxis_domain,
            dtick=1,
            tickfont=dict(size=annotation_font_size),
        ),
        yaxis2=dict(
            domain=target_yaxis_domain, tickfont=dict(size=annotation_font_size)
        ),
        title=title,
        annotations=[],
    )

    data = [
        dict(
            yaxis="y2",
            type="heatmap",
            z=target.to_frame().T.values,
            x=target.index,
            y=(target_name,),
            text=(target.index,),
            zmin=target_plot_min,
            zmax=target_plot_max,
            colorscale=target_colorscale,
            showscale=False,
        ),
        dict(
            yaxis="y",
            type="heatmap",
            z=features_to_plot.values[::-1],
            x=features_to_plot.columns,
            y=features_to_plot.index[::-1],
            zmin=features_plot_min,
            zmax=features_plot_max,
            colorscale=features_colorscale,
            showscale=False,
        ),
    ]

    layout_annotation_template = dict(
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="middle",
        font=dict(size=annotation_font_size),
        width=ANNOTATION_WIDTH,
        showarrow=False,
    )

    for annotation_index, (annotation, strs) in enumerate(annotations.items()):

        x = 1.0016 + annotation_index / 10

        layout["annotations"].append(
            dict(
                x=x,
                y=target_yaxis_domain[1] - (target_row_fraction / 2),
                text="<b>{}</b>".format(annotation),
                **layout_annotation_template,
            )
        )

        y = features_yaxis_domain[1] - (feature_row_fraction / 2)

        for str_ in strs:

            layout["annotations"].append(
                dict(
                    x=x,
                    y=y,
                    text="<b>{}</b>".format(str_),
                    **layout_annotation_template,
                )
            )

            y -= feature_row_fraction

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.html".format(file_path_prefix)

    if plotly_html_file_path_prefix is None:

        plotly_html_file_path = None

    else:

        plotly_html_file_path = "{}.html".format(plotly_html_file_path_prefix)

    plot_and_save(
        dict(layout=layout, data=data),
        html_file_path=html_file_path,
        plotly_html_file_path=plotly_html_file_path,
    )

    return score_moe_p_value_fdr
