from numpy import finfo

from ._make_annotations import _make_annotations
from ._process_target_or_features_for_plotting import (
    _process_target_or_features_for_plotting,
)
from .ANNOTATION_FONT_SIZE import ANNOTATION_FONT_SIZE
from .ANNOTATION_WIDTH import ANNOTATION_WIDTH
from .LAYOUT_SIDE_MARGIN import LAYOUT_SIDE_MARGIN
from .LAYOUT_WIDTH import LAYOUT_WIDTH
from .make_object_int_mapping import make_object_int_mapping
from .plot_and_save import plot_and_save
from .ROW_HEIGHT import ROW_HEIGHT

eps = finfo(float).eps


def make_summary_match_panel(
    target,
    data_dicts,
    score_moe_p_value_fdr,
    plot_only_columns_shared_by_target_and_all_features=False,
    target_ascending=True,
    target_type="continuous",
    plot_std=None,
    title="Summary Match Panel",
    layout_width=LAYOUT_WIDTH,
    row_height=ROW_HEIGHT,
    layout_side_margin=LAYOUT_SIDE_MARGIN,
    annotation_font_size=ANNOTATION_FONT_SIZE,
    html_file_path=None,
    plotly_html_file_path=None,
):

    if target.name is None:

        target_name = "Target"

    else:

        target_name = target.name

    if plot_only_columns_shared_by_target_and_all_features:

        for data_dict in data_dicts.values():

            target = target.loc[target.index & data_dict["df"].columns]

    if target.dtype == "O":

        target = target.map(make_object_int_mapping(target)[0])

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    target, target_plot_min, target_plot_max, target_colorscale = _process_target_or_features_for_plotting(
        target, target_type, plot_std
    )

    n_row = 1 + len(data_dicts)

    for data_dict in data_dicts.values():

        n_row += data_dict["df"].shape[0]

    layout = dict(
        width=layout_width,
        margin=dict(l=layout_side_margin, r=layout_side_margin),
        xaxis=dict(anchor="y"),
        height=row_height / 2 * max(10, n_row),
        title=title,
        annotations=[],
    )

    row_fraction = 1 / n_row

    yaxis_name = "yaxis{}".format(len(data_dicts) + 1).replace("axis1", "axis")

    domain_end = 1

    domain_start = domain_end - row_fraction

    if abs(domain_start) <= eps:

        domain_start = 0

    layout[yaxis_name] = dict(
        domain=(domain_start, domain_end), tickfont=dict(size=annotation_font_size)
    )

    data = [
        dict(
            yaxis=yaxis_name.replace("axis", ""),
            type="heatmap",
            z=target.to_frame().T.values,
            x=target.index,
            y=(target_name,),
            text=(target.index,),
            zmin=target_plot_min,
            zmax=target_plot_max,
            colorscale=target_colorscale,
            showscale=False,
        )
    ]

    for data_name_index, (data_name, data_dict) in enumerate(data_dicts.items()):

        print("Making match panel for {} ...".format(data_name))

        df = data_dict["df"]

        features_to_plot = df[df.columns & target.index]

        score_moe_p_value_fdr_to_plot = score_moe_p_value_fdr.loc[
            features_to_plot.index
        ].sort_values("Score", ascending=data_dict.get("emphasis", "high") == "low")

        features_to_plot = features_to_plot.loc[score_moe_p_value_fdr_to_plot.index]

        annotations = _make_annotations(
            score_moe_p_value_fdr_to_plot.dropna(axis=1, how="all")
        )

        features_to_plot, features_plot_min, features_plot_max, features_colorscale = _process_target_or_features_for_plotting(
            features_to_plot, data_dict["data_type"], plot_std
        )

        yaxis_name = "yaxis{}".format(len(data_dicts) - data_name_index).replace(
            "axis1", "axis"
        )

        domain_end = domain_start - row_fraction

        if abs(domain_end) <= eps:

            domain_end = 0

        domain_start = domain_end - data_dict["df"].shape[0] * row_fraction

        if abs(domain_start) <= eps:

            domain_start = 0

        layout[yaxis_name] = dict(
            domain=(domain_start, domain_end),
            dtick=1,
            tickfont=dict(size=annotation_font_size),
        )

        data.append(
            dict(
                yaxis=yaxis_name.replace("axis", ""),
                type="heatmap",
                z=features_to_plot.values[::-1],
                x=features_to_plot.columns,
                y=features_to_plot.index[::-1],
                zmin=features_plot_min,
                zmax=features_plot_max,
                colorscale=features_colorscale,
                showscale=False,
            )
        )

        layout_annotation_template = dict(
            xref="paper",
            yref="paper",
            yanchor="middle",
            font=dict(size=annotation_font_size),
            showarrow=False,
        )

        layout["annotations"].append(
            dict(
                xanchor="center",
                x=0.5,
                y=domain_end + (row_fraction / 2),
                text="<b>{}</b>".format(data_name),
                **layout_annotation_template,
            )
        )

        layout_annotation_template.update(dict(xanchor="left", width=ANNOTATION_WIDTH))

        for (
            annotation_index,
            (annotation_column_name, annotation_column_strs),
        ) in enumerate(annotations.items()):

            x = 1.0016 + annotation_index / 10

            if data_name_index == 0:

                layout["annotations"].append(
                    dict(
                        x=x,
                        y=1 - (row_fraction / 2),
                        text="<b>{}</b>".format(annotation_column_name),
                        **layout_annotation_template,
                    )
                )

            y = domain_end - (row_fraction / 2)

            for str_ in annotation_column_strs:

                layout["annotations"].append(
                    dict(
                        x=x,
                        y=y,
                        text="<b>{}</b>".format(str_),
                        **layout_annotation_template,
                    )
                )

                y -= row_fraction

    plot_and_save(dict(layout=layout, data=data), html_file_path, plotly_html_file_path)
