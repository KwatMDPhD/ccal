from numpy import absolute, asarray

from .plot_and_save import plot_and_save


def _plot_mountain(
    cumulative_sums,
    hits,
    gene_score,
    score,
    layout_width,
    layout_height,
    title,
    gene_score_name,
    annotation_text_font_size,
    annotation_text_width,
    annotation_text_yshift,
    html_file_path,
    plotly_html_file_path,
):

    if title is None:

        title = "GSEA Mountain Plot"

    if gene_score_name is None:

        gene_score_name = "Gene Score"

    layout = dict(
        width=layout_width,
        height=layout_height,
        hovermode="closest",
        title=title,
        xaxis=dict(anchor="y", title="Rank"),
        yaxis=dict(domain=(0, 0.16), title=gene_score_name),
        yaxis2=dict(domain=(0.20, 1), title="Enrichment"),
        legend=dict(orientation="h"),
    )

    data = []

    grid = asarray(range(cumulative_sums.size))

    line_width = 3.2

    data.append(
        dict(
            yaxis="y2",
            type="scatter",
            name="Cumulative Sum",
            x=grid,
            y=cumulative_sums,
            line=dict(width=line_width, color="#20d9ba"),
            fill="tozeroy",
        )
    )

    cumulative_sums_argmax = absolute(cumulative_sums).argmax()

    negative_color = "#4e40d8"

    positive_color = "#ff1968"

    data.append(
        dict(
            yaxis="y2",
            type="scatter",
            name="Peak",
            x=(grid[cumulative_sums_argmax],),
            y=(cumulative_sums[cumulative_sums_argmax],),
            mode="markers",
            marker=dict(size=16, color=(negative_color, positive_color)[0 <= score]),
        )
    )

    gene_xs = tuple(i for i in grid if hits[i])

    gene_texts = tuple("<b>{}</b>".format(text) for text in gene_score[hits].index)

    data.append(
        dict(
            yaxis="y2",
            type="scatter",
            name="Gene Set",
            x=gene_xs,
            y=(0,) * len(gene_xs),
            text=gene_texts,
            mode="markers",
            marker=dict(
                symbol="line-ns-open",
                size=16,
                color="#9017e6",
                line=dict(width=line_width),
            ),
            hoverinfo="x+text",
        )
    )

    is_negative = gene_score < 0

    for indices, color in (
        (is_negative, negative_color),
        (~is_negative, positive_color),
    ):

        data.append(
            dict(
                type="scatter",
                name="Gene Score",
                legendgroup="Gene Score",
                x=grid[indices],
                y=gene_score[indices],
                line=dict(width=line_width, color=color),
                fill="tozeroy",
            )
        )

    layout["annotations"] = [
        dict(
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            text="<b>Score = {:.3f}</b>".format(score),
            showarrow=False,
            font=dict(size=16, color="#ffffff"),
            bgcolor=(negative_color, positive_color)[0 <= score],
            borderpad=3.2,
        )
    ] + [
        dict(
            x=x,
            y=0,
            yref="y2",
            clicktoshow="onoff",
            text=text,
            showarrow=False,
            font=dict(size=annotation_text_font_size),
            textangle=-90,
            width=annotation_text_width,
            borderpad=0,
            yshift=(-annotation_text_yshift, annotation_text_yshift)[i % 2],
        )
        for i, (x, text) in enumerate(zip(gene_xs, gene_texts))
    ]

    plot_and_save(dict(layout=layout, data=data), html_file_path, plotly_html_file_path)
