from numpy import absolute, asarray

from .plot_and_save import plot_and_save


def plot_gsea_mountain(
    cumulative_sums,
    hits,
    gene_score,
    score,
    layout_width,
    layout_height,
    title,
    annotation_text_font_size,
    annotation_text_width,
    annotation_text_yshift,
    html_file_path,
):

    layout = {
        "width": layout_width,
        "height": layout_height,
        "title": {"text": title},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, 0.16), "title": gene_score.name},
        "yaxis2": {"domain": (0.20, 1), "title": "Enrichment"},
    }

    data = []

    grid = asarray(range(cumulative_sums.size))

    line_width = 3.2

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Cumulative Sum",
            "x": grid,
            "y": cumulative_sums,
            "line": {"width": line_width, "color": "#20d9ba"},
            "fill": "tozeroy",
        }
    )

    cumulative_sums_argmax = absolute(cumulative_sums).argmax()

    negative_color = "#4e40d8"

    positive_color = "#ff1968"

    if score < 0:

        color = negative_color

    else:

        color = positive_color

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Peak ({:.3f})".format(score),
            "x": (grid[cumulative_sums_argmax],),
            "y": (cumulative_sums[cumulative_sums_argmax],),
            "mode": "markers",
            "marker": {"size": 12, "color": color},
        }
    )

    gene_xs = tuple(i for i in grid if hits[i])

    gene_texts = tuple("<b>{}</b>".format(text) for text in gene_score[hits].index)

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Gene",
            "x": gene_xs,
            "y": (0,) * len(gene_xs),
            "text": gene_texts,
            "mode": "markers",
            "marker": {
                "symbol": "line-ns-open",
                "size": 16,
                "color": "#9017e6",
                "line": {"width": line_width},
            },
            "hoverinfo": "x+text",
        }
    )

    is_negative = gene_score < 0

    for indices, name, color in (
        (is_negative, "- Gene Score", negative_color),
        (~is_negative, "+ Gene Score", positive_color),
    ):

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": grid[indices],
                "y": gene_score[indices],
                "line": {"width": line_width, "color": color},
                "fill": "tozeroy",
            }
        )

    layout["annotations"] = [
        {
            "x": x,
            "y": 0,
            "yref": "y2",
            "clicktoshow": "onoff",
            "text": text,
            "showarrow": False,
            "font": {"size": annotation_text_font_size},
            "textangle": -90,
            "width": annotation_text_width,
            "borderpad": 0,
            "yshift": (-annotation_text_yshift, annotation_text_yshift)[i % 2],
        }
        for i, (x, text) in enumerate(zip(gene_xs, gene_texts))
    ]

    plot_and_save({"layout": layout, "data": data}, html_file_path)
