from numpy import absolute, in1d


from numpy import asarray

from .plot_and_save import plot_and_save


def _plot_gsea_mountain(
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
):

    layout = {
        "width": layout_width,
        "height": layout_height,
        "title": {"text": title},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, 0.16), "title": gene_score_name},
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


def single_sample_gsea(
    gene_score,
    gene_set_genes,
    statistic="ks",
    plot=True,
    title="GSEA Mountain Plot",
    gene_score_name="Gene Score",
    annotation_text_font_size=16,
    annotation_text_width=88,
    annotation_text_yshift=64,
    html_file_path=None,
):

    gene_score = gene_score.dropna()

    gene_score_sorted = gene_score.sort_values(ascending=False)

    in_ = in1d(gene_score_sorted.index, gene_set_genes.dropna(), assume_unique=True)

    in_sum = in_.sum()

    if in_sum == 0:

        print("Gene scores did not have any of the gene-set genes.")

        return

    gene_score_sorted_values = gene_score_sorted.values

    gene_score_sorted_values_absolute = absolute(gene_score_sorted_values)

    in_int = in_.astype(int)

    hit = (
        gene_score_sorted_values_absolute * in_int
    ) / gene_score_sorted_values_absolute[in_].sum()

    miss = (1 - in_int) / (in_.size - in_sum)

    y = hit - miss

    cumulative_sums = y.cumsum()

    if statistic == "ks":

        max_ = cumulative_sums.max()

        min_ = cumulative_sums.min()

        if absolute(min_) < absolute(max_):

            score = max_

        else:

            score = min_

    elif statistic == "auc":

        score = cumulative_sums.sum()

    if plot:

        _plot_gsea_mountain(
            cumulative_sums,
            in_,
            gene_score_sorted,
            score,
            None,
            None,
            title,
            gene_score_name,
            annotation_text_font_size,
            annotation_text_width,
            annotation_text_yshift,
            html_file_path,
        )

    return score
