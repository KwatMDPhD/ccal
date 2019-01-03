from warnings import warn

from numpy import absolute, in1d

from ._plot_mountain import _plot_mountain


def single_sample_gsea(
    gene_score,
    gene_set_genes,
    statistic="ks",
    plot=True,
    title=None,
    gene_score_name=None,
    annotation_text_font_size=16,
    annotation_text_width=88,
    annotation_text_yshift=64,
    html_file_path=None,
    plotly_html_file_path=None,
):

    gene_score = gene_score.dropna()

    gene_score_sorted = gene_score.sort_values(ascending=False)

    in_ = in1d(gene_score_sorted.index, gene_set_genes.dropna(), assume_unique=True)

    in_sum = in_.sum()

    if in_sum == 0:

        warn("Gene scores did not have any of the gene-set genes.")

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

    if statistic not in ("ks", "auc"):

        raise ValueError("Unknown statistic: {}.".format(statistic))

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

        _plot_mountain(
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
            plotly_html_file_path,
        )

    return score
