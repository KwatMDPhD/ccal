"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from os.path import isfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import Paired, bwr
from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import plot
from numpy import array, asarray, unique
from pandas import Series, DataFrame
from seaborn import light_palette, heatmap, clustermap, distplot, boxplot, violinplot, set_style, despine

from .d2 import get_dendrogram_leaf_indices
from .file import establish_filepath
from .str_ import title_str, untitle_str
from ..machine_learning.normalize import normalize_dataframe_or_series
from ..machine_learning.score import compute_association_and_pvalue

# ======================================================================================================================
# Parameter
# ======================================================================================================================
FIGURE_SIZE = (16, 10)

SPACING = 0.05

# Fonts
FONT_TITLE = {'fontsize': 26, 'weight': 'bold'}
FONT_SUBTITLE = {'fontsize': 20, 'weight': 'bold'}
FONT = {'fontsize': 12, 'weight': 'bold'}

# Color maps
CMAP_CONTINUOUS = bwr
CMAP_CONTINUOUS.set_bad('wheat')

reds = [0.26, 0.26, 0.26, 0.39, 0.69, 1, 1, 1, 1, 1, 1]
greens_half = [0.26, 0.16, 0.09, 0.26, 0.69]
colordict = {'red': tuple([(0.1 * i, r, r) for i, r in enumerate(reds)]),
             'green': tuple([(0.1 * i, r, r) for i, r in enumerate(greens_half + [1] + list(reversed(greens_half)))]),
             'blue': tuple([(0.1 * i, r, r) for i, r in enumerate(reversed(reds))])}
CMAP_ASSOCIATION = LinearSegmentedColormap('association', colordict)
CMAP_ASSOCIATION.set_bad('wheat')

CMAP_CATEGORICAL = Paired
CMAP_CATEGORICAL.set_bad('wheat')

CMAP_BINARY = light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad('wheat')

DPI = 1000


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_heatmap(dataframe, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g',
                 annot_kws=None, linewidths=0, linecolor='white', cbar=False, cbar_kws=None, cbar_ax=None, square=False,
                 xticklabels=True, yticklabels=True, mask=None,
                 data_type='continuous', normalization_method='-0-', normalization_axis=0, max_std=3, sort_axis=None,
                 cluster=False, row_annotation=(), column_annotation=(), title=None, xlabel=None, ylabel=None,
                 xlabel_rotation=0, ylabel_rotation=90, yticklabels_rotation='auto',
                 filepath=None, **kwargs):
    df = dataframe.copy()

    if normalization_method:
        df = normalize_dataframe_or_series(df, normalization_method, axis=normalization_axis).clip(-max_std, max_std)
    values = unique(df.values)

    if any(row_annotation) or any(column_annotation):
        if any(row_annotation):
            if isinstance(row_annotation, Series):
                if not any(row_annotation.index & df.index):
                    row_annotation.index = df.index
            else:
                row_annotation = Series(row_annotation, index=df.index)

            row_annotation.sort_values(inplace=True)
            df = df.ix[row_annotation.index, :]

        if any(column_annotation):
            if isinstance(column_annotation, Series):
                if not any(column_annotation.index & df.index):
                    column_annotation.index = df.columns
            else:
                column_annotation = Series(column_annotation, index=df.columns)

            column_annotation.sort_values(inplace=True)
            df = df.ix[:, column_annotation.index]

    if cluster:
        row_indices, column_indices = get_dendrogram_leaf_indices(dataframe)
        df = df.iloc[row_indices, column_indices]
        if isinstance(row_annotation, Series):
            row_annotation = row_annotation.iloc[row_indices]
        if isinstance(column_annotation, Series):
            column_annotation = column_annotation.iloc[column_indices]

    elif sort_axis in (0, 1):
        a = array(df)
        a.sort(axis=sort_axis)
        df = DataFrame(a, index=df.index)

    plt.figure(figsize=FIGURE_SIZE)

    gridspec = GridSpec(10, 10)

    ax_top = plt.subplot(gridspec[0:1, 2:-2])
    ax_center = plt.subplot(gridspec[1:8, 2:-2])
    ax_bottom = plt.subplot(gridspec[8:10, 2:-2])
    ax_left = plt.subplot(gridspec[1:8, 1:2])
    ax_right = plt.subplot(gridspec[1:8, 8:9])

    ax_top.axis('off')
    ax_bottom.axis('off')
    ax_left.axis('off')
    ax_right.axis('off')

    if not cmap:
        if data_type == 'continuous':
            cmap = CMAP_CONTINUOUS
        elif data_type == 'categorical':
            cmap = CMAP_CATEGORICAL
        elif data_type == 'binary':
            cmap = CMAP_BINARY
        else:
            raise ValueError('Target data type must be one of {continuous, categorical, binary}.')

    heatmap(df, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust, annot=annot, fmt=fmt,
            annot_kws=annot_kws, linewidths=linewidths, linecolor=linecolor, cbar=cbar, cbar_kws=cbar_kws,
            cbar_ax=cbar_ax, square=square, ax=ax_center, xticklabels=xticklabels, yticklabels=yticklabels, mask=mask,
            **kwargs)

    title_and_label(title, xlabel, ylabel, xlabel_rotation=xlabel_rotation, ylabel_rotation=ylabel_rotation,
                    ax=ax_center)

    xticklabels = ax_center.get_xticklabels()
    ax_center.set_xticklabels([t.get_text()[:10].strip() for t in xticklabels])
    for t in xticklabels:
        t.set(**FONT)

    yticks = ax_center.get_yticklabels()
    if any(yticks):
        if yticklabels_rotation == 'auto':
            if max([len(t.get_text()) for t in yticks]) <= 1:
                yticklabels_rotation = 90
            else:
                yticklabels_rotation = 0
        for t in yticks:
            t.set(rotation=yticklabels_rotation, **FONT)

    if data_type in ('categorical', 'binary'):
        if len(values) < 30:
            horizontal_span = ax_center.axis()[1]
            vertival_span = ax_center.axis()[3]
            for i, v in enumerate(values):
                x = (horizontal_span / len(values) / 2) + i * horizontal_span / len(values)
                y = 0 - vertival_span * 0.09
                ax_center.plot(x, y, 'o', markersize=16, aa=True, clip_on=False)
                ax_center.text(x, y - vertival_span * 0.05, v, horizontalalignment='center', **FONT)

    if data_type == 'continuous':
        cax, kw = make_axes(ax_bottom, location='bottom', fraction=0.16,
                            cmap=CMAP_CONTINUOUS,
                            norm=Normalize(values.min(), values.max()),
                            ticks=[values.min(), values.mean(), values.max()])
        ColorbarBase(cax, **kw)
        for t in cax.get_xticklabels():
            t.set(**FONT)

    if any(row_annotation):
        if len(set(row_annotation)) <= 2:
            cmap = CMAP_BINARY
        else:
            cmap = CMAP_CATEGORICAL
        heatmap(DataFrame(row_annotation), ax=ax_right, cbar=False, xticklabels=False, yticklabels=False,
                cmap=cmap)

    if any(column_annotation):
        if len(set(column_annotation)) <= 2:
            cmap = CMAP_BINARY
        else:
            cmap = CMAP_CATEGORICAL
        heatmap(DataFrame(column_annotation).T, ax=ax_top, cbar=False, xticklabels=False, yticklabels=False,
                cmap=cmap)

    save_plot(filepath)


def plot_clustermap(dataframe, pivot_kws=None, method='average', metric='euclidean', z_score=None, standard_scale=None,
                    figsize=FIGURE_SIZE, cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None,
                    col_linkage=None,
                    row_colors=None, col_colors=None, mask=None, cmap=CMAP_CONTINUOUS,
                    title=None, xlabel=None, ylabel=None, xticklabels=True, yticklabels=True, filepath=None, **kwargs):
    """

    :param dataframe:
    :param pivot_kws:
    :param method:
    :param metric:
    :param z_score:
    :param standard_scale:
    :param figsize:
    :param cbar_kws:
    :param row_cluster:
    :param col_cluster:
    :param row_linkage:
    :param col_linkage:
    :param row_colors:
    :param col_colors:
    :param mask:
    :param cmap:
    :param title:
    :param xlabel:
    :param ylabel:
    :param xticklabels:
    :param yticklabels:
    :param filepath:
    :param kwargs:
    :return:
    """

    # Initialize a figure
    plt.figure(figsize=figsize)

    # Plot cluster map
    clustergrid = clustermap(dataframe, pivot_kws=pivot_kws, method=method, metric=metric, z_score=z_score,
                             standard_scale=standard_scale, figsize=figsize, cbar_kws=cbar_kws, row_cluster=row_cluster,
                             col_cluster=col_cluster, row_linkage=row_linkage, col_linkage=col_linkage,
                             row_colors=row_colors, col_colors=col_colors, mask=mask, cmap=cmap,
                             xticklabels=xticklabels, yticklabels=yticklabels, **kwargs)

    ax_heatmap = clustergrid.ax_heatmap

    # X & Y ticks
    for t in ax_heatmap.get_xticklabels():
        t.set(**FONT)
    for t in ax_heatmap.get_yticklabels():
        t.set(**FONT)

    title_and_label(title, xlabel, ylabel, ax=ax_heatmap)

    save_plot(filepath)


def plot_points(*args, title='', xlabel='', ylabel='', filepath=None, **kwargs):
    """

    :param args:
    :param title:
    :param xlabel:
    :param ylabel:
    :param filepath:
    :param kwargs:
    :return: None
    """

    if 'ax' not in kwargs:
        plt.figure(figsize=FIGURE_SIZE)

    # Preprocess args
    processed_args = []
    for i, a in enumerate(args):
        if isinstance(a, Series):
            processed_args.append(a.tolist())
        else:
            processed_args.append(a)

    plot(*processed_args, **kwargs)

    title_and_label(title, xlabel, ylabel)

    save_plot(filepath)


def plot_distribution(a, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None,
                      fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None,
                      ax=None, title='', xlabel='', ylabel='Frequency', filepath=None):
    """

    :param a:
    :param bins:
    :param hist:
    :param kde:
    :param rug:
    :param fit:
    :param hist_kws:
    :param kde_kws:
    :param rug_kws:
    :param fit_kws:
    :param color:
    :param vertical:
    :param norm_hist:
    :param axlabel:
    :param label:
    :param ax:
    :param title:
    :param xlabel:
    :param ylabel:
    :param filepath:
    :return:
    """

    if not ax:
        plt.figure(figsize=FIGURE_SIZE)

    distplot(a, bins=bins, hist=hist, kde=kde, rug=rug, fit=fit, hist_kws=hist_kws, kde_kws=kde_kws, rug_kws=rug_kws,
             fit_kws=fit_kws, color=color, vertical=vertical, norm_hist=norm_hist, axlabel=axlabel, label=label,
             ax=ax)

    title_and_label(title, xlabel, ylabel)

    save_plot(filepath)


def plot_violine(target, features, features_name, feature_names=(), box_or_violine='violine',
                 title=None, xlabel=None, ylabel=None,
                 filepath_prefix=None):
    """

    :param target:
    :param features:
    :param features_name:
    :param feature_names:
    :param box_or_violine:
    :param title:
    :param xlabel:
    :param ylabel:
    :param filepath_prefix:
    :return:
    """

    plt.figure(figsize=FIGURE_SIZE)

    set_style('whitegrid')
    for r_i, r in features.ix[feature_names, :].iterrows():
        common_r = r.ix[target.index]

        plt.figure(figsize=FIGURE_SIZE)
        if box_or_violine == 'violine':
            violinplot(x=target, y=common_r)
        if box_or_violine == 'box':
            boxplot(x=target, y=common_r)
        despine(left=True)
        score, pval = compute_association_and_pvalue(asarray(target), asarray(common_r), n_permutations=1000)
        l, r, b, t = plt.gca().axis()
        plt.gca().text((l + r) / 2, t + 0.016 * t, 'Score = {0:.3f} & P-val = {1:.3f}'.format(score, pval),
                       horizontalalignment='center', **FONT_SUBTITLE)

        if not title:
            title = '{} {}'.format(features_name, r_i)
        title_and_label(title, xlabel, ylabel)

        for t in plt.gca().get_xticklabels():
            t.set(**FONT)
        for t in plt.gca().get_yticklabels():
            t.set(**FONT)

    if filepath_prefix:
        save_plot(filepath_prefix + '{}.pdf'.format(untitle_str(title)))


def plot_nmf(nmf_results=None, k=None, w_matrix=None, h_matrix=None, max_std=3, filepath=None):
    """
    Plot nmf_results dictionary (can be generated by ccal.analyze.nmf function).
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param k: int; k for NMF
    :param w_matrix: DataFrame
    :param h_matrix: DataFrame
    :param max_std: number; threshold to clip standardized values
    :param filepath: str;
    :return: None
    """

    # Check for W and H matrix
    if isinstance(nmf_results, dict) and k:
        w_matrix = nmf_results[k]['w']
        h_matrix = nmf_results[k]['h']
    elif not (isinstance(w_matrix, DataFrame) and isinstance(h_matrix, DataFrame)):
        raise ValueError('Need either: 1) NMF result ({k: {W:w, H:h, ERROR:error}) and k; or 2) W and H matrices.')

    # Initialize a PDF
    if filepath:
        establish_filepath(filepath)
        if not filepath.endswith('.pdf'):
            filepath += '.pdf'
        pdf = PdfPages(filepath)

    # Initialize a figure
    plt.figure(figsize=FIGURE_SIZE)

    # Plot cluster map for W
    plot_heatmap(w_matrix, cluster=True,
                 title='W Matrix for k={}'.format(w_matrix.shape[1]), xlabel='Component', ylabel='Feature',
                 yticklabels=False, normalization_method='-0-', normalization_axis=0, max_std=max_std)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    # Plot cluster map for H
    plot_heatmap(h_matrix, cluster=True,
                 title='H Matrix for k={}'.format(h_matrix.shape[0]), xlabel='Sample', ylabel='Component',
                 xticklabels=False, normalization_method='-0-', normalization_axis=1, max_std=max_std)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    if filepath:
        pdf.close()


def title_and_label(title, xlabel, ylabel, xlabel_rotation=0, ylabel_rotation=90, ax=None):
    """

    :param title:
    :param xlabel:
    :param ylabel:
    :param xlabel_rotation:
    :param ylabel_rotation:
    :param ax:
    :return:
    """

    # Title
    if title:
        plt.suptitle(title_str(title), **FONT_TITLE)

    # Label
    if not ax:
        ax = plt.gca()

    if not xlabel:
        xlabel = ax.get_xlabel()
    ax.set_xlabel(title_str(xlabel), rotation=xlabel_rotation, **FONT_SUBTITLE)

    if not ylabel:
        ylabel = ax.get_ylabel()
    ax.set_ylabel(title_str(ylabel), rotation=ylabel_rotation, **FONT_SUBTITLE)


def save_plot(filepath, suffix='.pdf', overwrite=True, dpi=DPI):
    """
    Establish filepath and save plot (.pdf) at dpi resolution.
    :param filepath: str;
    :param suffix: str;
    :param overwrite: bool;
    :param dpi: int;
    :return: None
    """

    if filepath:
        if not filepath.endswith(suffix):
            filepath += suffix

        if not isfile(filepath) or overwrite:  # If the figure doesn't exist or overwriting
            establish_filepath(filepath)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
