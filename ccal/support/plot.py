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
from numpy import array, asarray, unique
from pandas import Series, DataFrame
from seaborn import light_palette, heatmap, clustermap, pointplot, boxplot, violinplot, set_style, despine

from .file import establish_filepath
from .str_ import untitle_str
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
def plot_heatmap(dataframe, data_type='continuous',
                 normalization_method=None, normalization_axis=0, sort_axis=None, vmin=None, vmax=None,
                 row_annotation=(), column_annotation=(),
                 center=None,
                 annot=None, fmt='.2g', annot_kws=None,
                 linewidth=0, linecolor='white',
                 mask=None,
                 square=False,
                 title=None, xlabel=None, ylabel=None, xlabel_rotation=0, ylabel_rotation=90,
                 xticklabels=True, yticklabels=True, yticklabels_rotation='auto',
                 filepath=None):
    """

    :param dataframe:
    :param data_type:
    :param normalization_method:
    :param normalization_axis:
    :param sort_axis:
    :param vmin:
    :param vmax:
    :param row_annotation:
    :param column_annotation:
    :param center:
    :param annot:
    :param fmt:
    :param annot_kws:
    :param linewidth:
    :param linecolor:
    :param mask:
    :param square:
    :param title:
    :param xlabel:
    :param ylabel:
    :param xlabel_rotation:
    :param ylabel_rotation:
    :param xticklabels:
    :param yticklabels:
    :param yticklabels_rotation:
    :param filepath:
    :return:
    """

    df = dataframe.copy()

    if normalization_method:
        df = normalize_dataframe_or_series(df, normalization_method, axis=normalization_axis)
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
            column_annotation = Series(column_annotation, index=df.columns)
            column_annotation.sort_values(inplace=True)
            df = df.ix[:, column_annotation.index]
    elif sort_axis in (0, 1):
        a = array(df)
        a.sort(axis=sort_axis)
        df = DataFrame(a, index=df.index)

    plt.figure(figsize=FIGURE_SIZE)

    if title:
        plt.suptitle(title, **FONT_TITLE)

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

    if data_type == 'continuous':
        cmap = CMAP_CONTINUOUS
    elif data_type == 'categorical':
        cmap = CMAP_CATEGORICAL
    elif data_type == 'binary':
        cmap = CMAP_BINARY
    else:
        raise ValueError('Target data type must be one of {continuous, categorical, binary}.')

    heatmap(df, vmin=vmin, vmax=vmax, center=center, annot=annot, fmt=fmt, annot_kws=annot_kws,
            linewidths=linewidth, linecolor=linecolor, cbar=False, square=square, mask=mask,
            cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax_center)

    if xlabel:
        ax_center.set_xlabel(xlabel, rotation=xlabel_rotation, **FONT_SUBTITLE)
    if ylabel:
        ax_center.set_ylabel(ylabel, rotation=ylabel_rotation, **FONT_SUBTITLE)

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

    if filepath:
        save_plot(filepath)


def plot_clustermap(dataframe, cmap=CMAP_CONTINUOUS, row_colors=None, col_colors=None,
                    title=None, xlabel=None, ylabel=None, xticklabels=True, yticklabels=True,
                    filepath=None):
    """
    Plot heatmap for dataframe.
    :param dataframe: DataFrame;
    :param cmap: colormap;
    :param row_colors: list-like or DataFrame/Series; List of colors to label for either the rows.
        Useful to evaluate whether samples within a group_iterable are clustered together.
        Can use nested lists or DataFrame for multiple color levels of labeling.
        If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param col_colors: list-like or DataFrame/Series; List of colors to label for either the column.
        Useful to evaluate whether samples within a group_iterable are clustered together.
        Can use nested lists or DataFrame for multiple color levels of labeling.
        If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param title: str;
    :param xlabel: str;
    :param ylabel: str;
    :param xticklabels: bool;
    :param yticklabels: bool;
    :param filepath: str;
    :return: None
    """

    # Initialize a figure
    plt.figure(figsize=FIGURE_SIZE)

    # Plot cluster map
    clustergrid = clustermap(dataframe, cmap=cmap, row_colors=row_colors, col_colors=col_colors,
                             xticklabels=xticklabels, yticklabels=yticklabels)

    # Title
    if title:
        plt.suptitle(title, **FONT_TITLE)

    # X & Y labels
    if xlabel:
        clustergrid.ax_heatmap.set_xlabel(xlabel, **FONT_SUBTITLE)
    if ylabel:
        clustergrid.ax_heatmap.set_ylabel(ylabel, **FONT_SUBTITLE)

    # X & Y ticks
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set(**FONT)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set(**FONT)

    # Save
    if filepath:
        save_plot(filepath)


def plot_x_vs_y(x, y, title='title', xlabel='xlabel', ylabel='ylabel', filepath=None):
    """
    Plot x vs y.
    :param x:
    :param y:
    :param title:
    :param xlabel:
    print(xlabel, ylabel)
    :param ylabel:
    :param filepath:
    :return:
    """

    plt.figure(figsize=FIGURE_SIZE)

    if title:
        plt.suptitle(title, **FONT_TITLE)

    pointplot(x, y)

    plt.gca().set_xlabel(xlabel, **FONT_SUBTITLE)
    plt.gca().set_ylabel(ylabel, **FONT_SUBTITLE)

    if filepath:
        save_plot(filepath)


def plot_box_or_violine(target, features, features_name, feature_names=(), box_or_violine='violine',
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

    set_style('whitegrid')
    for r_i, r in features.ix[feature_names, :].iterrows():
        common_r = r.ix[target.index]

        plt.figure(figsize=FIGURE_SIZE)
        if not title:
            title = '{} {}'.format(features_name, r_i)
        plt.suptitle(title, **FONT_TITLE)
        if box_or_violine == 'violine':
            violinplot(x=target, y=common_r)
        if box_or_violine == 'box':
            boxplot(x=target, y=common_r)
        despine(left=True)
        score, pval = compute_association_and_pvalue(asarray(target), asarray(common_r), n_permutations=1000)
        l, r, b, t = plt.gca().axis()
        plt.gca().text((l + r) / 2, t + 0.016 * t, 'Score = {0:.3f} & P-val = {1:.3f}'.format(score, pval),
                       horizontalalignment='center', **FONT_SUBTITLE)

        if not xlabel:
            xlabel = plt.gca().get_xlabel()
        plt.gca().set_xlabel(xlabel, **FONT_SUBTITLE)

        if not ylabel:
            ylabel = plt.gca().get_ylabel()
        plt.gca().set_ylabel(ylabel, **FONT_SUBTITLE)

        for t in plt.gca().get_xticklabels():
            t.set(**FONT)
        for t in plt.gca().get_yticklabels():
            t.set(**FONT)

    if filepath_prefix:
        save_plot(filepath_prefix + '{}.pdf'.format(untitle_str(title)))


def plot_nmf(nmf_results=None, k=None, w_matrix=None, h_matrix=None, normalize=True, max_std=3, title=None,
             filepath=None):
    """
    Plot nmf_results dictionary (can be generated by ccal.analyze.nmf function).
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param k: int; k for NMF
    :param w_matrix: DataFrame
    :param h_matrix: DataFrame
    :param normalize: bool; normalize W and H matrices or not ('-0-' normalization on the component axis)
    :param max_std: number; threshold to clip standardized values
    :param title: str;
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

    # Plot W and H
    plt.figure(figsize=FIGURE_SIZE)
    gridspec = GridSpec(10, 16)
    ax_w = plt.subplot(gridspec[1:, :5])
    ax_h = plt.subplot(gridspec[3:8, 7:])
    if not title:
        title = 'NMF Result for k={}'.format(w_matrix.shape[1])
    plt.suptitle(title, **FONT_TITLE)

    # Plot W
    if normalize:
        w_matrix = normalize_dataframe_or_series(w_matrix, method='-0-', axis=0).clip(-max_std, max_std)
    heatmap(w_matrix, cmap=CMAP_CONTINUOUS, yticklabels=False, ax=ax_w)
    ax_w.set_title('W Matrix for k={}'.format(w_matrix.shape[1]), **FONT_TITLE)
    ax_w.set_xlabel('Component', **FONT_SUBTITLE)
    ax_w.set_ylabel('Feature', **FONT_SUBTITLE)
    for t in ax_w.get_xticklabels():
        t.set(**FONT)

    # Plot H
    if normalize:
        h_matrix = normalize_dataframe_or_series(h_matrix, method='-0-', axis=1).clip(-max_std, max_std)
    heatmap(h_matrix, cmap=CMAP_CONTINUOUS, xticklabels=False, cbar_kws={'orientation': 'horizontal'}, ax=ax_h)
    ax_h.set_title('H Matrix for k={}'.format(h_matrix.shape[0]), **FONT_TITLE)
    ax_h.set_xlabel('Sample', **FONT_SUBTITLE)
    ax_h.set_ylabel('Component', **FONT_SUBTITLE)
    for t in ax_h.get_yticklabels():
        t.set(**FONT)

    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    # Plot cluster map for W
    clustergrid = clustermap(w_matrix, standard_scale=0, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('W Matrix for k={}'.format(w_matrix.shape[1]), **FONT_TITLE)
    clustergrid.ax_heatmap.set_xlabel('Component', **FONT_SUBTITLE)
    clustergrid.ax_heatmap.set_ylabel('Feature', **FONT_SUBTITLE)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_fontweight('bold')
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_visible(False)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    # Plot cluster map for H
    clustergrid = clustermap(h_matrix, standard_scale=1, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('H Matrix for k={}'.format(h_matrix.shape[0]), **FONT_TITLE)
    clustergrid.ax_heatmap.set_xlabel('Sample', **FONT_SUBTITLE)
    clustergrid.ax_heatmap.set_ylabel('Component', **FONT_SUBTITLE)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_visible(False)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_fontweight('bold')
        t.set_rotation(0)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    if filepath:
        pdf.close()


def save_plot(filepath, suffix='.pdf', overwrite=True, dpi=DPI):
    """
    Establish filepath and save plot (.pdf) at dpi resolution.
    :param filepath: str;
    :param suffix: str;
    :param overwrite: bool;
    :param dpi: int;
    :return: None
    """

    if not filepath.endswith(suffix):
        filepath += suffix

    if not isfile(filepath) or overwrite:  # If the figure doesn't exist or overwriting
        establish_filepath(filepath)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
