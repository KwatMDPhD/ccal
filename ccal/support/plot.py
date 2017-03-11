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
from matplotlib.cm import Paired, Dark2, bwr
from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize, ColorConverter
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import plot
from numpy import array, unique, isnan
from pandas import Series, DataFrame
from seaborn import set_style, despine, distplot, barplot, violinplot, boxplot, heatmap, clustermap

from .d2 import get_dendrogram_leaf_indices, normalize_2d_or_1d
from .file import establish_filepath

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
C_BAD = 'wheat'

# Continuous 1
CMAP_CONTINUOUS = bwr
CMAP_CONTINUOUS.set_bad('yellow')
CMAP_CONTINUOUS.set_bad(C_BAD)

# Continuous 2
reds = [0.26, 0.26, 0.26, 0.39, 0.69, 1, 1, 1, 1, 1, 1]
greens_half = [0.26, 0.16, 0.09, 0.26, 0.69]
colordict = {'red': tuple([(0.1 * i, r, r) for i, r in enumerate(reds)]),
             'green': tuple([(0.1 * i, r, r) for i, r in enumerate(greens_half + [1] + list(reversed(greens_half)))]),
             'blue': tuple([(0.1 * i, r, r) for i, r in enumerate(reversed(reds))])}
CMAP_ASSOCIATION = LinearSegmentedColormap('association', colordict)
CMAP_ASSOCIATION.set_bad(C_BAD)

# Categorical
CMAP_CATEGORICAL = Paired
CMAP_CATEGORICAL_2 = Dark2
CMAP_CATEGORICAL.set_bad(C_BAD)

# Binary
CMAP_BINARY = ListedColormap(['#cdcdcd', '#404040'])
CMAP_BINARY.set_bad(C_BAD)

DPI = 100


# ======================================================================================================================
# Functions
# ======================================================================================================================
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

    plot(*[list(a) for a in args], marker='o', **kwargs)

    decorate(style='ticks', title=title, xlabel=xlabel, ylabel=ylabel)

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

    decorate(style='ticks', title=title, xlabel=xlabel, ylabel=ylabel)

    save_plot(filepath)


def plot_violin_box_or_bar(x=None, y=None, hue=None, data=None, order=None, hue_order=None, bw='scott', cut=2,
                           scale='count', scale_hue=True, gridsize=100, width=0.8, inner='quartile', split=False,
                           orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None,
                           fliersize=5, whis=1.5, notch=False,
                           ci=95, n_boot=1000, units=None, errcolor='0.26', errwidth=None, capsize=None,
                           violin_or_box='violin', colors=(),
                           figure_size=FIGURE_SIZE, title=None, xlabel=None, ylabel=None, filepath=None,
                           **kwargs):
    """
    Plot violin plot.
    :param x:
    :param y:
    :param hue:
    :param data:
    :param order:
    :param hue_order:
    :param bw:
    :param cut:
    :param scale:
    :param scale_hue:
    :param gridsize:
    :param width:
    :param inner:
    :param split:
    :param orient:
    :param linewidth:
    :param color:
    :param palette:
    :param saturation:
    :param ax:
    :param fliersize:
    :param whis:
    :param notch:
    :param ci:
    :param n_boot:
    :param units:
    :param errcolor:
    :param errwidth:
    :param capsize:
    :param violin_or_box:
    :param colors: iterable;
    :param figure_size: tuple;
    :param title:
    :param xlabel:
    :param ylabel:
    :param filepath:
    :param kwargs:
    :return: None
    """

    # Initialize a figure
    if not ax:
        plt.figure(figsize=figure_size)

    if isinstance(x, str):
        x = data[x]
    if isinstance(y, str):
        y = data[y]

    if not palette:
        palette = assign_colors_to_states(x, colors=colors)

    if len(set([v for v in y if v and ~isnan(v)])) <= 2:  # Use barplot for binary
        barplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, ci=ci, n_boot=n_boot, units=units,
                orient=orient, color=color, palette=palette, saturation=saturation, errcolor=errcolor,
                ax=ax, errwidth=errwidth, capsize=capsize, **kwargs)
    else:  # Use violin or box plot for continuous or categorical
        if violin_or_box == 'violin':
            violinplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, bw=bw, cut=cut, scale=scale,
                       scale_hue=scale_hue, gridsize=gridsize, width=width, inner=inner, split=split, orient=orient,
                       linewidth=linewidth, color=color, palette=palette, saturation=saturation, ax=ax, **kwargs)
        elif violin_or_box == 'box':
            boxplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, orient=orient, color=color,
                    palette=palette, saturation=saturation, width=width, fliersize=fliersize, linewidth=linewidth,
                    whis=whis, notch=notch, ax=ax, **kwargs)
        else:
            raise ValueError('\'violin_or_box\' must be either \'violin\' or \'box\'.')

    decorate(style='ticks', title=title, xlabel=xlabel, ylabel=ylabel)

    if filepath:
        save_plot(filepath)


def plot_heatmap(dataframe, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g',
                 annot_kws=None, linewidths=0, linecolor='white', cbar=False, cbar_kws=None, cbar_ax=None, square=False,
                 xticklabels=True, yticklabels=True, mask=None,
                 data_type='continuous', normalization_method=None, normalization_axis=0, max_std=3, axis_to_sort=None,
                 cluster=False, row_annotation=(), column_annotation=(), annotation_colors=(),
                 title=None, xlabel=None, ylabel=None, xlabel_rotation=0, ylabel_rotation=90,
                 filepath=None, **kwargs):
    """
    Plot heatmap.
    :param dataframe:
    :param vmin:
    :param vmax:
    :param cmap:
    :param center:
    :param robust:
    :param annot:
    :param fmt:
    :param annot_kws:
    :param linewidths:
    :param linecolor:
    :param cbar:
    :param cbar_kws:
    :param cbar_ax:
    :param square:
    :param xticklabels:
    :param yticklabels:
    :param mask:
    :param data_type:
    :param normalization_method:
    :param normalization_axis:
    :param max_std:
    :param axis_to_sort:
    :param cluster:
    :param row_annotation:
    :param column_annotation:
    :param annotation_colors: list; a list of matplotlib color specifications
    :param title:
    :param xlabel:
    :param ylabel:
    :param xlabel_rotation:
    :param ylabel_rotation:
    :param filepath:
    :param kwargs:
    :return:
    """

    df = dataframe.copy()

    if normalization_method:
        df = normalize_2d_or_1d(df, normalization_method, axis=normalization_axis).clip(-max_std, max_std)
    values = unique(df.values)

    if len(row_annotation) or len(column_annotation):
        if len(row_annotation):
            if isinstance(row_annotation, Series):
                row_annotation = row_annotation.copy()
                if not len(row_annotation.index & df.index):  # Series but without proper index
                    row_annotation.index = df.index
            else:
                row_annotation = Series(row_annotation, index=df.index)

            row_annotation.sort_values(inplace=True)
            df = df.ix[row_annotation.index, :]

        if len(column_annotation):
            if isinstance(column_annotation, Series):
                column_annotation = column_annotation.copy()
                if not len(column_annotation.index & df.columns):  # Series but without proper index
                    column_annotation.index = df.columns
            else:
                column_annotation = Series(column_annotation, index=df.columns)

            column_annotation.sort_values(inplace=True)
            df = df.ix[:, column_annotation.index]

    if axis_to_sort in (0, 1):
        a = array(df)
        a.sort(axis=axis_to_sort)
        df = DataFrame(a, index=df.index)

    elif cluster:
        row_indices, column_indices = get_dendrogram_leaf_indices(dataframe)
        df = df.iloc[row_indices, column_indices]
        if isinstance(row_annotation, Series):
            row_annotation = row_annotation.iloc[row_indices]
        if isinstance(column_annotation, Series):
            column_annotation = column_annotation.iloc[column_indices]
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

    if data_type == 'continuous':  # Plot colorbar
        cax, kw = make_axes(ax_bottom, location='bottom', fraction=0.16,
                            cmap=CMAP_CONTINUOUS,
                            norm=Normalize(values.min(), values.max()),
                            ticks=[values.min(), values.mean(), values.max()])
        ColorbarBase(cax, **kw)
        decorate(ax=cax)

    elif data_type in ('categorical', 'binary'):  # Plot category legends
        if len(values) < 30:
            horizontal_span = ax_center.axis()[1]
            vertival_span = ax_center.axis()[3]
            for i, v in enumerate(values):
                x = (horizontal_span / len(values) / 2) + i * horizontal_span / len(values)
                y = 0 - vertival_span * 0.09
                ax_center.plot(x, y, 'o', markersize=16, aa=True, clip_on=False)
                ax_center.text(x, y - vertival_span * 0.05, v, horizontalalignment='center', **FONT)

    decorate(title=title, xlabel=xlabel, ylabel=ylabel,
             xlabel_rotation=xlabel_rotation, ylabel_rotation=ylabel_rotation,
             max_xtick_size=10,
             ax=ax_center)

    if len(row_annotation):
        if len(set(row_annotation)) <= 2:
            cmap = CMAP_BINARY
        else:
            if len(annotation_colors):
                cmap = ListedColormap(annotation_colors)
            else:
                cmap = CMAP_CATEGORICAL
        heatmap(DataFrame(row_annotation), ax=ax_right, cbar=False, xticklabels=False, yticklabels=False, cmap=cmap)

    if len(column_annotation):
        if len(set(column_annotation)) <= 2:
            cmap = CMAP_BINARY
        else:
            if len(annotation_colors):
                cmap = ListedColormap(annotation_colors)
            else:
                cmap = CMAP_CATEGORICAL
        heatmap(DataFrame(column_annotation).T, ax=ax_top, cbar=False, xticklabels=False, yticklabels=False, cmap=cmap)

    save_plot(filepath)


def plot_clustermap(dataframe, pivot_kws=None, method='complete', metric='euclidean', z_score=None, standard_scale=None,
                    figsize=FIGURE_SIZE, cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None,
                    col_linkage=None,
                    row_colors=None, col_colors=None, annotate=False, mask=None, cmap=CMAP_CONTINUOUS,
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
    :param annotate: bool; show values in the matrix or not
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
                             row_colors=row_colors, col_colors=col_colors, annot=annotate, mask=mask, cmap=cmap,
                             xticklabels=xticklabels, yticklabels=yticklabels, **kwargs)

    ax_heatmap = clustergrid.ax_heatmap

    decorate(title=title, xlabel=xlabel, ylabel=ylabel, ax=ax_heatmap)

    save_plot(filepath)


def plot_nmf(nmf_results=None, k=None, w_matrix=None, h_matrix=None, max_std=3, pdf=None, filepath=None):
    """
    Plot nmf_results dictionary (can be generated by ccal.machine_learning.matrix_decompose.nmf function).
    :param nmf_results: dict; {k: {w:w, h:h, e:error}}
    :param k: int; k for NMF
    :param w_matrix: DataFrame
    :param h_matrix: DataFrame
    :param max_std: number; threshold to clip standardized values
    :param pdf: PdfPages;
    :param filepath: str;
    :return: None
    """

    # Check for W and H matrix
    if isinstance(nmf_results, dict) and k:
        w_matrix = nmf_results[k]['w']
        h_matrix = nmf_results[k]['h']
    elif not (isinstance(w_matrix, DataFrame) and isinstance(h_matrix, DataFrame)):
        raise ValueError('Need either: 1) NMFCC result ({k: {w:w, h:h, e:error}) and k; or 2) W & H matrices.')

    # Initialize a PDF
    if pdf:  # Specified pdf will be closed by the caller
        close_pdf = False

    elif filepath:  # Initialize a pdf and close it later
        establish_filepath(filepath)
        if not filepath.endswith('.pdf'):
            filepath += '.pdf'
        pdf = PdfPages(filepath)
        close_pdf = True

    # Initialize a figure
    plt.figure(figsize=FIGURE_SIZE)

    # Plot W matrix
    plot_heatmap(w_matrix, cluster=True,
                 title='W Matrix for k={}'.format(w_matrix.shape[1]), xlabel='Component', ylabel='Feature',
                 yticklabels=False, normalization_method='-0-', normalization_axis=0, max_std=max_std)
    if pdf:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    # Plot cluster map for H
    plot_heatmap(h_matrix, cluster=True,
                 title='H Matrix for k={}'.format(h_matrix.shape[0]), xlabel='Sample', ylabel='Component',
                 xticklabels=False, normalization_method='-0-', normalization_axis=1, max_std=max_std)
    if pdf:
        plt.savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    if pdf and close_pdf:
        pdf.close()


# TODO: enable str states
def assign_colors_to_states(states, colors=()):
    """
    Assign colors to states
    :param states: int or iterable; number of states or iterable of int representing state
    :param colors:
    :return: dict; {state: color}
    """

    if isinstance(states, int):  # Number of states
        unique_states = range(1, states + 1)
    elif any(states):  # Iterable of int representing state
        unique_states = sorted(set(states))
    else:
        raise ValueError('Error with states.')

    if isinstance(colors, ListedColormap) or isinstance(colors, LinearSegmentedColormap):  # Use given colormap
        colors = [colors(s) for s in unique_states]

    elif len(colors):  # Use given colors to make a colormap
        color_converter = ColorConverter()
        colors = [tuple(c) for c in color_converter.to_rgba_array(colors)]

    else:  # Use categorical colormap
        colors = [CMAP_CATEGORICAL(int(s / max(unique_states) * CMAP_CATEGORICAL.N)) for s in unique_states]

    state_colors = {}
    for i, s in enumerate(unique_states):
        state_colors[s] = colors[i]

    return state_colors


def decorate(style=None,
             title=None,
             xlabel=None, ylabel=None,
             xlabel_rotation=0, ylabel_rotation=90,
             xticks=None, yticks=None,
             max_n_xticks=80, max_n_yticks=50,
             max_xtick_size=None, max_ytick_size=None,
             ax=None):
    """

    :param style: str;
    :param title: str;
    :param xlabel: str;
    :param ylabel: str;
    :param xlabel_rotation:
    :param ylabel_rotation:
    :param xticks: iterable;
    :param yticks: iterable;
    :param max_n_xticks:
    :param max_n_yticks:
    :param max_xtick_size:
    :param max_ytick_size:
    :param ax: Axes; sets the current ax to be ax
    :return: None
    """

    # Set ax
    if not ax:
        ax = plt.gca()
    else:
        plt.sca(ax)

    # Set global plot aesthetics
    if style:
        set_style(style)
        if style == 'ticks':
            despine(top=True, right=True)

    # Title
    if title:
        plt.suptitle(title, **FONT_TITLE)

    # Label x axis
    if not xlabel:
        xlabel = ax.get_xlabel()
    ax.set_xlabel(xlabel, rotation=xlabel_rotation, **FONT_SUBTITLE)

    # Label y axis
    if not ylabel:
        ylabel = ax.get_ylabel()
    ax.set_ylabel(ylabel, rotation=ylabel_rotation, **FONT_SUBTITLE)

    # Label x ticks
    if not xticks:
        xticks = [t.get_text() for t in ax.get_xticklabels()]

    if len(xticks):
        if xticks[0] == '':
            xticks = ax.get_xticks()

        # if isinstance(xticks[0], float):
        #     xticks = ['{:.3f}'.format(t) for t in xticks]

        if max_n_xticks < len(xticks):
            xticks = []

        if max_xtick_size:
            xticks = [t[:max_xtick_size] for t in xticks]
        ax.set_xticklabels(xticks, **FONT)

    # Label y ticks
    if not yticks:
        yticks = [t.get_text() for t in ax.get_yticklabels()]

    if len(yticks):
        if yticks[0] == '':
            yticks = ax.get_yticks()

        # if isinstance(yticks[0], float):
        #     yticks = ['{:.3f}'.format(t) for t in yticks]

        if max_n_yticks < len(yticks):
            yticks = []

        if max_ytick_size:
            yticks = [t[:max_ytick_size] for t in yticks]
        ax.set_yticklabels(yticks, **FONT)


def save_plot(filepath, overwrite=True, dpi=DPI):
    """
    Establish filepath and save plot (.pdf) at dpi resolution.
    :param filepath: str;
    :param suffix: str;
    :param overwrite: bool;
    :param dpi: int;
    :return: None
    """
    if filepath:
        if not isfile(filepath) or overwrite:  # If the figure doesn't exist or overwriting
            establish_filepath(filepath)
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
