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

from math import pow

from numpy import asarray, unique, linspace
from pandas import DataFrame, Series, isnull
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.cm import bwr, Paired
from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.backends.backend_pdf import PdfPages
from seaborn import light_palette, heatmap, clustermap, pointplot, violinplot, boxplot

from .support import print_log, establish_path, get_unique_in_order, normalize_pandas_object, compute_score_and_pvalue

# ======================================================================================================================
# Set up glonal parameters
# ======================================================================================================================
# Color maps
BAD_COLOR = 'wheat'
CMAP_CONTINUOUS = bwr
CMAP_CONTINUOUS.set_bad(BAD_COLOR)
CMAP_CATEGORICAL = Paired
CMAP_CATEGORICAL.set_bad(BAD_COLOR)
CMAP_BINARY = light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad(BAD_COLOR)

# Figure parameters
FIGURE_SIZE = (16, 10)
DPI = 1000


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_clustermap(matrix, figure_size=FIGURE_SIZE, title=None, title_fontsize=20,
                    xticklabels=True, yticklabels=True, xticklabels_rotation=90, yticklabels_rotation=0,
                    row_colors=None, col_colors=None, filepath=None, dpi=DPI):
    """
    Plot heatmap for `matrix`.
    :param matrix: pandas DataFrame;
    :param figure_size: tuple; (n_rows, n_cols)
    :param title: str;
    :param title_fontsize: number;
    :param xticklabels_rotation: number;
    :param yticklabels_rotation: number;
    :param row_colors: list-like or pandas DataFrame/Series; List of colors to label for either the rows.
        Useful to evaluate whether samples within a group are clustered together. Can use nested lists or DataFrame for
        multiple color levels of labeling. If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param col_colors: list-like or pandas DataFrame/Series; List of colors to label for either the column.
        Useful to evaluate whether samples within a group are clustered together. Can use nested lists or DataFrame for
        multiple color levels of labeling. If given as a DataFrame or Series, labels for the colors are extracted from
        the DataFrames column names or from the name of the Series. DataFrame/Series colors are also matched to the data
        by their index, ensuring colors are drawn in the correct order.
    :param filepath: str;
    :param dpi: int;
    :return: None
    """
    plt.figure(figsize=figure_size)

    clustergrid = clustermap(matrix, xticklabels=xticklabels, yticklabels=yticklabels,
                             row_colors=row_colors, col_colors=col_colors, cmap=CMAP_CONTINUOUS)

    if title:
        figuretitle_font_properties = {'fontsize': title_fontsize, 'fontweight': 'bold'}
        plt.suptitle(title, **figuretitle_font_properties)

    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_rotation(xticklabels_rotation)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_rotation(yticklabels_rotation)

    if filepath:
        save_plot(filepath, dpi=dpi)


def plot_clusterings(matrix, figure_size=FIGURE_SIZE, title='Clustering Labels', title_fontsize=20, filepath=None,
                     dpi=DPI):
    """
    Plot clustering matrix.
    :param matrix: pandas DataFrame; (n_clusterings, n_samples)
    :param figure_size: tuple; (n_rows, n_cols)
    :param title: str;
    :param title_fontsize: number;
    :param filepath: str;
    :param dpi: int;
    :return: None
    """

    a = asarray(matrix)
    a.sort()

    plt.figure(figsize=figure_size)
    heatmap(DataFrame(a, index=matrix.index), cmap=CMAP_CATEGORICAL, xticklabels=False,
            cbar_kws={'orientation': 'horizontal'})

    if title:
        figuretitle_font_properties = {'fontsize': title_fontsize}
        plt.suptitle(title, **figuretitle_font_properties)

    plt.gca().set_xlabel('Sample')

    for t in plt.gca().get_yticklabels():
        t.set_weight('bold')

    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks(list(range(1, a.max() + 1)))

    if filepath:
        save_plot(filepath, dpi=dpi)


def plot_nmf_result(nmf_results=None, k=None, w_matrix=None, h_matrix=None, normalize=False, max_std=3,
                    figure_size=FIGURE_SIZE, title=None, title_fontsize=20, filepath=None, dpi=DPI):
    """
    Plot `nmf_results` dictionary (can be generated by `ccal.analyze.nmf` function).
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param k: int; k for NMF
    :param w_matrix: pandas DataFrame
    :param h_matrix: Pandas DataFrame
    :param normalize: bool; normalize W and H matrices or not ('-0-' normalization on the component axis)
    :param max_std: number; threshold to clip standardized values
    :param figure_size: tuple; (n_rows, n_cols)
    :param title: str;
    :param title_fontsize: number;
    :param filepath: str;
    :param dpi: int;
    :return: None
    """
    # Check for W and H matrix
    if isinstance(nmf_results, dict) and k:
        w_matrix = nmf_results[k]['W']
        h_matrix = nmf_results[k]['H']
    elif not (isinstance(w_matrix, DataFrame) and isinstance(h_matrix, DataFrame)):
        raise ValueError('Need either: 1) NMF result ({k: {W:w, H:h, ERROR:error}) and k; or 2) W and H matrices.')

    # Initialize a PDF
    if filepath:
        establish_path(filepath)
        if not filepath.endswith('.pdf'):
            filepath += '.pdf'
        pdf = PdfPages(filepath)

    # Set up aesthetic properties
    figuretitle_font_properties = {'fontsize': title_fontsize, 'fontweight': 'bold'}
    axtitle_font_properties = {'fontsize': title_fontsize * 0.9, 'fontweight': 'bold'}
    label_font_properties = {'fontsize': title_fontsize * 0.81, 'fontweight': 'bold'}

    # Plot W and H
    plt.figure(figsize=figure_size)
    gridspec = GridSpec(10, 16)
    ax_w = plt.subplot(gridspec[1:, :5])
    ax_h = plt.subplot(gridspec[3:8, 7:])
    if not title:
        title = 'NMF Result for k={}'.format(w_matrix.shape[1])
    plt.suptitle(title, **figuretitle_font_properties)
    # Plot W
    if normalize:
        w_matrix = normalize_pandas_object(w_matrix, method='-0-', axis=0).clip(-max_std, max_std)
    heatmap(w_matrix, cmap=CMAP_CONTINUOUS, yticklabels=False, ax=ax_w)
    ax_w.set_title('W Matrix', **axtitle_font_properties)
    ax_w.set_xlabel('Component', **label_font_properties)
    ax_w.set_ylabel('Feature', **label_font_properties)
    # Plot H
    if normalize:
        h_matrix = normalize_pandas_object(h_matrix, method='-0-', axis=1).clip(-max_std, max_std)
    heatmap(h_matrix, cmap=CMAP_CONTINUOUS, xticklabels=False, cbar_kws={'orientation': 'horizontal'}, ax=ax_h)
    ax_h.set_title('H Matrix', **axtitle_font_properties)
    ax_h.set_xlabel('Sample', **label_font_properties)
    ax_h.set_ylabel('Component', **label_font_properties)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=dpi, bbox_inches='tight')

    # Plot cluster map for W
    clustergrid = clustermap(w_matrix, standard_scale=0, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('W Matrix for k={}'.format(w_matrix.shape[1]), **figuretitle_font_properties)
    clustergrid.ax_heatmap.set_xlabel('Component', **label_font_properties)
    clustergrid.ax_heatmap.set_ylabel('Feature', **label_font_properties)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_fontweight('bold')
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_visible(False)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=dpi, bbox_inches='tight')

    # Plot cluster map for H
    clustergrid = clustermap(h_matrix, standard_scale=1, figsize=FIGURE_SIZE, cmap=CMAP_CONTINUOUS)
    plt.suptitle('H Matrix for k={}'.format(w_matrix.shape[1]), **figuretitle_font_properties)
    clustergrid.ax_heatmap.set_xlabel('Sample', **label_font_properties)
    clustergrid.ax_heatmap.set_ylabel('Component', **label_font_properties)
    for t in clustergrid.ax_heatmap.get_xticklabels():
        t.set_visible(False)
    for t in clustergrid.ax_heatmap.get_yticklabels():
        t.set_fontweight('bold')
        t.set_rotation(0)
    if filepath:
        plt.savefig(pdf, format='pdf', dpi=dpi, bbox_inches='tight')

    if filepath:
        pdf.close()


def plot_clustering_scores(scores, figure_size=FIGURE_SIZE, title='Clustering Score vs. k', title_fontsize=20,
                           filepath=None, dpi=DPI):
    """
    Plot `scores` dictionary.
    :param scores: dict or pandas DataFrame; {k: score_dataframe_against_series}
    :param figure_size: tuple;
    :param title: str;
    :param title_fontsize: number;
    :param filepath: str;
    :param dpi: int;
    :return: None
    """
    if isinstance(scores, DataFrame):
        scores = scores.to_dict()
        scores = scores.popitem()[1]

    plt.figure(figsize=figure_size)

    if title:
        plt.suptitle(title, fontsize=title_fontsize, fontweight='bold')

    pointplot(x=[k for k, v in scores.items()], y=[v for k, v in scores.items()])

    label_font_properties = {'fontsize': title_fontsize * 0.81, 'fontweight': 'bold'}
    plt.gca().set_xlabel('k', **label_font_properties)
    plt.gca().set_ylabel('Score', **label_font_properties)

    if filepath:
        save_plot(filepath, dpi=dpi)


def plot_onco_gps(component_coordinates, samples, grid_probabilities, grid_states, n_states_train,
                  annotations=(), annotation_name='', annotation_type='continuous', std_max=3,
                  figure_size=FIGURE_SIZE, title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
                  subtitle_fontsize=16, subtitle_fontcolor='#FF0039', colors=None,
                  component_markersize=13, component_markerfacecolor='#000726', component_markeredgewidth=1.69,
                  component_markeredgecolor='#FFFFFF', component_text_position='auto', component_fontsize=16,
                  delaunay_linewidth=1, delaunay_linecolor='#000000',
                  n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
                  background_markersize=5.55, background_mask_markersize=7, background_max_alpha=0.9,
                  sample_markersize=12, sample_without_annotation_markerfacecolor='#999999',
                  sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
                  legend_markersize=10, legend_fontsize=11,
                  effectplot_type='violine', effectplot_mean_markerfacecolor='#FFFFFF',
                  effectplot_mean_markeredgecolor='#FF0082', effectplot_median_markeredgecolor='#FF0082',
                  filepath=None, dpi=DPI):
    """
    :param component_coordinates: pandas DataFrame; (n_components, [x, y])
    :param samples: pandas DataFrame; (n_samples, [x, y, state, annotation])
    :param grid_probabilities: numpy 2D array; (n_grids, n_grids)
    :param grid_states: numpy 2D array; (n_grids, n_grids)
    :param n_states_train: int; number of states used to create Onco-GPS
    :param annotations: pandas Series; (n_samples); sample annotations; will color samples based on annotations
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param std_max: number; threshold to clip standardized values
    :param figure_size: tuple;
    :param title: str;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib color;
    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib color;
    :param colors: matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap, or list;
    :param component_markersize: number;
    :param component_markerfacecolor: matplotlib color;
    :param component_markeredgewidth: number;
    :param component_markeredgecolor: matplotlib color;
    :param component_text_position: str; {'auto', 'top', 'bottom'}
    :param component_fontsize: number;
    :param delaunay_linewidth: number;
    :param delaunay_linecolor: matplotlib color;
    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]
    :param background_markersize: number; set to 0 to disable drawing backgrounds
    :param background_mask_markersize: number; set to 0 to disable masking
    :param background_max_alpha: float; [0, 1]; the maximum background alpha (transparency)
    :param sample_markersize: number;
    :param sample_without_annotation_markerfacecolor: matplotlib color;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param legend_markersize: number;
    :param legend_fontsize: number;
    :param effectplot_type: str; {'violine', 'box'}
    :param effectplot_mean_markerfacecolor: matplotlib color;
    :param effectplot_mean_markeredgecolor: matplotlib color;
    :param effectplot_median_markeredgecolor: matplotlib color;
    :param filepath: str;
    :param dpi: int;
    :return: None
    """
    x_grids = linspace(0, 1, grid_probabilities.shape[0])
    y_grids = linspace(0, 1, grid_probabilities.shape[1])

    # Set up figure and axes
    plt.figure(figsize=figure_size)
    gridspec = GridSpec(10, 16)
    ax_title = plt.subplot(gridspec[0, :7])
    ax_title.axis([0, 1, 0, 1])
    ax_title.axis('off')
    ax_colorbar = plt.subplot(gridspec[0, 7:12])
    ax_colorbar.axis([0, 1, 0, 1])
    ax_colorbar.axis('off')
    ax_map = plt.subplot(gridspec[1:, :12])
    ax_map.axis([0, 1, 0, 1])
    ax_map.axis('off')
    ax_legend = plt.subplot(gridspec[1:, 14:])
    ax_legend.axis('off')

    # Plot title
    ax_title.text(0, 0.9, title, fontsize=title_fontsize, color=title_fontcolor, weight='bold')
    ax_title.text(0, 0.39,
                  '{} samples, {} components, and {} states'.format(samples.shape[0], component_coordinates.shape[0],
                                                                    n_states_train),
                  fontsize=subtitle_fontsize, color=subtitle_fontcolor, weight='bold')

    # Plot components and their labels
    ax_map.plot(component_coordinates.ix[:, 'x'], component_coordinates.ix[:, 'y'], marker='D', linestyle='',
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor, clip_on=False,
                aa=True, zorder=6)
    # Compute convexhull
    convexhull = ConvexHull(component_coordinates)
    convexhull_region = Path(convexhull.points[convexhull.vertices])
    # Put labels on top or bottom of the component markers
    component_text_verticalshift = -0.03
    for i in component_coordinates.index:
        if component_text_position == 'auto':

            if convexhull_region.contains_point((component_coordinates.ix[i, 'x'],
                                                 component_coordinates.ix[i, 'y'] + component_text_verticalshift)):
                component_text_verticalshift *= -1
        elif component_text_position == 'top':
            component_text_verticalshift *= -1
        elif component_text_position == 'bottom':
            pass
        x, y = component_coordinates.ix[i, 'x'], component_coordinates.ix[i, 'y'] + component_text_verticalshift

        ax_map.text(x, y, i,
                    fontsize=component_fontsize, color=component_markerfacecolor, weight='bold',
                    horizontalalignment='center', verticalalignment='center', zorder=6)

    # Plot Delaunay triangulation
    delaunay = Delaunay(component_coordinates)
    ax_map.triplot(delaunay.points[:, 0], delaunay.points[:, 1], delaunay.simplices.copy(),
                   linewidth=delaunay_linewidth, color=delaunay_linecolor, aa=True, zorder=4)

    # Plot contours
    if n_contours > 0:
        ax_map.contour(x_grids, y_grids, grid_probabilities, n_contours, corner_mask=True,
                       linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha, aa=True, zorder=2)

    # Assign colors to states
    if colors:
        if not (isinstance(colors, ListedColormap) and isinstance(colors, LinearSegmentedColormap)):
            colors = ListedColormap(colors)
    states_color = {}
    for s in range(1, n_states_train + 1):
        if colors:
            states_color[s] = colors(s)
        else:
            states_color[s] = CMAP_CATEGORICAL(int(s / n_states_train * CMAP_CATEGORICAL.N))

    # Plot background
    if background_markersize > 0:
        grid_probabilities_min = grid_probabilities.min()
        grid_probabilities_max = grid_probabilities.max()
        grid_probabilities_range = grid_probabilities_max - grid_probabilities_min
        for i in range(grid_probabilities.shape[0]):
            for j in range(grid_probabilities.shape[1]):
                if convexhull_region.contains_point((x_grids[i], y_grids[j])):
                    c = states_color[grid_states[i, j]]
                    a = min(background_max_alpha,
                            (grid_probabilities[i, j] - grid_probabilities_min) / grid_probabilities_range)
                    ax_map.plot(x_grids[i], y_grids[j], marker='s', markersize=background_markersize, markerfacecolor=c,
                                alpha=a, aa=True, zorder=1)
    # Plot background mask
    if background_mask_markersize > 0:
        for i in range(grid_probabilities.shape[0]):
            for j in range(grid_probabilities.shape[1]):
                if not convexhull_region.contains_point((x_grids[i], y_grids[j])):
                    ax_map.plot(x_grids[i], y_grids[j], marker='s', markersize=background_mask_markersize,
                                markerfacecolor='w', aa=True, zorder=3)

    if any(annotations):  # Plot samples, annotations, sample legends, and annotation legends
        # Set up annotations
        a = Series(annotations)
        a.index = samples.index
        # Set up annotation min, mean, max, and colormap.
        if annotation_type == 'continuous':
            samples.ix[:, 'annotation'] = normalize_pandas_object(a, method='-0-').clip(-std_max, std_max)
            annotation_min = max(-std_max, samples.ix[:, 'annotation'].min())
            annotation_mean = samples.ix[:, 'annotation'].mean()
            annotation_max = min(std_max, samples.ix[:, 'annotation'].max())
            cmap = CMAP_CONTINUOUS
        else:
            samples.ix[:, 'annotation'] = annotations
            annotation_min = 0
            annotation_mean = int(samples.ix[:, 'annotation'].mean())
            annotation_max = int(samples.ix[:, 'annotation'].max())
            if annotation_type == 'categorical':
                cmap = CMAP_CATEGORICAL
            elif annotation_type == 'binary':
                cmap = CMAP_BINARY
            else:
                raise ValueError('Unknown annotation_type {}.'.format(annotation_type))
        annotation_range = annotation_max - annotation_min
        # Plot annotated samples
        for idx, s in samples.iterrows():
            if isnull(s.ix['annotation']):
                c = sample_without_annotation_markerfacecolor
            else:
                if annotation_type == 'continuous':
                    c = cmap(s.ix['annotation'])
                elif annotation_type in ('categorical', 'binary'):
                    c = cmap((s.ix['annotation'] - annotation_min) / annotation_range)
                else:
                    raise ValueError('Unknown annotation_type {}.'.format(annotation_type))
            if 'pullratio' in samples.columns:
                a = samples.ix[idx, 'pullratio']
            else:
                a = 1
            ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c, alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                        zorder=5)
            if a < 1:
                ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor='none',
                            markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                            zorder=5)
        # Plot sample legends
        ax_legend.axis('on')
        ax_legend.patch.set_visible(False)
        score, p_val = compute_score_and_pvalue(samples.ix[:, 'state'], annotations)
        ax_legend.set_title('{}\nIC={:.3f} (p-val={:.3f})'.format(annotation_name, score, p_val),
                            fontsize=legend_fontsize * 1.26, weight='bold')
        # Plot effect plot
        if effectplot_type == 'violine':
            violinplot(x=samples.ix[:, 'annotation'], y=samples.ix[:, 'state'], palette=states_color, scale='count',
                       inner=None, orient='h', ax=ax_legend, clip_on=False)
            boxplot(x=samples.ix[:, 'annotation'], y=samples.ix[:, 'state'], showbox=False, showmeans=True,
                    medianprops={'marker': 'o',
                                 'markerfacecolor': effectplot_mean_markerfacecolor,
                                 'markeredgewidth': 0.9,
                                 'markeredgecolor': effectplot_mean_markeredgecolor},
                    meanprops={'color': effectplot_median_markeredgecolor}, orient='h', ax=ax_legend)
        elif effectplot_type == 'box':
            boxplot(x=samples.ix[:, 'annotation'], y=samples.ix[:, 'state'], palette=states_color, showmeans=True,
                    medianprops={'marker': 'o',
                                 'markerfacecolor': effectplot_mean_markerfacecolor,
                                 'markeredgewidth': 0.9,
                                 'markeredgecolor': effectplot_mean_markeredgecolor},
                    meanprops={'color': effectplot_median_markeredgecolor}, orient='h', ax=ax_legend)
        else:
            raise ValueError('Unknown effectplot_type {}. effectplot_type = [\'violine\', \'box\'].')
        # Set up x label, ticks, and lines
        ax_legend.set_xlabel('')
        ax_legend.set_xticks([annotation_min, annotation_mean, annotation_max])
        for t in ax_legend.get_xticklabels():
            t.set(rotation=90, size=legend_fontsize * 0.9, weight='bold')
        ax_legend.axvline(annotation_min, color='#000000', ls='-', alpha=0.16, aa=True, clip_on=False)
        ax_legend.axvline(annotation_mean, color='#000000', ls='-', alpha=0.39, aa=True, clip_on=False)
        ax_legend.axvline(annotation_max, color='#000000', ls='-', alpha=0.16, aa=True, clip_on=False)
        # Set up y label, ticks, and lines
        ax_legend.set_ylabel('')
        ax_legend.set_yticklabels(
            ['State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)) for s in range(1, n_states_train + 1)],
            fontsize=legend_fontsize, weight='bold')
        ax_legend.yaxis.tick_right()
        # Plot sample markers
        l, r = ax_legend.axis()[:2]
        x = l - float((r - l) / 5)
        for i, s in enumerate(range(1, n_states_train + 1)):
            c = states_color[s]
            ax_legend.plot(x, i, marker='o', markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)
        # Plot colorbar
        if annotation_type == 'continuous':
            cax, kw = make_axes(ax_colorbar, location='top', fraction=0.39, shrink=1, aspect=16,
                                cmap=cmap, norm=Normalize(vmin=annotation_min, vmax=annotation_max),
                                ticks=[annotation_min, annotation_mean, annotation_max])
            ColorbarBase(cax, **kw)

    else:  # Plot samples and sample legends
        ax_legend.axis([0, 1, 0, 1])
        # Plot samples
        for idx, s in samples.iterrows():
            c = states_color[s.ix['state']]
            if 'pullratio' in samples.columns:
                a = samples.ix[idx, 'pullratio']
            else:
                a = 1
            ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c, alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                        zorder=5)
            if a < 1:
                ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor='none',
                            markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                            zorder=5)
        # Plot sample legends
        for i, s in enumerate(range(1, n_states_train + 1)):
            y = 1 - float(1 / (n_states_train + 1)) * (i + 1)
            c = states_color[s]
            ax_legend.plot(0.16, y, marker='o', markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)
            ax_legend.text(0.26, y, 'State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)),
                           fontsize=legend_fontsize, weight='bold', verticalalignment='center')

    if filepath:
        save_plot(filepath, dpi=dpi)


def plot_features_against_target(features, ref, annotations, feature_type='continuous', target_type='continuous',
                                 std_max=3, figure_size='auto', title=None, title_size=20,
                                 annotation_header=None, annotation_label_size=9,
                                 plot_colname=False, filepath=None, dpi=DPI):
    """
    Plot a heatmap panel.
    :param features: pandas DataFrame; (n_features, n_elements); must have indices and columns
    :param ref: pandas Series; (n_elements); must have indices, which must match `features`'s columns
    :param annotations:  pandas DataFrame; (n_features, n_annotations); must have indices, which must match `features`'s
    :param feature_type: str; {'continuous', 'categorical', 'binary'}
    :param target_type: str; {'continuous', 'categorical', 'binary'}
    :param std_max: number;
    :param figure_size: 'auto' or tuple;
    :param title: str;
    :param title_size: number;
    :param annotation_header: str; annotation header to be plotted
    :param annotation_label_size: number;
    :param plot_colname: bool; plot column names or not
    :param filepath: str;
    :param dpi: int;
    :return: None
    """
    if feature_type == 'continuous':
        features_cmap = CMAP_CONTINUOUS
        features_min, features_max = -std_max, std_max
        print_log('Normalizing continuous features ...')
        features = normalize_pandas_object(features, method='-0-', axis=1)
    elif feature_type == 'categorical':
        features_cmap = CMAP_CATEGORICAL
        features_min, features_max = 0, len(unique(features))
    elif feature_type == 'binary':
        features_cmap = CMAP_BINARY
        features_min, features_max = 0, 1
    else:
        raise ValueError('Unknown feature_type {}.'.format(feature_type))
    if target_type == 'continuous':
        ref_cmap = CMAP_CONTINUOUS
        ref_min, ref_max = -std_max, std_max
        print_log('Normalizing continuous ref ...')
        ref = normalize_pandas_object(ref, method='-0-')
    elif target_type == 'categorical':
        ref_cmap = CMAP_CATEGORICAL
        ref_min, ref_max = 0, len(unique(ref))
    elif target_type == 'binary':
        ref_cmap = CMAP_BINARY
        ref_min, ref_max = 0, 1
    else:
        raise ValueError('Unknown ref_type {}.'.format(target_type))

    if figure_size == 'auto':
        figure_size = (min(pow(features.shape[1], 0.7), 7), pow(features.shape[0], 0.9))
    plt.figure(figsize=figure_size)
    gridspec = GridSpec(features.shape[0] + 1, features.shape[1] + 1)
    ax_ref = plt.subplot(gridspec[:1, :features.shape[1]])
    ax_features = plt.subplot(gridspec[1:, :features.shape[1]])
    ax_annotation_header = plt.subplot(gridspec[:1, features.shape[1]:])
    ax_annotation_header.axis('off')
    horizontal_text_margin = pow(features.shape[1], 0.39)

    # Plot ref, ref label, and title,
    heatmap(DataFrame(ref).T, ax=ax_ref, vmin=ref_min, vmax=ref_max, cmap=ref_cmap, xticklabels=False, cbar=False)
    for t in ax_ref.get_yticklabels():
        t.set(rotation=0, weight='bold')

    if title:
        ax_ref.text(features.shape[1] / 2, 1.9, title, horizontalalignment='center', size=title_size, weight='bold')

    if target_type in ('binary', 'categorical'):
        # Add binary or categorical ref labels
        boundaries = [0]
        prev_v = ref.iloc[0]
        for i, v in enumerate(ref.iloc[1:]):
            if prev_v != v:
                boundaries.append(i + 1)
            prev_v = v
        boundaries.append(features.shape[1])
        label_horizontal_positions = []
        prev_b = 0
        for b in boundaries[1:]:
            label_horizontal_positions.append(b - (b - prev_b) / 2)
            prev_b = b
        unique_ref_labels = get_unique_in_order(ref.values)

        for i, pos in enumerate(label_horizontal_positions):
            ax_ref.text(pos, 1, unique_ref_labels[i], horizontalalignment='center', weight='bold')

    # Plot features
    heatmap(features, ax=ax_features, vmin=features_min, vmax=features_max, cmap=features_cmap,
            xticklabels=plot_colname, cbar=False)
    for t in ax_features.get_yticklabels():
        t.set(rotation=0, weight='bold')

    # Plot annotations
    if not annotation_header:
        annotation_header = '\t'.join(annotations.columns).expandtabs()
    ax_annotation_header.text(horizontal_text_margin, 0.5, annotation_header, horizontalalignment='left',
                              verticalalignment='center', size=annotation_label_size, weight='bold')
    for i, (idx, s) in enumerate(annotations.iterrows()):
        ax = plt.subplot(gridspec[i + 1:i + 2, features.shape[1]:])
        ax.axis('off')
        a = '\t'.join(s.tolist()).expandtabs()
        ax.text(horizontal_text_margin, 0.5, a, horizontalalignment='left', verticalalignment='center',
                size=annotation_label_size, weight='bold')

    if filepath:
        save_plot(filepath, dpi=dpi)


def save_plot(filepath, dpi=DPI):
    """
    Save plot.
    :param filepath: str;
    :param dpi: int;
    :return: None
    """
    establish_path(filepath)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
