"""
Computational Cancer Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov


Description:
Plotting module for CCAL.
"""
import os
import math

import numpy as np
import pandas as pd
from sklearn import manifold
from scipy.spatial import Delaunay, ConvexHull
import matplotlib as mpl
import matplotlib.pyplot as plt
from seaborn import light_palette, heatmap, pointplot, violinplot, boxplot

import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr

from .support import SEED, print_log, establish_path, normalize_pandas_object

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')

# ======================================================================================================================
# Parameters
# ======================================================================================================================
# Color maps
BAD_COLOR = 'wheat'
CMAP_CONTINUOUS = mpl.cm.bwr
CMAP_CONTINUOUS.set_bad(BAD_COLOR)
CMAP_CATEGORICAL = mpl.cm.Paired
CMAP_CATEGORICAL.set_bad(BAD_COLOR)
CMAP_BINARY = light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad(BAD_COLOR)

FIGURE_SIZE = (10, 16)
DPI = 1000


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_nmf_result(nmf_results, k, title='NMF Result', title_fontsize=20,
                    output_filename=None, figure_size=FIGURE_SIZE, dpi=DPI):
    """
    Plot NMF results from cca.library.cca.nmf function.
    :param nmf_results: dict, result per k (key: k; value: dict(key: w, h, err; value: w matrix, h matrix, and error))
    :param k: int, k for NMF
    :param figure_size: tuple (width, height),
    :param title: str, figure title
    :param title_fontsize: int, title-font size
    :param output_filename: str, file path to save the figure
    :param dpi: int, dots-per-inch for the output figure
    :return: None
    """
    # Plot W and H
    fig, (ax_w, ax_h) = plt.subplots(1, 2, figure_size=figure_size)
    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight='bold')

    heatmap(normalize_pandas_object(nmf_results[k]['W']), cmap='bwr', yticklabels=False, ax=ax_w)
    ax_w.set_title('W matrix generated using k={}'.format(k), fontsize=title_fontsize * 0.9, fontweight='bold')
    ax_w.set_xlabel('Component', fontsize=title_fontsize * 0.69, fontweight='bold')
    ax_w.set_ylabel('Feature', fontsize=title_fontsize * 0.69, fontweight='bold')

    heatmap(normalize_pandas_object(nmf_results[k]['H']), cmap='bwr', xticklabels=False, ax=ax_h)
    ax_h.set_title('H matrix generated using k={}'.format(k), fontsize=title_fontsize * 0.9, fontweight='bold')
    ax_h.set_xlabel('Sample', fontsize=title_fontsize * 0.69, fontweight='bold')
    ax_h.set_ylabel('Component', fontsize=title_fontsize * 0.69, fontweight='bold')

    if output_filename:
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')

    plt.show()


def plot_nmf_scores(scores, title='NMF Clustering Score vs. k', title_fontsize=20, output_filename=None,
                    figure_size=(10, 10), dpi=DPI):
    """
    Plot NMF `scores`.
    :param scores: dict, NMF score per k (key: k; value: score)
    :param figure_size: tuple, figure size (width, height)
    :param title: str, figure title
    :param title_fontsize: int, title-font size
    :param output_filename: str, file path to save the figure
    :param dpi: int, dots-per-inch for the output figure
    :return: None
    """
    plt.figure(figure_size=figure_size)
    ax = pointplot(x=[k for k, v in scores.items()], y=[v for k, v in scores.items()])
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel('k', fontsize=title_fontsize * 0.81, fontweight='bold')
    ax.set_ylabel('Score', fontsize=title_fontsize * 0.81, fontweight='bold')

    if output_filename:
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')

    plt.show()


def plot_features_against_reference(features, ref, annotations, features_type='continuous', ref_type='continuous',
                                    title=None, title_size=16, annotation_header=None, annotation_label_size=9,
                                    plot_colname=False, figure_filename=None, dpi=DPI):
    """
    Plot a heatmap panel.
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param ref: pandas Series (m_elements), must have indices, which must match 'features`'s columns
    :param annotations:  pandas DataFrame (n_features, n_annotations), must have indices, which must match 'features`'s
    :param features_type: str, {continuous, categorical, binary}
    :param ref_type: str, {continuous, categorical, binary}
    :param title: str, figure title
    :param title_size: int, title text size
    :param annotation_header: str, annotation header to be plotted
    :param annotation_label_size: int, annotation text size
    :param plot_colname: bool, plot column names or not
    :param figure_filename: str, file path prefix to save the figure
    :param dpi: int, dots-per-inch for the output figure
    :return: None
    """
    features_cmap, features_min, features_max = _setup_cmap(features, features_type)
    ref_cmap, ref_min, ref_max = _setup_cmap(ref, ref_type)

    # Normalize
    if features_type is 'continuous':
        print_log('Normalizing continuous features ...')
        features = normalize_pandas_object(features)
    if ref_type is 'continuous':
        print_log('Normalizing continuous ref ...')
        ref = (ref - ref.mean()) / ref.std()
        ref = normalize_pandas_object(ref)

    fig = plt.figure(figure_size=(min(math.pow(features.shape[1], 0.7), 7), math.pow(features.shape[0], 0.9)))
    horizontal_text_margin = math.pow(features.shape[1], 0.39)
    gridspec = plt.GridSpec(features.shape[0] + 1, features.shape[1] + 1)

    # Plot ref, ref label, and title,
    ref_ax = plt.subplot(gridspec.new_subplotspec((0, 0), colspan=features.shape[1]))
    heatmap(pd.DataFrame(ref).T, vmin=ref_min, vmax=ref_max, cmap=ref_cmap, xticklabels=False, cbar=False)
    plt.setp(ref_ax.get_yticklabels(), rotation=0)
    plt.setp(ref_ax.get_yticklabels(), weight='bold')

    if title:
        ref_ax.text(features.shape[1] / 2, 1.9, title, horizontalalignment='center', size=title_size, weight='bold')

    if ref_type in ('binary', 'categorical'):
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
        # TODO: implement get_unique_in_order
        unique_ref_labels = np.unique(ref.values)[::-1]
        for i, pos in enumerate(label_horizontal_positions):
            ref_ax.text(pos, 1, unique_ref_labels[i], horizontalalignment='center', weight='bold')

    # Plot features
    features_ax = plt.subplot(gridspec.new_subplotspec((1, 0), rowspan=features.shape[0], colspan=features.shape[1]))
    heatmap(features, vmin=features_min, vmax=features_max, cmap=features_cmap, xticklabels=plot_colname,
            cbar=False)
    plt.setp(features_ax.get_yticklabels(), rotation=0)
    plt.setp(features_ax.get_yticklabels(), weight='bold')

    # Plot annotations
    annotation_header_ax = plt.subplot(gridspec.new_subplotspec((0, features.shape[1])))
    annotation_header_ax.set_axis_off()
    if not annotation_header:
        annotation_header = '\t'.join(annotations.columns).expandtabs()
    annotation_header_ax.text(horizontal_text_margin, 0.5, annotation_header, horizontalalignment='left',
                              verticalalignment='center',
                              size=annotation_label_size, weight='bold')
    for i, (idx, s) in enumerate(annotations.iterrows()):
        ax = plt.subplot(gridspec.new_subplotspec((1 + i, features.shape[1])))
        ax.set_axis_off()
        a = '\t'.join(s.tolist()).expandtabs()
        ax.text(horizontal_text_margin, 0.5, a, horizontalalignment='left', verticalalignment='center',
                size=annotation_label_size, weight='bold')

    # fig.subplots_adjust(left=0.15, right=0.7)
    plt.show(fig)

    if figure_filename:
        establish_path(os.path.split(figure_filename)[0])
        fig.savefig(figure_filename, dpi=dpi, bbox_inches='tight')
        print_log('Saved the figure as {}.'.format(figure_filename))


def _setup_cmap(pandas_obj, data_type, std_max=3):
    if data_type is 'continuous':
        data_cmap = CMAP_CONTINUOUS
        data_min, data_max = -std_max, std_max
    elif data_type is 'categorical':
        data_cmap = CMAP_CATEGORICAL
        data_min, data_max = 0, np.unique(pandas_obj.values).size
    elif data_type is 'binary':
        data_cmap = CMAP_BINARY
        data_min, data_max = 0, 1
    else:
        raise ValueError('Unknown data_type {}.'.format(data_type))
    return data_cmap, data_min, data_max


def plot_onco_gps(h, states, max_std=3, annotations=None, annotation_type='continuous',
                  coordinates_extending_factor=1 / 26, n_grid=128,
                  title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
                  subtitle_fontsize=16, subtitle_fontcolor='#FF0039',
                  mds_is_metric=True, mds_seed=SEED,
                  component_markersize=13, component_markerfacecolor='#000726',
                  component_markeredgewidth=1, component_markeredgecolor='#FFFFFF', component_fontsize=16,
                  delaunay_linewidth=1, delaunay_linecolor='#000000',
                  kde_bandwidths_factor=0.69, n_respective_component='all', sample_stretch_factor=2,
                  sample_markersize=12, sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
                  contour=True, n_contour=30, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.5,
                  background=True, background_markersize=3.73, background_max_alpha=0.9, background_alpha_factor=0.69,
                  legend_markersize=10, legend_fontsize=11,
                  effectplot_type='violine', effectplot_mean_markerfacecolor='#FFFFFF',
                  effectplot_mean_markeredgecolor='#FF0082', effectplot_median_markeredgecolor='#FF0082',
                  output_filename=None, figure_size=FIGURE_SIZE, dpi=DPI):
    """
    :param h: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param states: array-like; (n_samples); sample states
    :param max_std: number;
    :param annotations: pandas Series; (n_samples); sample annotations; color samples based on annotations when given
    :param annotation_type: string; {'continuous', 'categorical', 'binary'}
    :param coordinates_extending_factor: number; factor to extend the map ax's coordinate dimensions
    :param n_grid: int;
    :param title: string;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib compatible color;
    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib compatible color;
    :param mds_is_metric: bool; use metric multidimensional scaling or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param component_markersize: number;
    :param component_markerfacecolor: matplotlib compatible color;
    :param component_markeredgewidth: number;
    :param component_markeredgecolor: matplotlib compatible color;
    :param component_fontsize: number;
    :param delaunay_linewidth: int or flaot;
    :param delaunay_linecolor: matplotlib compatible color;
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :param n_respective_component: int; [1, n_components]; number of components influencing a sample's coordinate
    :param sample_stretch_factor: number; power to raise components' influence on each sample
    :param sample_markersize: number;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib compatible color;
    :param contour: bool; plot contours or not
    :param n_contour: int;
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib compatible color;
    :param contour_alpha: float; [0, 1]
    :param background: bool; plot background or not
    :param background_markersize: number;
    :param background_max_alpha: float; [0, 1]; the maximum background alpha (transparency)
    :param background_alpha_factor: number; factor to multiply background alpha
    :param legend_markersize: number;
    :param legend_fontsize: number;
    :param effectplot_type: string; {'violine', 'box'}
    :param effectplot_mean_markerfacecolor:
    :param effectplot_mean_markeredgecolor:
    :param effectplot_median_markeredgecolor:
    :param output_filename: string;
    :param figure_size: tuple; (height, width)
    :param dpi: int;
    :return: None
    """
    unique_states = sorted(set(states))
    print_log('Creating Onco-GPS with {} samples, {} components, and states: {} ...'.format(*reversed(h.shape),
                                                                                            unique_states))

    # Standardize H and clip values with extreme standard deviation
    standardized_h = normalize_pandas_object(h)
    standardized_clipped_h = standardized_h.clip(-max_std, max_std)

    # Project the H's components from <n_sample>D to 2D, Delaunay triangulate the 2D-projected component coordinates,
    # and compute convexhull for the 2D-projected component coordinates
    mds = manifold.MDS(metric=mds_is_metric, random_state=mds_seed)
    components_coordinates = mds.fit_transform(standardized_clipped_h)
    delaunay = Delaunay(components_coordinates)
    convexhull = ConvexHull(components_coordinates)
    convexhull_region = mpl.path.Path(convexhull.points[convexhull.vertices])

    samples = pd.DataFrame(index=h.columns, columns=['state', 'x', 'y'])

    # Get sample states and (if any) annotations
    samples['state'] = states
    if annotations:
        if annotation_type == 'continuous':
            samples['annotation'] = normalize_pandas_object(annotations)
            samples.clip(-max_std, max_std)
        else:
            samples['annotation'] = normalize_pandas_object(annotations, method='0-1')

    # Get x & y coordinates of each sample
    for sample in samples.index:
        col = h.ix[:, sample]
        if n_respective_component == 'all':
            n_respective_component = h.shape[0]
        col = col.mask(col < col.sort_values()[-n_respective_component], other=0)

        x = sum(col ** sample_stretch_factor * components_coordinates[:, 0]) / sum(col ** sample_stretch_factor)
        y = sum(col ** sample_stretch_factor * components_coordinates[:, 1]) / sum(col ** sample_stretch_factor)
        samples.ix[sample, ['x', 'y']] = x, y

    # Get x & y coordinate boundaries
    x_min = min(components_coordinates[:, 0])
    x_max = max(components_coordinates[:, 0])
    x_margin = (x_max - x_min) * coordinates_extending_factor
    x_min -= x_margin
    x_max += x_margin
    y_min = min(components_coordinates[:, 1])
    y_max = max(components_coordinates[:, 1])
    y_margin = (y_max - y_min) * coordinates_extending_factor
    y_min -= y_margin
    y_max += y_margin

    # Make x & y grids
    x_grids = np.linspace(x_min, x_max, n_grid)
    y_grids = np.linspace(y_min, y_max, n_grid)

    # Get KDE for each state using bandwidth created from all states' x & y coordinates
    kdes = np.zeros((len(states), n_grid, n_grid))
    bandwidths = np.array([mass.bcv(np.array(samples.ix[:, 'x'].tolist()))[0],
                           mass.bcv(np.array(samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor
    for s in unique_states:
        coordiantes = samples.ix[samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = mass.kde2d(np.array(coordiantes.ix[:, 'x'], dtype=float), np.array(coordiantes.ix[:, 'y'], dtype=float),
                         bandwidths, n=np.array([n_grid]), lims=np.array([x_min, x_max, y_min, y_max]))
        kdes[s] = np.array(kde[2])

    # Assign the best KDE probability and state for each grid
    grid_probabilities = np.zeros((n_grid, n_grid))
    grid_states = np.empty((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            grid_probabilities[i, j] = max(kdes[:, j, i])
            grid_states[i, j] = np.argmax(kdes[:, i, j])

    # Set up figure
    figure = plt.figure(figsize=figure_size)

    # Set up axes
    gridspec = mpl.gridspec.GridSpec(10, 16)
    ax_title = plt.subplot(gridspec[0, :])
    ax_title.axis('off')
    ax_map = plt.subplot(gridspec[2:, :13])
    ax_map.axis('off')
    ax_legend = plt.subplot(gridspec[1:, 14:])
    ax_title.axis('off')

    # Plot title
    ax_title.text(0, 0.9, title, fontsize=title_fontsize, color=title_fontcolor, weight='bold')
    ax_title.text(0, 0.26, '{} samples, {} components, and {} states'.format(*reversed(h.shape), len(unique_states)),
                  fontsize=subtitle_fontsize, color=subtitle_fontcolor, weight='bold')

    # Plot components and their labels
    ax_map.plot(components_coordinates[:, 0], components_coordinates[:, 1], marker='D', linestyle='',
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor, zorder=6)
    # Put labels on top or bottom of the marker
    component_text_verticalshift = 1.39
    for i in range(h.shape[0]):
        if convexhull_region.contains_point(
                (components_coordinates[i, 0], components_coordinates[i, 1] - component_text_verticalshift)):
            x, y = components_coordinates[i, 0], components_coordinates[i, 1] + component_text_verticalshift
        else:
            x, y = components_coordinates[i, 0], components_coordinates[i, 1] - component_text_verticalshift
        ax_map.text(x, y, standardized_clipped_h.index[i],
                    fontsize=component_fontsize, color=component_markerfacecolor, weight='bold',
                    horizontalalignment='center', verticalalignment='center', zorder=6)

    # Plot Delaunay triangulation
    ax_map.triplot(delaunay.points[:, 0], delaunay.points[:, 1], delaunay.simplices.copy(),
                   linewidth=delaunay_linewidth, color=delaunay_linecolor, zorder=4)

    # Plot samples
    if annotations:
        cmap, cmap_min, cmap_max, = _setup_cmap(annotations, annotation_type)
    for idx, s in samples.iterrows():
        if annotations:
            c = cmap(s.ix['annotation'])
        else:
            c = CMAP_CATEGORICAL(int(s.ix['state'] / len(unique_states) * CMAP_CATEGORICAL.N))
        ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c,
                    markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, zorder=5)

    # Plot contours
    # TODO: don't draw masked contours
    # masked_coordinates = []
    # for i in range(n_grid):
    #     for j in range(n_grid):
    #         if not convexhull_region.contains_point((xgrids[i], ygrids[j])):
    #             masked_coordinates.append([i, j])
    # masked_coordinates = np.array(masked_coordinates)
    # z = grid_probabilities.copy()
    # z[masked_coordinates[:, 0], masked_coordinates[:, 1]] = np.nan
    # ax_map.contour(xgrids, ygrids, z, n_contour,
    #                linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha, zorder=2)
    if contour:
        ax_map.contour(x_grids, y_grids, grid_probabilities, n_contour,
                       linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha, zorder=2)

    # Plot background
    if background:
        for i in range(n_grid):
            for j in range(n_grid):
                if convexhull_region.contains_point((x_grids[i], y_grids[j])):
                    c = CMAP_CATEGORICAL(int(grid_states[i, j] / len(unique_states) * CMAP_CATEGORICAL.N))
                    a = min(background_max_alpha,
                            (grid_probabilities[i, j] - grid_probabilities.min()) /
                            (grid_probabilities.max() - grid_probabilities.min()))
                    a *= background_alpha_factor

                    ax_map.plot(x_grids[i], y_grids[j], marker='s',
                                markersize=background_markersize, markerfacecolor=c, alpha=a, zorder=1)
                else:
                    ax_map.plot(x_grids[i], y_grids[j], marker='s',
                                markersize=background_markersize * 1.26, markerfacecolor='w', zorder=3)

    # Plot legends
    if annotations:
        ax_legend.axis('on')

        ax_legend.set_title('Feature\nIC=xxx (p-val=xxx)', fontsize=legend_fontsize * 1.26, weight='bold')

        palette = {}
        for s in unique_states:
            palette[s] = CMAP_CATEGORICAL(int(s / len(unique_states) * CMAP_CATEGORICAL.N))

        if effectplot_type == 'violine':
            violinplot(x=annotations, y=states, palette=palette, scale='count', inner=None, orient='h', ax=ax_legend)
            boxplot(x=annotations, y=states, showbox=False, showmeans=True,
                    meanprops={'marker': 'o',
                               'markerfacecolor': effectplot_mean_markerfacecolor,
                               'markeredgewidth': 0.9,
                               'markeredgecolor': effectplot_mean_markeredgecolor},
                    medianprops={'color': effectplot_median_markeredgecolor}, orient='h', ax=ax_legend)
        elif effectplot_type == 'box':
            boxplot(x=annotations, y=states, palette=palette, showmeans=True,
                    meanprops={'marker': 'o',
                               'markerfacecolor': effectplot_mean_markerfacecolor,
                               'markeredgewidth': 0.9,
                               'markeredgecolor': effectplot_mean_markeredgecolor},
                    medianprops={'color': effectplot_median_markeredgecolor}, orient='h', ax=ax_legend)

        ax_legend.set_yticklabels(
            ['State {} (n={})'.format(s, sum(np.array(states) == s)) for s in unique_states],
            fontsize=legend_fontsize, weight='bold')
        ax_legend.yaxis.tick_right()
        ax_legend.set_xticks([np.min(annotations), np.mean(annotations), np.max(annotations)])
        for t in ax_legend.get_xticklabels():
            t.set(rotation=90, size=legend_fontsize * 0.9, weight='bold')

        ax_legend.patch.set_visible(False)
        ax_legend.axvline(np.min(annotations), color='#000000', ls='-', alpha=0.16)
        ax_legend.axvline(np.mean(annotations), color='#000000', ls='-', alpha=0.39)
        ax_legend.axvline(np.max(annotations), color='#000000', ls='-', alpha=0.16)
        right_adjust = 0.88

    else:
        ax_legend.axis([0, 1, 0, 1])
        for i, s in unique_states:
            y = 1 - float(1 / (len(unique_states) + 1)) * (i + 1)
            c = CMAP_CATEGORICAL(int(s / len(unique_states) * CMAP_CATEGORICAL.N))
            ax_legend.plot(0.5, y, marker='o', markersize=legend_markersize, markerfacecolor=c, zorder=5)
            ax_legend.text(0.6, y, 'State {} (n={})'.format(s, sum(states == s)),
                           fontsize=legend_fontsize, weight='bold', verticalalignment='center')
        right_adjust = 0.92

    if output_filename:
        figure.subplots_adjust(left=0.069, right=right_adjust, top=0.96, bottom=0.069)
        figure.savefig(output_filename, dpi=dpi)

    plt.show()
