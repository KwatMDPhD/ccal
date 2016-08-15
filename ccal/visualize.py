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
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri

ro.conversion.py2ri = numpy2ri
from rpy2.robjects.packages import importr

mass = importr('MASS')

import numpy as np
import pandas as pd
from sklearn import manifold
from scipy.spatial import Delaunay, ConvexHull
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from .support import print_log, standardize_pandas_object, establish_path

# ======================================================================================================================
# Parameters
# ======================================================================================================================
# Color maps
BAD_COLOR = 'wheat'
CMAP_CONTINUOUS = mpl.cm.bwr
CMAP_CONTINUOUS.set_bad(BAD_COLOR)
CMAP_CATEGORICAL = mpl.cm.Paired
CMAP_CATEGORICAL.set_bad(BAD_COLOR)
CMAP_BINARY = sns.light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad(BAD_COLOR)

DPI = 1000


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_features_against_reference(features, ref, annotations, features_type='continuous', ref_type='continuous',
                                    title=None, title_size=16, annotation_header=None, annotation_label_size=9,
                                    plot_colname=False,
                                    figure_filename=None, dpi=DPI):
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
        features = standardize_pandas_object(features)
    if ref_type is 'continuous':
        print_log('Normalizing continuous ref ...')
        ref = (ref - ref.mean()) / ref.std()
        ref = standardize_pandas_object(ref)

    fig = plt.figure(figsize=(min(math.pow(features.shape[1], 0.7), 7), math.pow(features.shape[0], 0.9)))
    horizontal_text_margin = math.pow(features.shape[1], 0.39)
    gridspec = plt.GridSpec(features.shape[0] + 1, features.shape[1] + 1)

    # Plot ref, ref label, and title,
    ref_ax = plt.subplot(gridspec.new_subplotspec((0, 0), colspan=features.shape[1]))
    sns.heatmap(pd.DataFrame(ref).T, vmin=ref_min, vmax=ref_max, cmap=ref_cmap, xticklabels=False, cbar=False)
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
    sns.heatmap(features, vmin=features_min, vmax=features_max, cmap=features_cmap, xticklabels=plot_colname,
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


def _setup_cmap(pandas_obj, data_type):
    if data_type is 'continuous':
        data_cmap = CMAP_CONTINUOUS
        data_min, data_max = -3, 3
    elif data_type is 'categorical':
        data_cmap = CMAP_CATEGORICAL
        data_min, data_max = 0, np.unique(pandas_obj.values).size
    elif data_type is 'binary':
        data_cmap = CMAP_BINARY
        data_min, data_max = 0, 1
    else:
        raise ValueError('Unknown data_type {}.'.format(data_type))
    return data_cmap, data_min, data_max


def plot_nmf_result(nmf_results, k, figsize=(7, 5), title=None, output_filename=None, dpi=DPI):
    """
    Plot NMF results from cca.library.cca.nmf function.
    :param nmf_results: dict, result per k (key: k; value: dict(key: w, h, err; value: w matrix, h matrix, and error))
    :param k: int, k for NMF
    :param figsize: tuple (width, height),
    :param title: str, figure title
    :param output_filename: str, file path to save the figure
    :param dpi: int, dots-per-inch for the output figure
    :return: None
    """
    # Plot W and H
    fig, (ax_w, ax_h) = plt.subplots(1, 2, figsize=figsize)
    if title:
        fig.suptitle(title)

    sns.heatmap(nmf_results[k]['W'], cmap='bwr', yticklabels=False, ax=ax_w)
    ax_w.set(xlabel='Component', ylabel='Gene')
    ax_w.set_title('W matrix generated using k={}'.format(k))

    sns.heatmap(nmf_results[k]['H'], cmap='bwr', xticklabels=False, ax=ax_h)
    ax_h.set(xlabel='Sample', ylabel='Component')
    ax_h.set_title('H matrix generated using k={}'.format(k))

    plt.show()

    if output_filename:
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')


def plot_nmf_scores(scores, figsize=(7, 5), title=None, output_filename=None):
    """
    Plot NMF `scores`.
    :param scores: dict, NMF score per k (key: k; value: score)
    :param figsize: tuple, figure size (width, height)
    :param title: str, figure title
    :param output_filename: str, file path to save the figure
    :return: None
    """
    plt.figure(figsize=figsize)
    if title:
        plt.gcf().suptitle(title)

    ax = sns.pointplot(x=[k for k, v in scores.items()], y=[v for k, v in scores.items()])
    ax.set(xlabel='k', ylabel='Score')

    plt.show()

    if output_filename:
        plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')


def plot_graph(graph, figsize=(7, 5), title=None, output_filename=None):
    """
    Plot networkx `graph`.
    :param graph: networkx Graph,
    :param figsize: tuple, figure size (width, height)
    :param title: str, figure title
    :param output_filename: str, file path to save the figure
    :return: None
    """
    plt.figure(num=None, figsize=figsize)
    if title:
        plt.gcf().suptitle(title)
    plt.axis('off')

    # Get position
    positions = nx.spring_layout(graph)

    # Draw
    nx.draw_networkx_nodes(graph, positions)
    nx.draw_networkx_edges(graph, positions)
    nx.draw_networkx_labels(graph, positions)
    nx.draw_networkx_edge_labels(graph, positions)

    # Configure figure
    cut = 1.00
    xmax = cut * max(x for x, y in positions.values())
    ymax = cut * max(y for x, y in positions.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.show()

    if output_filename:
        plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')


def plot_onco_gps(h, n_state, states, annotations=(), annotation_type='continuous', output_filename=None, dpi=DPI,
                  figure_size=(10, 8), ax_spacing=0.9, coordinates_extending_factor=1 / 24, n_grid=100,
                  title='OncoGenic Positional System (Onco-GPS) Map', title_fontsize=24, title_fontcolor='#3326c0',
                  subtitle_fontsize=16, subtitle_fontcolor='#990000',
                  delaunay_linewidth=1, delaunay_linecolor='#000000',
                  component_markersize=13, component_markerfacecolor='#000726',
                  component_markeredgewidth=1, component_markeredgecolor='#ffffff',
                  component_text_verticalshift=1.3, component_fontsize=16,
                  kde_bandwidths_factor=1.5, sample_stretch_factor=2,
                  sample_markersize=12, sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
                  n_contour=10, contour_linewidth=0.81, contour_linecolor='#5a5a5a', contour_alpha=0.5,
                  background_max_alpha=1, background_alpha_factor=0.69, background_markersize=5.55,
                  legend_markersize=10, legend_fontsize=13):
    """
    :param h: pandas DataFrame (n_nmf_component, n_samples), NMF H matrix
    :param n_state: int, number of states to plot
    :param states: array-like (n_samples), samples' state
    :param annotations: array-like (n_samples), samples' annotations; samples are plotted based on annotations
    :param annotation_type: str, {continuous, categorical, binary}
    :param output_filename: str, file path to save the output figure
    :param dpi: int, dots-per-inch for the output figure
    :param figure_size: array_like (2),
    :param ax_spacing: float,
    :param coordinates_extending_factor: float,
    :param n_grid: int,
    :param title: str,
    :param title_fontsize: float,
    :param title_fontcolor: matplotlib compatible color expression,
    :param subtitle_fontsize: float,
    :param subtitle_fontcolor: matplotlib compatible color expression,
    :param delaunay_linewidth: float,
    :param delaunay_linecolor: matplotlib compatible color expression,
    :param component_markersize: float,
    :param component_markerfacecolor: matplotlib compatible color expression,
    :param component_markeredgewidth: float,
    :param component_markeredgecolor: matplotlib compatible color expression,
    :param component_text_verticalshift: float,
    :param component_fontsize: float,
    :param kde_bandwidths_factor: float,
    :param sample_stretch_factor: float,
    :param sample_markersize: float,
    :param sample_markeredgewidth: float,
    :param sample_markeredgecolor: matplotlib compatible color expression
    :param n_contour: int,
    :param contour_linewidth: float,
    :param contour_linecolor: float,
    :param contour_alpha: float,
    :param background_max_alpha: float,
    :param background_markersize: float,
    :param background_alpha_factor: float, factor to multiply the background alpha
    :param legend_markersize: float,
    :param legend_fontsize: float,
    :return: None
    """
    # Standardize H and clip values less than -3 and more than 3
    standardized_h = standardize_pandas_object(h)
    standardized_clipped_h = standardized_h.clip(-3, 3)

    # Project the H's components from <nsample>D to 2D, getting the x & y coordinates
    mds = manifold.MDS()
    # TODO: freeze random seed
    components_coordinates = mds.fit_transform(standardized_clipped_h)

    # Delaunay triangulate the components' 2D projected coordinates
    delaunay = Delaunay(components_coordinates)

    # Compute convexhull for the components' 2D projected coordinates
    convexhull = ConvexHull(components_coordinates)
    convexhull_region = mpl.path.Path(convexhull.points[convexhull.vertices])

    # Sample and their state labels and x & y coordinates computed using Delaunay triangulation simplices
    samples = pd.DataFrame(index=h.columns, columns=['state', 'x', 'y'])

    # Get sample states
    samples['state'] = states

    # Get sample annotations
    if any(annotations):
        samples['annotation'] = (np.array(annotations) - min(annotations)) / (max(annotations) - min(annotations))

    # Get sample x & y coordinates using Delaunay triangulation simplices
    for sample in samples.index:
        col = h.ix[:, sample]
        third = col.sort_values()[-3]
        col = col.mask(col < third, other=0)

        x = sum(col ** sample_stretch_factor * components_coordinates[:, 0]) / sum(col ** sample_stretch_factor)
        y = sum(col ** sample_stretch_factor * components_coordinates[:, 1]) / sum(col ** sample_stretch_factor)
        samples.ix[sample, ['x', 'y']] = x, y

    # Set x & y coordinate boundaries
    xcoordinates = components_coordinates[:, 0]
    xmin = min(xcoordinates)
    xmax = max(xcoordinates)
    xmargin = (xmax - xmin) * coordinates_extending_factor
    xmin -= xmargin
    xmax += xmargin
    ycoordinates = components_coordinates[:, 1]
    ymin = min(ycoordinates)
    ymax = max(ycoordinates)
    ymargin = (ymax - ymin) * coordinates_extending_factor
    ymin -= ymargin
    ymax += ymargin

    # Make x & y grids
    xgrids = np.linspace(xmin, xmax, n_grid)
    ygrids = np.linspace(ymin, ymax, n_grid)

    # Get KDE for each state using bandwidth created from all n_state' x & y coordinates
    kdes = np.zeros((n_state + 1, n_grid, n_grid))
    bandwidth_x = mass.bcv(np.array(samples.ix[:, 'x'].tolist()))[0]
    bandwidth_y = mass.bcv(np.array(samples.ix[:, 'y'].tolist()))[0]
    bandwidths = np.array([bandwidth_x * kde_bandwidths_factor, bandwidth_y * kde_bandwidths_factor]) / 2
    for s in sorted(samples.ix[:, 'state'].unique()):
        coordiantes = samples.ix[samples.ix[:, 'state'] == s, ['x', 'y']]
        x = np.array(coordiantes.ix[:, 'x'], dtype=float)
        y = np.array(coordiantes.ix[:, 'y'], dtype=float)
        kde = mass.kde2d(x, y, bandwidths, n=np.array([n_grid]), lims=np.array([xmin, xmax, ymin, ymax]))
        kdes[s] = np.array(kde[2])

    # Assign the best KDE probability and state for each grid intersection
    grid_probabilities = np.zeros((n_grid, n_grid))
    grid_states = np.empty((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            grid_probabilities[i, j] = max(kdes[:, j, i])
            grid_states[i, j] = np.argmax(kdes[:, i, j])

    # Set up figure
    figure = plt.figure(figsize=figure_size)

    # Set up axes
    gridspec = mpl.gridspec.GridSpec(10, 11)
    ax_title = plt.subplot(gridspec[0, :])
    ax_title.axis('off')
    ax_map = plt.subplot(gridspec[1:, :10])
    ax_map.axis('off')
    ax_legend = plt.subplot(gridspec[1:, 10:])
    ax_legend.axis([0, 1, 0, 1])
    ax_legend.axis('off')

    # Plot title
    ax_title.text(0, ax_spacing, title,
                  fontsize=title_fontsize, color=title_fontcolor, weight='bold', horizontalalignment='left')
    ax_title.text(0, ax_spacing * 0.5,
                  '{} samples, {} components, and {} states'.format(samples.shape[0], h.shape[0], n_state),
                  fontsize=subtitle_fontsize, color=subtitle_fontcolor, weight='bold', horizontalalignment='left')

    # Plot components
    ax_map.plot(components_coordinates[:, 0], components_coordinates[:, 1], linestyle='', marker='D',
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor, zorder=6)

    # Plot component labels
    for i in range(components_coordinates.shape[0]):
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
    for i, (idx, s) in enumerate(samples.iterrows()):
        if any(annotations):
            if annotation_type is 'continuous':
                cmap = CMAP_CONTINUOUS
            elif annotation_type is 'categorical':
                cmap = CMAP_CATEGORICAL
            elif annotation_type is 'binary':
                cmap = CMAP_BINARY
            c = cmap(s.ix['annotation'])
        else:
            c = CMAP_CATEGORICAL(int(s.ix['state'] / n_state * 255))
        ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize,
                    markerfacecolor=c, markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor,
                    zorder=5)

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
    ax_map.contour(xgrids, ygrids, grid_probabilities, n_contour,
                   linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha, zorder=2)

    # Plot background
    for i in range(n_grid):
        for j in range(n_grid):
            if convexhull_region.contains_point((xgrids[i], ygrids[j])):
                c = CMAP_CATEGORICAL(int(grid_states[i, j] / n_state * 255))
                a = min(background_max_alpha,
                        (grid_probabilities[i, j] - grid_probabilities.min()) /
                        (grid_probabilities.max() - grid_probabilities.min()))
                a *= background_alpha_factor

                ax_map.plot(xgrids[i], ygrids[j], marker='s',
                            markersize=background_markersize, markerfacecolor=c, alpha=a, zorder=1)
            else:
                ax_map.plot(xgrids[i], ygrids[j], marker='s',
                            markersize=background_markersize * 1.1, markerfacecolor='w', zorder=3)

    # Plot legends
    ax_legend.text(ax_spacing - 0.1, 0.96, 'States', fontsize=legend_fontsize, weight='bold', color='#000000')
    for i, state in enumerate(sorted(samples.ix[:, 'state'].unique())):
        y = 1 - 0.04 * (i + 2)  # 1 - (i + 1) / (n_state + 1)
        c = CMAP_CATEGORICAL(int(state / n_state * 255))
        ax_legend.plot(ax_spacing, y, marker='o', markersize=legend_markersize, markerfacecolor=c, zorder=5)
        ax_legend.text(ax_spacing * 1.5, y, 'S{}'.format(state),
                       fontsize=legend_fontsize, weight='bold', color='#000000', verticalalignment='center')

    if output_filename:
        figure.subplots_adjust(left=0.06, right=0.9, top=0.97, bottom=0.03)
        figure.savefig(output_filename, dpi=dpi)

    plt.show()
