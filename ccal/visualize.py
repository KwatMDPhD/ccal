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
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from .support import verbose_print

# ======================================================================================================================
# Parameters
# ======================================================================================================================
# Colors
BLACK = '000000'
GRAY = '888888'
SILVER = 'C2C2C2'
WHITE = 'FEFEFE'
MAROON = '800000'
RED = 'FF0000'
RED_ORANGE = 'EC6313'
ORANGE = 'FFA600'
GOLD = 'EEC727'
YELLOW = 'F6F63C'
GREEN = '008000'
LIGHT_GREEN = '69D969'
LIME_GREEN = 'ABFF07'
TEAL = '05A29F'
AQUA = '04FFFB'
LIGHT_BLUE = '04CAFF'
BLUE = '0411FF'
NAVY = '0D126E'
LIGHT_PURPLE = 'F0C6FF'
PURPLE = '800080'
LIGHT_PINK = 'FFC0CB '
PINK = 'FFB3DE'
HOT_PINK = 'FF69B4 '
RED_VIOLET = 'C71585'

# Color maps
CMAP_CONTINUOUS = mpl.cm.bwr
CMAP_CATEGORICAL = mpl.cm.Paired
CMAP_BINARY = sns.light_palette('black', n_colors=128, as_cmap=True)

# Fonts
FONT12 = {'family': 'arial',
          'size': 12,
          'color': BLACK,
          'weight': 'normal'}
FONT12_BOLD = {'family': 'arial',
               'size': 12,
               'color': BLACK,
               'weight': 'bold'}
FONT16_BOLD = {'family': 'arial',
               'size': 16,
               'color': BLACK,
               'weight': 'bold'}
FONT20_BOLD = {'family': 'arial',
               'size': 20,
               'color': BLACK,
               'weight': 'bold'}


# ======================================================================================================================
# Functions
# ======================================================================================================================

def plot_features_and_reference(features, ref, annotations, features_type='continuous', ref_type='continuous',
                                title=None, rowname_size=25, filename_prefix=None, figure_type='.png'):
    """
    Plot a heatmap panel.
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param ref: pandas Series (m_elements), must have indices, which must match 'features`'s columns
    :param annotations:  pandas DataFrame (n_features, n_annotations), must have indices, which must match 'features`'s
    :param features_type: str, {continuous, categorical, binary}
    :param ref_type: str, {continuous, categorical, binary}
    :param title: str, figure title
    :param rowname_size: int, the maximum length of a feature name label
    :param filename_prefix: str, file path prefix to save the figure
    :param figure_type: str, file type to save the figure
    :return: None
    """
    features_nrow, features_ncol = features.shape

    # Set figure size
    if features_ncol < 10:
        fig_width = 10 / 6
    elif features_ncol < 35:
        fig_width = features_ncol / 6
    else:
        fig_width = 35 / 6

    if features_nrow < 3:
        fig_height = 4 / 3
    else:
        fig_height = features_nrow / 3

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=900)
    text_margin = 1

    # Set heatmap parameters for ref
    if ref_type is 'continuous':
        ref_cmap = CMAP_CONTINUOUS
        ref_min, ref_max = -2.5, 2.5
    elif ref_type is 'categorical':
        ref_cmap = CMAP_CATEGORICAL
        ref_min, ref_max = 0, np.unique(ref.values).size
    elif ref_type is 'binary':
        ref_cmap = CMAP_BINARY
        ref_min, ref_max = 0, 1
    else:
        raise ValueError('Unknown ref_type {}.'.format(ref_type))

    # Set heatmap parameters for features
    if features_type is 'continuous':
        features_cmap = CMAP_CONTINUOUS
        features_min, features_max = -2.5, 2.5
    elif features_type is 'categorical':
        features_cmap = CMAP_CATEGORICAL
        features_min, features_max = 0, np.unique(features.values).size
    elif features_type is 'binary':
        features_cmap = CMAP_BINARY
        features_min, features_max = -0.025, 1
    else:
        raise ValueError('Unknown features_type {}.'.format(features_type))

    # Normalize
    if features_type is 'continuous':
        verbose_print('Normalizing continuous features ...')
        for i, (idx, s) in enumerate(features.iterrows()):
            mean = s.mean()
            std = s.std()
            for j, v in enumerate(s):
                features.iloc[i, j] = (v - mean) / std
    if ref_type is 'continuous':
        verbose_print('Normalizing continuous ref ...')
        ref = (ref - ref.mean()) / ref.std()

    # Plot ref
    ref_ax = plt.subplot2grid((features_nrow, 1), (0, 0))
    if title:
        ref_ax.text(features_ncol / 2, 4 * text_margin, title,
                    horizontalalignment='center', verticalalignment='bottom', **FONT16_BOLD)
    sns.heatmap(pd.DataFrame(ref).T, vmin=ref_min, vmax=ref_max, robust=True,
                cmap=ref_cmap, linecolor=BLACK, fmt=None, xticklabels=False, yticklabels=False, cbar=False)
    # Add ref texts
    ref_ax.text(-text_margin, 0.5, ref.name,
                horizontalalignment='right', verticalalignment='center', **FONT12_BOLD)
    for j, a in enumerate(annotations.columns):
        ref_ax.text(features_ncol + text_margin * (4 * j + text_margin), 0.5, a,
                    horizontalalignment='left', verticalalignment='center', **FONT12_BOLD)

    # Add binary or categorical ref labels
    if ref_type in ('binary', 'categorical'):
        # Find boundaries
        boundaries = [0]
        prev_v = ref.iloc[0]
        for i, v in enumerate(ref.iloc[1:]):
            if prev_v != v:
                boundaries.append(i + 1)
            prev_v = v
        boundaries.append(features_ncol)
        # Calculate label's horizontal positions
        label_horizontal_positions = []
        prev_b = 0
        for b in boundaries[1:]:
            label_horizontal_positions.append(b - (b - prev_b) / 2)
            prev_b = b
        # TODO: get_unique_in_order
        unique_ref_labels = np.unique(ref.values)[::-1]
        # Add labels
        for i, pos in enumerate(label_horizontal_positions):
            ref_ax.text(pos, 1, unique_ref_labels[i],
                        horizontalalignment='center', **FONT12_BOLD)

    # Plot features
    features_ax = plt.subplot2grid((features_nrow, 1), (0, 1), rowspan=features_nrow)
    sns.heatmap(features, vmin=features_min, vmax=features_max, robust=True,
                cmap=features_cmap, linecolor=BLACK, fmt=None, xticklabels=False, yticklabels=False, cbar=False)

    for i, idx in enumerate(features.index):
        y = features_nrow - i - 0.5
        features_ax.text(-text_margin, y, idx[:rowname_size],
                         horizontalalignment='right', verticalalignment='center', **FONT12_BOLD)
        for j, a in enumerate(annotations.iloc[i, :]):
            features_ax.text(features_ncol + text_margin * (4 * j + text_margin), y, a,
                             horizontalalignment='left', verticalalignment='center', **FONT12_BOLD)

    fig.tight_layout()
    plt.show(fig)

    if filename_prefix:
        filename = filename_prefix + figure_type
        fig.savefig(filename)
        verbose_print('Saved the figure as {}.'.format(filename))


def plot_nmf_result(nmf_results, k, figsize=(25, 10), dpi=80, output_filename=None):
    """
    Plot NMF results from cca.library.cca.nmf.
    :param nmf_results: dict, NMF result per k (key: k; value: dict(key: w, h, err; value: w matrix, h matrix, and reconstruction error))
    :param k: int, k for NMF
    :param figsize: tuple (width, height),
    :param dpi: int, DPI
    :param output_filename: str, file path to save the figure
    :return: None
    """
    # Plot W and H
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    sns.heatmap(nmf_results[k]['W'], cmap='bwr', yticklabels=False, ax=ax1)
    ax1.set(xlabel='Component', ylabel='Gene')
    ax1.set_title('W matrix generated using k={}'.format(k))

    sns.heatmap(nmf_results[k]['H'], cmap='bwr', xticklabels=False, ax=ax2)
    ax2.set(xlabel='Sample', ylabel='Component')
    ax2.set_title('H matrix generated using k={}'.format(k))

    if output_filename:
        plt.savefig(output_filename + '.png')


def plot_nmf_scores(scores, figsize=(25, 10), title=None, filename=None):
    """
    Plot NMF score
    :param scores: dict, NMF score per k (key: k; value: score)
    :param figsize: tuple (width, height),
    :param title: str, figure title
    :param filename: str, file path to save the output figure
    :return: None
    """
    plt.figure(figsize=figsize)
    ax = sns.pointplot(x=[k for k, v in scores.items()], y=[v for k, v in scores.items()])
    ax.set(xlabel='k', ylabel='Score')
    if title:
        ax.set_title(title)

    if filename:
        plt.savefig(filename + '.png')


# TODO: finalize
def plot_graph(graph, filename=None):
    """
    Plot networkx `graph`.
    :param graph: networkx graph,
    :param filename: str, file path to save the figure
    :return: None
    """
    # Initialze figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
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

    if filename:
        plt.savefig(filename, bbox_inches='tight')


def make_colorbar():
    """
    Make colorbar examples.
    """
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which the colorbar will be used.
    cmap = CMAP_CONTINUOUS
    norm = mpl.colors.Normalize(vmin=5, vmax=10)

    # ColorbarBase derives from ScalarMappable and puts a colorbar in a specified axes,
    # so it has everything needed for a standalone colorbar.
    # There are many more kwargs, but the following gives a basic continuous colorbar with ticks and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1,
                                    cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Unit')

    # The length of the bounds array must be one greater than the length of the color list.
    cmap = mpl.colors.ListedColormap([RED, PURPLE, GREEN])
    # The bounds must be monotonically increasing.
    bounds = [1, 2, 6, 8]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Eextended ends to show the 'over' and 'under' value colors.
    cmap.set_over(SILVER)
    cmap.set_under(SILVER)
    cb2 = mpl.colorbar.ColorbarBase(ax2,
                                    cmap=cmap,
                                    norm=norm,
                                    boundaries=[bounds[0] - 3] + bounds + [bounds[-1] + 3],
                                    extend='both',
                                    extendfrac='auto',
                                    ticks=bounds,
                                    spacing='proportional',
                                    orientation='horizontal')
    cb2.set_label('Unit')
