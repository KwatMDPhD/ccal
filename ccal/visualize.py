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
import math

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from .support import _print

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
BAD_COLOR = 'wheat'
CMAP_CONTINUOUS = mpl.cm.bwr
CMAP_CONTINUOUS.set_bad(BAD_COLOR)
CMAP_CATEGORICAL = mpl.cm.Paired
CMAP_CATEGORICAL.set_bad(BAD_COLOR)
CMAP_BINARY = sns.light_palette('black', n_colors=2, as_cmap=True)
CMAP_BINARY.set_bad(BAD_COLOR)

# Fonts
FONT = 'arial'
FONT9_BOLD = {'family': FONT,
              'size': 9,
              'color': BLACK,
              'weight': 'bold'}
FONT12 = {'family': FONT,
          'size': 12,
          'color': BLACK,
          'weight': 'normal'}
FONT12_BOLD = {'family': FONT,
               'size': 12,
               'color': BLACK,
               'weight': 'bold'}
FONT16_BOLD = {'family': FONT,
               'size': 16,
               'color': BLACK,
               'weight': 'bold'}
FONT20_BOLD = {'family': FONT,
               'size': 20,
               'color': BLACK,
               'weight': 'bold'}


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_features_and_reference(features, ref, annotations, features_type='continuous', ref_type='continuous',
                                title=None, rowname_size=25, plot_colname=False, filename_prefix=None,
                                figure_type='.png'):
    """
    Plot a heatmap panel.
    :param features: pandas DataFrame (n_features, m_elements), must have indices and columns
    :param ref: pandas Series (m_elements), must have indices, which must match 'features`'s columns
    :param annotations:  pandas DataFrame (n_features, n_annotations), must have indices, which must match 'features`'s
    :param features_type: str, {continuous, categorical, binary}
    :param ref_type: str, {continuous, categorical, binary}
    :param title: str, figure title
    :param rowname_size: int, the maximum length of a feature name label
    :param plot_colname: bool, plot column names or not
    :param filename_prefix: str, file path prefix to save the figure
    :param figure_type: str, file type to save the figure
    :return: None
    """
    fig = plt.figure(figsize=(min(math.pow(features.shape[1], 0.5), 7), math.pow(features.shape[0], 0.9)))
    plot_grid = (features.shape[0] + 1, 1)

    horizontal_text_margin = math.pow(features.shape[1], 0.73)
    horizontal_annotation_pos = lambda x: x * horizontal_text_margin + horizontal_text_margin / 9

    if features_type is 'continuous':
        features_cmap = CMAP_CONTINUOUS
        features_min, features_max = -3, 3
    elif features_type is 'categorical':
        features_cmap = CMAP_CATEGORICAL
        features_min, features_max = 0, np.unique(features.values).size
    elif features_type is 'binary':
        features_cmap = CMAP_BINARY
        features_min, features_max = 0, 1
    else:
        raise ValueError('Unknown features_type {}.'.format(features_type))
    if ref_type is 'continuous':
        ref_cmap = CMAP_CONTINUOUS
        ref_min, ref_max = -3, 3
    elif ref_type is 'categorical':
        ref_cmap = CMAP_CATEGORICAL
        ref_min, ref_max = 0, np.unique(ref.values).size
    elif ref_type is 'binary':
        ref_cmap = CMAP_BINARY
        ref_min, ref_max = 0, 1
    else:
        raise ValueError('Unknown ref_type {}.'.format(ref_type))

    if features_type is 'continuous':
        _print('Normalizing continuous features ...')
        for i, (idx, s) in enumerate(features.iterrows()):
            mean = s.mean()
            std = s.std()
            for j, v in enumerate(s):
                features.ix[i, j] = (v - mean) / std
    if ref_type is 'continuous':
        _print('Normalizing continuous ref ...')
        ref = (ref - ref.mean()) / ref.std()

    # Plot ref
    ref_ax = plt.subplot2grid(plot_grid, (0, 0))
    if title:
        ref_ax.text(features.shape[1] / 2, 1.9, title,
                    horizontalalignment='center', **FONT16_BOLD)
    sns.heatmap(pd.DataFrame(ref).T, vmin=ref_min, vmax=ref_max, robust=True,
                cmap=ref_cmap, linecolor=BLACK, fmt=None, xticklabels=False, yticklabels=False, cbar=False)
    # Add ref texts
    ref_ax.text(-horizontal_annotation_pos(0), 0.5, ref.name,
                horizontalalignment='right', verticalalignment='center', **FONT12_BOLD)
    for j, a in enumerate(annotations.columns):
        ref_ax.text(features.shape[1] + horizontal_annotation_pos(j), 0.5, a,
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
        boundaries.append(features.shape[1])
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
            ref_ax.text(pos, 1.19, unique_ref_labels[i],
                        horizontalalignment='center', **FONT12_BOLD)

    # # Plot features
    features_ax = plt.subplot2grid(plot_grid, (1, 0), rowspan=features.shape[0])
    sns.heatmap(features, vmin=features_min, vmax=features_max, robust=True,
                cmap=features_cmap, linecolor=BLACK, fmt=None, xticklabels=False, yticklabels=False, cbar=False)

    for i, idx in enumerate(features.index):
        y = features.shape[0] - i - 0.5
        features_ax.text(-horizontal_annotation_pos(0), y, idx[:rowname_size],
                         horizontalalignment='right', verticalalignment='center', **FONT12_BOLD)
        for j, a in enumerate(annotations.iloc[i, :]):
            features_ax.text(features.shape[1] + horizontal_annotation_pos(j), y, a,
                             horizontalalignment='left', verticalalignment='center', **FONT12_BOLD)

    # Plot column names at the bottom
    if plot_colname:
        for j, c in enumerate(features.columns):
            features_ax.text(j + 0.5, -1, c,
                             rotation=90, horizontalalignment='center', verticalalignment='top', **FONT9_BOLD)

    # fig.tight_layout()
    plt.show(fig)

    if filename_prefix:
        filename = filename_prefix + figure_type
        fig.savefig(filename)
        _print('Saved the figure as {}.'.format(filename))


def plot_nmf_result(nmf_results, k, figsize=(7, 5), dpi=80, output_filename=None):
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
    Plot NMF `scores`.
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
