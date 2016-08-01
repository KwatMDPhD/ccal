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

import pandas as pd
import numpy as np
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

DPI = 900


# ======================================================================================================================
# Functions
# ======================================================================================================================
def plot_features_and_reference(features, ref, annotations, features_type='continuous', ref_type='continuous',
                                title=None, title_size=16, annotation_header=None, annotation_label_size=9,
                                plot_colname=False,
                                figure_filename=None):
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
    :return: None
    """
    features_cmap, features_min, features_max = _setup_heatmap_parameters(features, features_type)
    ref_cmap, ref_min, ref_max = _setup_heatmap_parameters(ref, ref_type)

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
        fig.savefig(figure_filename, dpi=DPI, bbox_inches='tight')
        print_log('Saved the figure as {}.'.format(figure_filename))


def _setup_heatmap_parameters(pandas_obj, data_type):
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


def plot_nmf_result(nmf_results, k, figsize=(7, 5), title=None, output_filename=None):
    """
    Plot NMF results from cca.library.cca.nmf function.
    :param nmf_results: dict, result per k (key: k; value: dict(key: w, h, err; value: w matrix, h matrix, and error))
    :param k: int, k for NMF
    :param figsize: tuple (width, height),
    :param title: str, figure title
    :param output_filename: str, file path to save the figure
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
        plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')


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
