"""
Computational Cancer Biology Analysis Library v0.1


Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Computational Cancer Biology, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Biology, UCSD Cancer Center


Description:
TODO
"""
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Parameters
# ======================================================================================================================
# Colors
CMAP_CONTINUOUS = mpl.cm.bwr
CMAP_BINARY = mpl.cm.Greys
# TODO: set up categorical color map
# CMAP_CATEGORICAL =
WHITE = '#FFFFFF'
SILVER = '#C0C0C0'
GRAY = '#808080'
BLACK = '#000000'
RED = '#FF0000'
MAROON = '#800000'
YELLOW = '#FFFF00'
OLIVE = '#808000'
LIME = '#00FF00'
GREEN = '#008000'
AQUA = '#00FFFF'
TEAL = '#008080'
BLUE = '#0000FF'
NAVY = '#000080'
FUCHSIA = '#FF00FF'
PURPLE = '#800080'

# Fonts
FONT1 = {'family': 'serif',
         'color': BLACK,
         'weight': 'bold',
         'size': 36}
FONT2 = {'family': 'serif',
         'color': BLACK,
         'weight': 'bold',
         'size': 24}
FONT3 = {'family': 'serif',
         'color': BLACK,
         'weight': 'normal',
         'size': 16}


# ======================================================================================================================
# Functions
# ======================================================================================================================
# TODO: use reference to make colorbar
def plot_heatmap_panel(dataframe, reference, annotation,
                       figure_size=(30, 30), title=None,
                       font1=FONT1, font2=FONT2, font3=FONT3):
    """
    Plot horizonzal heatmap panels.
    :param dataframe: pandas DataFrame (n_samples, n_features),
    :param reference: array-like (1, n_features),
    :param annotation: array_like (n_samples, ),
    :param figure_size: tuple (width, height),
    :param title: str, figure title
    :param font1: dict,
    :param font2: dict,
    :param font3: dict,
    :return: None
    """
    # Visualization parameters
    # TODO: Set size automatically
    figure_height = dataframe.shape[0] + 1
    figure_width = 7
    heatmap_left = 0
    heatmap_height = 1
    heatmap_width = 6

    # Initialize figure
    fig = plt.figure(figsize=figure_size)
    # TODO: consider removing
    # fig.suptitle(title, fontdict=font1)

    # Initialize reference axe
    # TODO: use reference as colorbar
    ref_min = dataframe.values.min()
    ref_max = dataframe.values.max()
    ax_ref = plt.subplot2grid((figure_height, figure_width), (0, heatmap_left), rowspan=heatmap_height,
                              colspan=heatmap_width)
    if title:
        ax_ref.set_title(title, fontdict=font1)
    norm_ref = mpl.colors.Normalize(vmin=ref_min, vmax=ref_max)
    mpl.colorbar.ColorbarBase(ax_ref, cmap=CMAP_CONTINUOUS, norm=norm_ref,
                              orientation='horizontal', ticks=[ref_min, ref_max], ticklocation='top')
    plt.setp(ax_ref.get_xticklabels(), **font2)

    # Add reference annotations
    ax_ref_ann = plt.subplot2grid((figure_height, figure_width), (0, heatmap_left + heatmap_width),
                                  rowspan=heatmap_height, colspan=1)
    ax_ref_ann.set_axis_off()
    ann = '\t\t'.join(annotation).expandtabs()
    ax_ref_ann.text(0, 0.5, ann, fontdict=font2)

    # Initialie feature axe
    for i in range(dataframe.shape[0]):
        # Make row axe
        ax = plt.subplot2grid((figure_height, figure_width), (i + 1, heatmap_left), rowspan=heatmap_height,
                              colspan=heatmap_width)
        sns.heatmap(dataframe.ix[i:i + 1, :-len(annotation)], ax=ax,
                    vmin=ref_min, vmax=ref_max, robust=True,
                    center=None, mask=None,
                    square=False, cmap=CMAP_CONTINUOUS, linewidth=0, linecolor=WHITE,
                    annot=False, fmt=None, annot_kws={},
                    xticklabels=False, yticklabels=True,
                    cbar=False)
        plt.setp(ax.get_xticklabels(), **font3, rotation=0)
        plt.setp(ax.get_yticklabels(), **font3, rotation=0)

    # Add feature annotations
    for i in range(dataframe.shape[0]):
        ax = plt.subplot2grid((figure_height, figure_width), (i + 1, heatmap_left + heatmap_width),
                              rowspan=heatmap_height, colspan=1)
        ax.set_axis_off()
        ann = '\t\t'.join(['{:.2e}'.format(n) for n in dataframe.ix[i, annotation]]).expandtabs()
        ax.text(0, 0.5, ann, fontdict=font3)

    # Clean up the layout
    fig.tight_layout()


def plot_nmf_result(nmf_results, k, figsize=(25, 10), dpi=80, output_filename=None):
    """
    Plot NMF results from ccba.library.ccba.nmf.
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

    # Plot reconstruction error
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.pointplot(x=list(nmf_results.keys()), y=[v['ERROR'] for v in nmf_results.values()])
    ax.set(xlabel='k', ylabel='Reconstruction Error')
    ax.set_title('k vs. Reconstruction Error')


def plot_nmf_scores(scores, figsize=(25, 10), title=None, output_filename=None):
    """
    Plot NMF score
    :param scores: dict, NMF score per k (key: k; value: score)
    :param figsize: tuple (width, height),
    :param title: str, figure title
    :param output_filename: str, file path to save the figure
    :return: None
    """
    plt.figure(figsize=figsize)
    ax = sns.pointplot(x=[k for k, v in scores.items()], y=[v for k, v in scores.items()])
    ax.set(xlabel='k', ylabel='Score')
    ax.set_title('k vs. Score')

    if output_filename:
        plt.savefig(output_filename + '.png')


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
