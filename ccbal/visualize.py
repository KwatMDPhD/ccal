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
CMAP_CONTINUOUS = mpl.cm.bwr
CMAP_BINARY = sns.light_palette('black', n_colors=128, as_cmap=True)
CMAP_CATEGORICAL = mpl.cm.Set2

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


def plot_heatmap_panel_v2(target, df, row_annot, title, t_type="auto", sort_target=True, class_labels=None, cbar=False):
    ncol = df.shape[1]
    nrow = df.shape[0]

    df2 = pd.DataFrame(np.zeros((nrow, ncol)))

    CMAP_CONTINUOUS = mpl.cm.bwr
    CMAP_BINARY = sns.light_palette("black", n_colors=128, as_cmap=True)
    CMAP_CATEGORICAL = mpl.cm.Set2

    f_width = np.where(ncol / 5 < 5, 5, np.where(ncol / 5 > 7, 7, ncol / 5))
    f_height = np.where(nrow < 25, nrow / 1.5, nrow / 2.5)
    figure_size = (f_width, f_height)

    # Target pre-processing
    if t_type == "binary" or (t_type == "auto" and len(np.unique(target.iloc[0, :]))) == 2:
        cmap_target = CMAP_BINARY
        (t_vmin, t_vmax) = (0, 1)
    elif t_type == "categ" or (t_type == "auto" and len(np.unique(target.iloc[0, :])) > 2 and
                                       len(np.unique(target.iloc[0, :])) < ncol):
        cmap_target = CMAP_CATEGORICAL
        (t_vmin, t_vmax) = (0, len(np.unique(target.iloc[0:]) - 1))
    else:  # t_type = "cont"
        cmap_target = CMAP_CONTINUOUS
        target.iloc[0, :] = (target.iloc[0, :] - np.mean(target.iloc[0, :])) / np.std(target.iloc[0, :])
        (t_vmin, t_vmax) = (-2.5, 2.5)
        means = df.mean(1, skipna=True)
        stds = df.std(1, skipna=True)

    # Feature pre-processing
    row_names = df.index
    max_len_row = max([len(x) for x in row_names])
    if len(np.unique(df)) == 2:
        cmap_features = CMAP_BINARY
        (f_vmin, f_vmax) = (0, 1)
        df = df + 0.25
    else:
        cmap_features = CMAP_CONTINUOUS
        (f_vmin, f_vmax) = (-2.5, 2.5)
        means = df.mean(1, skipna=True)
        stds = df.std(1, skipna=True)
        for i in range(nrow):
            for j in range(ncol):
                df.values[i, j] = (df.values[i, j] - means[i]) / stds[i]

    if sort_target == True:
        t_order = list(reversed(np.argsort(target.values[0, :], kind='quicksort')))
        target = target.reindex_axis(target.columns[t_order], axis=1)
        df2 = df.reindex_axis(df.columns[t_order], axis=1)

        # Computer locations for class labels (for binary or categorical)
        if (t_type == "binary" or t_type == "categ") and class_labels != None:
            boundaries = np.zeros(len(np.unique(target.iloc[0, :])))
            locs_labels = np.zeros(len(np.unique(target.iloc[0, :])))
            k = 0
            for i in range(1, ncol):
                if target.iloc[0, i] != target.iloc[0, i - 1]:
                    boundaries[k] = i
                    k = k + 1
            boundaries[len(boundaries) - 1] = ncol
            locs_labels[0] = boundaries[0] / 2
            for k in range(1, len(locs_labels)):
                locs_labels[k] = boundaries[k] - (boundaries[k] - boundaries[k - 1]) / 2.0

    ## Initialize figure
    fig = plt.figure(figsize=figure_size)

    ax1 = plt.subplot2grid((nrow + 1, 1), (0, 0))

    sns.heatmap(target, vmin=t_vmin, vmax=t_vmax, robust=True, center=None, mask=None,
                square=False, cmap=cmap_target, linewidth=0.0, linecolor='b',
                annot=False, fmt=None, annot_kws={}, xticklabels=False,
                yticklabels=['  '], cbar=False)
    ax1.text(-0.1, 0.2, target.index[0], fontsize=13, horizontalalignment='right', fontweight='bold')
    ax1.text(ncol / 2, 1.6, title, fontsize=16, horizontalalignment='center', fontweight='bold')
    if (t_type == "binary" or t_type == "categ") and class_labels != None:
        for k in range(len(locs_labels)):
            ax1.text(locs_labels[k], 0.25, class_labels[k], fontsize=13, horizontalalignment='center',
                     fontweight='bold')

    ax2 = plt.subplot2grid((nrow + 1, 1), (0, 1), rowspan=nrow)
    sns.heatmap(df2, vmin=f_vmin, vmax=f_vmax, robust=True, center=None, mask=None,
                square=False, cmap=cmap_features, linewidth=0.0, linecolor='b',
                annot=False, fmt=None, annot_kws={}, xticklabels=False,
                yticklabels=[' ' for i in range(nrow)], cbar=cbar, cbar_kws={"orientation": "horizontal"})
    for i in range(nrow):
        ax2.text(-0.1, nrow - i - 1 + 0.3, row_names[i], fontsize=13, horizontalalignment='right', fontweight='bold')
        ax2.text(ncol + 0.1, nrow - i - 1 + 0.3, row_annot[i], fontsize=13, fontweight='bold')
    ax2.text(ncol + 1, nrow + 0.3, row_annot.name, fontsize=13, fontweight='bold')

    fig.tight_layout()
    plt.show(fig)

    return


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
