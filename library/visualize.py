"""
Cancer Computational Biology Analysis Visualization Library v0.1

Authors:
Pablo Tamayo
pablo.tamayo.r@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Genomics and Computational Biology, UCSD Moore's Cancer Center

James Jensen
Email
Affiliation
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


## Global parameters
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

CMAP = mpl.cm.ocean

FONT1 = {'family': 'serif',
         'color':  BLACK,
         'weight': 'bold',
         'size': 36}
FONT2 = {'family': 'serif',
         'color':  BLACK,
         'weight': 'bold',
         'size': 24}
FONT3 = {'family': 'serif',
         'color':  BLACK,
         'weight': 'normal',
         'size': 16}


## Functions
def plot_graph(graph, filename=None):

    # Initialze figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)

    # Get position
    pos = nx.spring_layout(graph)
    # Draw
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos)

    # Configure figure
    cut = 1.00
    xmax = cut * max(x for x, y in pos.values())
    ymax = cut * max(y for x, y in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.show()
    if filename:
        plt.savefig(filename, bbox_inches='tight')

    # TODO: exit properly
    #pylab.close()
    #del fig


def plot_heatmap_panel(referece, dataframe, annotation_dataframe, figure_size=(30, 30), title=None, font1=FONT1, font2=FONT2, font3=FONT3):
    """
    Plot a heatmap panels (horizontal).
    """
    # TODO: use reference as colorbar
    # TODO: raise exception
    assert dataframe.shape[0] == annotation_dataframe.shape[0]
    
    ref_min = dataframe.values.min()
    ref_max = dataframe.values.max()
    
    ## Visualization parameters
    # TODO: Set size automatically
    figure_height = dataframe.shape[0] + 1
    figure_width = 7
    heatmap_left = 0
    heatmap_height = 1
    heatmap_width = 6

    ## Initialize figure
    fig = plt.figure(figsize=figure_size)
    #fig.suptitle(title, fontdict=font1)

    ## Initialize axes
    # Reference
    ax_ref = plt.subplot2grid((figure_height, figure_width), (0, heatmap_left), rowspan=heatmap_height, colspan=heatmap_width)
    if title:
        ax_ref.set_title(title, fontdict=font1)
    norm_ref = mpl.colors.Normalize(vmin=ref_min, vmax=ref_max)
    cbar_ref = mpl.colorbar.ColorbarBase(ax_ref, cmap=CMAP, norm=norm_ref, orientation='horizontal', ticks=[ref_min, ref_max], ticklocation='top')
    plt.setp(ax_ref.get_xticklabels(), **font2)

    # Reference annotation
    ax_ref_ann = plt.subplot2grid((figure_height, figure_width), (0, heatmap_left + heatmap_width), rowspan=heatmap_height, colspan=1)
    ax_ref_ann.set_axis_off()
    a1, a2, a3 = annotation_dataframe.columns
    ann = '{}\t\t{}\t\t{}'.format(a1, a2, a3).expandtabs()
    ax_ref_ann.text(0, 0.5, ann, fontdict=font2)

    # Features
    for i in range(dataframe.shape[0]):
        # Make row axes
        ax = plt.subplot2grid((figure_height, figure_width), (i + 1, heatmap_left), rowspan=heatmap_height, colspan=heatmap_width)
        sns.heatmap(dataframe.ix[i:i + 1], ax=ax,
                    vmin=ref_min, vmax=ref_max, robust=True,
                    center=None, mask=None,
                    square=False, cmap=CMAP, linewidth=0, linecolor=WHITE,
                    annot=False, fmt=None, annot_kws={},
                    xticklabels=False, yticklabels=True,
                    cbar=False)
        plt.setp(ax.get_xticklabels(), **font3, rotation=0)
        plt.setp(ax.get_yticklabels(), **font3, rotation=0)

    # Feature annotations
    for i in range(annotation_dataframe.shape[0]):
        a1, a2, a3 = annotation_dataframe.ix[i]
        ax = plt.subplot2grid((figure_height, figure_width), (i + 1, heatmap_left + heatmap_width), rowspan=heatmap_height, colspan=1)
        ax.set_axis_off()
        ann = '{:.2e}\t\t{:.2e}\t\t{:.2e}'.format(a1, a2, a3).expandtabs()
        ax.text(0, 0.5, ann, fontdict=font3)

    # Clean up the layout
    fig.tight_layout()