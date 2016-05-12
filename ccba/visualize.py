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