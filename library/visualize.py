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
from matplotlib import pylab
import matplotlib.pyplot as plt


def plot_heatmap(matrix):
    """
    Make a heat map of <matrix>.
    """
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()


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
