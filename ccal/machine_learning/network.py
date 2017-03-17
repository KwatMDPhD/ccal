"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from networkx import DiGraph, Graph
from pandas import read_csv


def make_network_from_similarity_matrix(similarity_matrix):
    """

    :param similarity_matrix:
    :return:
    """

    graph = Graph()
    for i, s in similarity_matrix.iterrows():
        for j in s.index:
            graph.add_edge(s.name, j, weight=s.ix[j])


def make_network_from_edge_file(edge_file, di=False, sep='\t'):
    """
    Make networkx graph from edge_file: from<sep>to.
    :param edge_file:
    :param di: boolean, directed or not
    :param sep: separator, default \t
    :return:
    """

    # Load edge
    e = read_csv(edge_file, sep=sep)

    # Make graph from edge
    if di:
        # Directed graph
        g = DiGraph()
    else:
        # Undirected graph
        g = Graph()

    g.add_edges_from(e.values)

    return g
