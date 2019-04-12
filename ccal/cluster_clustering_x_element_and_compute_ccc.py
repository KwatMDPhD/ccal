from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform

from .compute_coclustering_from_clustering_x_element import (
    compute_coclustering_from_clustering_x_element,
)


def cluster_clustering_x_element_and_compute_ccc(
    clustering_x_element, k, linkage_method
):

    clustering_distance = squareform(
        1 - compute_coclustering_from_clustering_x_element(clustering_x_element)
    )

    clustering_distance_linkage = linkage(clustering_distance, method=linkage_method)

    return (
        fcluster(clustering_distance_linkage, k, criterion="maxclust") - 1,
        cophenet(clustering_distance_linkage, clustering_distance)[0],
    )
