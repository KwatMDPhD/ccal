from os.path import join

from numpy import full, nan
from numpy.random import randint, seed
from pandas import DataFrame, Index, Series
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .make_binary_df_from_categorical_series import (
    make_binary_df_from_categorical_series,
)
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def hierarchical_consensus_cluster(
    df,
    k,
    axis,
    distance__element_x_element=None,
    distance_function="euclidean",
    n_clustering=10,
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_df=True,
    directory_path=None,
):

    if axis == 1:

        df = df.T

    if distance__element_x_element is None:

        print(
            "Computing distance__element_x_element distance with {} ...".format(
                distance_function.__name__
            )
        )

        distance__element_x_element = DataFrame(
            squareform(pdist(df.values, distance_function)),
            index=df.index,
            columns=df.index,
        )

        if directory_path is not None:

            distance__element_x_element.to_csv(
                join(directory_path, "distance.element_x_element.tsv"), sep="\t"
            )

    print("HCC K={} ...".format(k))

    clustering_x_element = full((n_clustering, df.index.size), nan)

    n_per_print = max(1, n_clustering // 10)

    seed(seed=random_seed)

    for clustering in range(n_clustering):

        if not clustering % n_per_print:

            print("\t(K={}) {}/{} ...".format(k, clustering + 1, n_clustering))

        random_elements_with_repeat = randint(0, high=df.index.size, size=df.index.size)

        clustering_x_element[clustering, random_elements_with_repeat] = fcluster(
            linkage(
                squareform(
                    distance__element_x_element.iloc[
                        random_elements_with_repeat, random_elements_with_repeat
                    ]
                ),
                method=linkage_method,
            ),
            k,
            criterion="maxclust",
        )

    element_cluster, element_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_element, k, linkage_method
    )

    element_cluster = Series(element_cluster, name="Cluster", index=df.index)

    cluster_x_element = make_binary_df_from_categorical_series(element_cluster)

    cluster_x_element.index = Index(
        ("Cluster{}".format(i) for i in cluster_x_element.index),
        name=cluster_x_element.index.name,
    )

    if directory_path is not None:

        cluster_x_element.to_csv(
            join(directory_path, "cluster_x_element.tsv"), sep="\t"
        )

    if axis == 1:

        df = df.T

    if plot_df:

        print("Plotting df.clustered ...")

        file_name = "df.cluster.html"

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, file_name)

            element_cluster_sorted = element_cluster.sort_values()

            if axis == 0:

                df = df.loc[element_cluster_sorted.index]

                keyword_arguments = {"row_annotation": element_cluster_sorted}

            elif axis == 1:

                df = df[element_cluster_sorted.index]

                keyword_arguments = {"column_annotation": element_cluster_sorted}

        plot_heat_map(
            df,
            title="HCC K={}".format(k),
            xaxis_title=df.columns.name,
            yaxis_title=df.index.name,
            html_file_path=html_file_path,
            **keyword_arguments,
        )

    return element_cluster, element_cluster__ccc
