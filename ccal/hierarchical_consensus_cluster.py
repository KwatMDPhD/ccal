from os.path import join

from numpy import full, nan
from numpy.random import randint, seed
from pandas import DataFrame, Index, Series
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .make_membership_df_from_categorical_series import (
    make_membership_df_from_categorical_series,
)
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def hierarchical_consensus_cluster(
    df,
    k,
    distance__column_x_column=None,
    distance_function="euclidean",
    n_clustering=10,
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_df=True,
    directory_path=None,
):

    if distance__column_x_column is None:

        print("Computing distance with {} ...".format(distance_function))

        distance__column_x_column = DataFrame(
            squareform(pdist(df.values.T, distance_function)),
            index=df.columns,
            columns=df.columns,
        )

    print("HCC with K={} ...".format(k))

    clustering_x_column = full((n_clustering, distance__column_x_column.shape[1]), nan)

    n_per_print = max(1, n_clustering // 10)

    seed(random_seed)

    for clustering in range(n_clustering):

        if clustering % n_per_print == 0:

            print("\t(K={}) {}/{} ...".format(k, clustering + 1, n_clustering))

        random_columns_with_repeat = randint(
            0,
            high=distance__column_x_column.shape[0],
            size=distance__column_x_column.shape[0],
        )

        clustering_x_column[clustering, random_columns_with_repeat] = fcluster(
            linkage(
                squareform(
                    distance__column_x_column.iloc[
                        random_columns_with_repeat, random_columns_with_repeat
                    ]
                ),
                method=linkage_method,
            ),
            k,
            criterion="maxclust",
        )

    column_cluster, column_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_column, k, linkage_method
    )

    if directory_path is not None:

        cluster_x_column = make_membership_df_from_categorical_series(
            Series(column_cluster, index=df.columns)
        )

        cluster_x_column.index = Index(
            ("Cluster{}".format(cluster) for cluster in cluster_x_column.index),
            name="Cluster",
        )

        cluster_x_column.to_csv(join(directory_path, "cluster_x_column.tsv"), sep="\t")

    if plot_df:

        print("Plotting df ...")

        file_name = "df.html"

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, file_name)

        plot_heat_map(
            df,
            normalization_axis=0,
            normalization_method="-0-",
            column_annotation=column_cluster,
            title="HCC K={} Column Cluster".format(k),
            xaxis_title=df.columns.name,
            yaxis_title=df.index.name,
            html_file_path=html_file_path,
        )

    return column_cluster, column_cluster__ccc
