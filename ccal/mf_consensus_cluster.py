from os.path import join

from numpy import full, nan
from pandas import DataFrame, Index, Series

from .cluster_2d_array import cluster_2d_array
from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .mf_by_multiplicative_update import mf_by_multiplicative_update
from .nmf_by_sklearn import nmf_by_sklearn
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def mf_consensus_cluster(
    df,
    k,
    mf_function="nmf_by_sklearn",
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_w=True,
    plot_h=True,
    plot_df=True,
    directory_path=None,
):

    print("MFCC K={} ...".format(k))

    clustering_x_w_element = full((n_clustering, df.shape[0]), nan)

    clustering_x_h_element = full((n_clustering, df.shape[1]), nan)

    n_per_print = max(1, n_clustering // 10)

    if mf_function == "mf_by_multiplicative_update":

        mf_function = mf_by_multiplicative_update

    elif mf_function == "nmf_by_sklearn":

        mf_function = nmf_by_sklearn

    for clustering in range(n_clustering):

        if not clustering % n_per_print:

            print("\t(K={}) {}/{} ...".format(k, clustering + 1, n_clustering))

        w, h, e = mf_function(
            df.values, k, n_iteration=n_iteration, random_seed=random_seed + clustering
        )

        if clustering == 0:

            w_0 = w

            h_0 = h

            e_0 = e

            factors = Index(("Factor{}".format(i) for i in range(k)), name="Factor")

            w_0 = DataFrame(w_0, index=df.index, columns=factors)

            h_0 = DataFrame(h_0, index=factors, columns=df.columns)

            if directory_path is not None:

                w_0.to_csv(join(directory_path, "w.tsv"), sep="\t")

                h_0.to_csv(join(directory_path, "h.tsv"), sep="\t")

            if plot_w:

                print("Plotting w ...")

                file_name = "w.html"

                if directory_path is None:

                    html_file_path = None

                else:

                    html_file_path = join(directory_path, file_name)

                plot_heat_map(
                    w_0.iloc[cluster_2d_array(w_0.values, 0)],
                    title="MF K={} W".format(k),
                    xaxis_title=w_0.columns.name,
                    yaxis_title=w_0.index.name,
                    html_file_path=html_file_path,
                )

            if plot_h:

                print("Plotting h ...")

                file_name = "h.html"

                if directory_path is None:

                    html_file_path = None

                else:

                    html_file_path = join(directory_path, file_name)

                plot_heat_map(
                    h_0.iloc[:, cluster_2d_array(h_0.values, 1)],
                    title="MF K={} H".format(k),
                    xaxis_title=h_0.columns.name,
                    yaxis_title=h_0.index.name,
                    html_file_path=html_file_path,
                )

        clustering_x_w_element[clustering, :] = w.argmax(axis=1)

        clustering_x_h_element[clustering, :] = h.argmax(axis=0)

    w_element_cluster, w_element_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_w_element, k, linkage_method
    )

    w_element_cluster = Series(w_element_cluster, name="Cluster", index=df.index)

    h_element_cluster, h_element_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_h_element, k, linkage_method
    )

    h_element_cluster = Series(h_element_cluster, name="Cluster", index=df.columns)

    if plot_df:

        print("Plotting df.clustered ...")

        file_name = "df.cluster.html"

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, file_name)

        w_element_cluster_sorted = w_element_cluster.sort_values()

        h_element_cluster_sorted = h_element_cluster.sort_values()

        df = df.loc[w_element_cluster_sorted.index, h_element_cluster_sorted.index]

        plot_heat_map(
            df,
            row_annotation=w_element_cluster_sorted,
            column_annotation=h_element_cluster_sorted,
            title="MFCC K={}".format(k),
            xaxis_title=df.columns.name,
            yaxis_title=df.index.name,
            html_file_path=html_file_path,
        )

    return (
        w_0,
        h_0,
        e_0,
        w_element_cluster,
        w_element_cluster__ccc,
        h_element_cluster,
        h_element_cluster__ccc,
    )