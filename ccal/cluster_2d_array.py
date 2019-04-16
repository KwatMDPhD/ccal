from numpy import concatenate, where
from pandas import unique
from scipy.cluster.hierarchy import dendrogram, linkage

from .check_nd_array_for_bad import check_nd_array_for_bad
from .ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays import (
    ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays,
)


def cluster_2d_array(
    _2d_array,
    axis,
    groups=None,
    distance_function=None,
    linkage_method="average",
    optimal_ordering=True,
    raise_for_bad=True,
):

    check_nd_array_for_bad(_2d_array, raise_for_bad=raise_for_bad)

    if axis == 1:

        _2d_array = _2d_array.T

    if distance_function is None:

        distance_function = (
            ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays
        )

    if groups is None:

        return dendrogram(
            linkage(
                _2d_array,
                method=linkage_method,
                metric=distance_function,
                optimal_ordering=optimal_ordering,
            ),
            no_plot=True,
        )["leaves"]

    else:

        if len(groups) != _2d_array.shape[0]:

            raise ValueError(
                "len(groups) ({}) != len(axis-{} slices) ({})".format(
                    len(groups), axis, _2d_array.shape[0]
                )
            )

        indices = []

        for group in unique(groups):

            group_indices = where(groups == group)[0]

            clustered_indices = dendrogram(
                linkage(
                    _2d_array[group_indices, :],
                    method=linkage_method,
                    metric=distance_function,
                    optimal_ordering=optimal_ordering,
                ),
                no_plot=True,
            )["leaves"]

            indices.append(group_indices[clustered_indices])

        return concatenate(indices)
