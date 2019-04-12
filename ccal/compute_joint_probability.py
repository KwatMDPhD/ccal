from numpy import rot90

from .estimate_kernel_density import estimate_kernel_density
from .plot_and_save import plot_and_save


def compute_joint_probability(
    variables,
    variable_types=None,
    bandwidths="normal_reference",
    grid_size=64,
    plot_kernel_density=True,
    plot_probability=True,
    names=None,
):

    n_dimension = len(variables)

    if variable_types is None:

        variable_types = "c" * n_dimension

    kernel_density = estimate_kernel_density(
        variables,
        variable_types,
        bandwidths=bandwidths,
        grid_sizes=(grid_size,) * n_dimension,
    )

    probability = kernel_density / kernel_density.sum()

    if n_dimension == 2:

        if names is None:

            names = tuple("variables[{}]".format(i) for i in range(n_dimension))

        if plot_kernel_density:

            plot_and_save(
                {
                    "layout": {
                        "title": {"text": "KDE({}, {})".format(names[0], names[1])},
                        "xaxis": {"title": names[0]},
                        "yaxis": {"title": names[1]},
                    },
                    "data": [{"type": "heatmap", "z": rot90(kernel_density)[::-1]}],
                },
                None,
            )

        if plot_probability:

            plot_and_save(
                {
                    "layout": {
                        "title": {"text": "P({}, {})".format(names[0], names[1])},
                        "xaxis": {"title": names[0]},
                        "yaxis": {"title": names[1]},
                    },
                    "data": [{"type": "heatmap", "z": rot90(probability)[::-1]}],
                },
                None,
            )

    return probability
