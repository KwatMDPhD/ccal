from numpy import absolute, argmax, linspace, rot90

from .compute_joint_probability import compute_joint_probability
from .compute_posterior_probability import compute_posterior_probability
from .get_target_grid_indices import get_target_grid_indices
from .plot_and_save import plot_and_save


def infer(
    variables,
    variable_types=None,
    bandwidths="normal_reference",
    grid_size=64,
    target="max",
    plot=True,
    names=None,
):

    n_dimension = len(variables)

    if variable_types is None:

        variable_types = "c" * n_dimension

    if plot:

        if names is None:

            names = tuple("variables[{}]".format(i) for i in range(n_dimension))

    p_vs = compute_joint_probability(
        variables,
        variable_types=variable_types,
        bandwidths=bandwidths,
        grid_size=grid_size,
        plot_kernel_density=False,
        plot_probability=plot,
        names=names,
    )

    p_tv__ntvs = compute_posterior_probability(p_vs, plot=plot, names=names)

    if target is "max":

        t_grid_coordinates = get_target_grid_indices(p_tv__ntvs, argmax)

    else:

        t_grid = linspace(variables[-1].min(), variables[-1].max(), grid_size)

        t_i = absolute(t_grid - target).argmin()

        t_grid_coordinates = get_target_grid_indices(p_tv__ntvs, lambda _: t_i)

    p_tvt__ntvs = p_tv__ntvs[t_grid_coordinates].reshape(
        (grid_size,) * (n_dimension - 1)
    )

    if plot:

        if n_dimension == 2:

            name = "P({} = {} | {})".format(names[-1], target, names[0])

            plot_and_save(
                {
                    "layout": {
                        "title": {"text": name},
                        "xaxis": {"title": names[0]},
                        "yaxis": {"title": "Probability"},
                    },
                    "data": [
                        {
                            "type": "scatter",
                            "name": name,
                            "x": tuple(range(grid_size)),
                            "y": p_tvt__ntvs,
                        }
                    ],
                },
                None,
            )

        elif n_dimension == 3:

            plot_and_save(
                {
                    "layout": {
                        "title": {
                            "text": "P({} = {} | {}, {})".format(
                                names[-1], target, names[0], names[1]
                            )
                        },
                        "xaxis": {"title": names[0]},
                        "yaxis": {"title": names[1]},
                    },
                    "data": [{"type": "heatmap", "z": rot90(p_tvt__ntvs)[::-1]}],
                },
                None,
            )

    return p_tv__ntvs, p_tvt__ntvs
