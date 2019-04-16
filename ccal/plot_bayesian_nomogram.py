from numpy import absolute, linspace, log2, sort

from .plot_and_save import plot_and_save


def plot_bayesian_nomogram(
    target,
    target_hit,
    target_miss,
    n_grid,
    conditional_probabilities,
    names,
    html_file_path=None,
):

    target_hit_probability = (target == target_hit).sum() / target.size

    target_miss_probability = (target == target_miss).sum() / target.size

    target_grid = linspace(target.min(), target.max(), n_grid)

    target_grid_hit_index = absolute(target_grid - target_hit).argmin()

    target_grid_miss_index = absolute(target_grid - target_miss).argmin()

    target = sort(target)[:: target.size // n_grid]

    grid_shape = (n_grid, n_grid)

    for conditional_probability in conditional_probabilities:

        if conditional_probability.shape != grid_shape:

            raise ValueError(
                "conditional_probabilities[i].shape should be {}.".format(grid_shape)
            )

    layout = {
        "height": 80 * max(8, len(conditional_probabilities)),
        "title": {"text": "Bayesian Nomogram"},
        "xaxis": {"title": "Log Odds Ratio"},
        "yaxis": {
            "zeroline": False,
            "ticks": "",
            "showticklabels": False,
            "title": "Evidence",
            "dtick": 1,
        },
    }

    data = []

    for i, (conditional_probability, name) in enumerate(
        zip(conditional_probabilities, names)
    ):

        target_hit_conditional_probability = conditional_probability[
            :, target_grid_hit_index
        ]

        target_miss_conditional_probability = conditional_probability[
            :, target_grid_miss_index
        ]

        log_odds_ratios = log2(
            (target_hit_conditional_probability / target_miss_conditional_probability)
            / (target_hit_probability / target_miss_probability)
        )

        x = tuple(range(n_grid))

        plot_and_save(
            {
                "layout": {"title": {"text": name}, "legend": {"orientation": "v"}},
                "data": [
                    {"type": "scatter", "name": name, "x": x, "y": target},
                    {
                        "type": "scatter",
                        "name": "P(hit | {})".format(name),
                        "x": x,
                        "y": target_hit_conditional_probability,
                    },
                    {
                        "type": "scatter",
                        "name": "P(miss | {})".format(name),
                        "x": x,
                        "y": target_miss_conditional_probability,
                    },
                    {
                        "type": "scatter",
                        "name": "Log Odds Ratio",
                        "x": x,
                        "y": log_odds_ratios,
                    },
                ],
            },
            None,
        )

        data.append(
            {
                "type": "scatter",
                "legendgroup": name,
                "name": name,
                "x": (log_odds_ratios.min(), log_odds_ratios.max()),
                "y": (i, i),
            }
        )

    plot_and_save({"layout": layout, "data": data}, html_file_path)
