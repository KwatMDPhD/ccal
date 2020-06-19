from math import ceil
from multiprocessing import Pool

from numpy import apply_along_axis, asarray, concatenate, full, isnan, nan, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame, Series, unique

from .array import ignore_nan_and_function_1, ignore_nan_and_function_2, normalize
from .clustering import cluster
from .CONSTANT import RANDOM_SEED
from .plot import DATA_TYPE_COLORSCALE, plot_plotly
from .series import select_extreme
from .significance import get_moe, get_p_values_and_q_values


def function_heat_map(
    series,
    dataframe,
    function,
    series_ascending=True,
    scores=None,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    score_ascending=False,
    plot=True,
    n_extreme=8,
    fraction_extreme=None,
    series_data_type="continuous",
    dataframe_data_type="continuous",
    plot_std=nan,
    directory_path=None,
):

    common_pandas_index = series.index & dataframe.columns

    print(
        "series.index ({}) and dataframe.columns ({}) have {} in common.".format(
            series.index.size, dataframe.columns.size, common_pandas_index.size
        )
    )

    series = series[common_pandas_index]

    if series_ascending is not None:

        series.sort_values(ascending=series_ascending, inplace=True)

    dataframe = dataframe[series.index]

    if scores is None:

        scores = DataFrame(
            index=dataframe.index, columns=("Score", "0.95 MoE", "P-Value", "Q-Value")
        )

        n_row, n_column = dataframe.shape

        n_job = min(n_job, n_row)

        print("Computing scores (n_job={})...".format(n_job))

        pool = Pool(n_job)

        print("\tScore (function={})...".format(function.__name__))

        seed(seed=random_seed)

        vector = series.values

        scores["Score"] = asarray(
            pool.starmap(
                ignore_nan_and_function_2,
                ((vector, matrix_row, function) for matrix_row in dataframe.values),
            )
        )

        if 0 < n_sampling:

            print("\t0.95 MoE (n_sampling={})...".format(n_sampling))

            row_x_sampling = full((n_row, n_sampling), nan)

            n_column_to_sample = ceil(n_column * 0.632)

            for sampling_index in range(n_sampling):

                row_name = choice(n_column, size=n_column_to_sample)

                # TODO: confirm that initializing outside tuple comprehension is faster
                vector = series.values[row_name]

                row_x_sampling[:, sampling_index] = pool.starmap(
                    ignore_nan_and_function_2,
                    (
                        (vector, matrix_row, function)
                        for matrix_row in dataframe.values[:, row_name]
                    ),
                )

            scores["0.95 MoE"] = apply_along_axis(
                lambda scores: get_moe(scores[~isnan(scores)]), 1, row_x_sampling,
            )

        if 0 < n_permutation:

            print("\tP-Value and Q-Value (n_permutation={})...".format(n_permutation))

            row_x_permutation = full((n_row, n_permutation), nan)

            vector = series.values.copy()

            for permuting_index in range(n_permutation):

                shuffle(vector)

                row_x_permutation[:, permuting_index] = pool.starmap(
                    ignore_nan_and_function_2,
                    ((vector, matrix_row, function) for matrix_row in dataframe.values),
                )

            scores["P-Value"], scores["Q-Value"] = get_p_values_and_q_values(
                scores["Score"].values, row_x_permutation.flatten(), "<>"
            )

        pool.terminate()

    else:

        scores = scores.reindex(row_name=dataframe.index)

    scores.sort_values("Score", ascending=score_ascending, inplace=True)

    if directory_path is not None:

        tsv_file_path = "{}/function_heat_map.tsv".format(directory_path)

        scores.to_csv(tsv_file_path, sep="\t")

    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Scores"},
                    "xaxis": {"title": {"text": "Rank"}},
                },
                "data": [
                    {
                        "name": score_name,
                        "x": score_values.index,
                        "y": score_values.values,
                    }
                    for score_name, score_values in scores.items()
                ],
            },
        )

        series_plot = series.copy()

        scores_plot = scores.copy()

        if n_extreme is not None or fraction_extreme is not None:

            scores_plot = scores_plot.loc[
                select_extreme(
                    scores_plot["Score"],
                    "<>",
                    n=n_extreme,
                    fraction=fraction_extreme,
                    plot=False,
                )
            ].sort_values("Score", ascending=score_ascending)

        dataframe_plot = dataframe.loc[scores_plot.index]

        if series_data_type == "continuous":

            series_plot = Series(
                ignore_nan_and_function_1(
                    series_plot.values, normalize, "-0-", update=True
                ),
                name=series_plot.name,
                index=series_plot.index,
            ).clip(lower=-plot_std, upper=plot_std)

        if dataframe_data_type == "continuous":

            dataframe_plot = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    dataframe_plot.values,
                    normalize,
                    "-0-",
                    update=True,
                ),
                index=dataframe_plot.index,
                columns=dataframe_plot.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        if (
            not series_plot.isna().any()
            and isinstance(series_ascending, bool)
            and 1 < series_plot.value_counts().min()
        ):

            vector = series_plot.values

            matrix = dataframe_plot.values

            leave_index = []

            # TODO: check order
            for n in unique(vector):

                row_name = where(vector == n)[0]

                leave_index.append(row_name[cluster(matrix.T[row_name])[0]])

            dataframe_plot = dataframe_plot.iloc[:, concatenate(leave_index)]

            series_plot = series_plot[dataframe_plot.columns]

        n_row_plot = 1 + 1 + dataframe_plot.shape[0]

        fraction_row = 1 / n_row_plot

        layout = {
            "height": max(480, 24 * n_row_plot),
            "width": 800,
            "margin": {"l": 200, "r": 200},
            "title": {"x": 0.5},
            "yaxis": {"domain": (0, 1 - 2 * fraction_row), "showticklabels": False},
            "yaxis2": {"domain": (1 - fraction_row, 1), "showticklabels": False},
            "annotations": [],
        }

        annotation_template = {
            "xref": "paper",
            "yref": "paper",
            "yanchor": "middle",
            "font": {"size": 10},
            "showarrow": False,
        }

        y = 1 - fraction_row / 2

        layout["annotations"].append(
            {
                "x": 0,
                "y": y,
                "xanchor": "right",
                "text": "<b>{}</b>".format(series_plot.name),
                **annotation_template,
            }
        )

        def get_x(score_index):

            return 1.1 + score_index / 6.4

        for score_index, score_name in enumerate(
            ("Score (\u0394)", "P-Value", "Q-Value")
        ):

            layout["annotations"].append(
                {
                    "x": get_x(score_index),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(score_name),
                    **annotation_template,
                }
            )

        y -= 2 * fraction_row

        for row_name, (score, moe, p_value, q_value) in scores_plot.iterrows():

            layout["annotations"].append(
                {
                    "x": 0,
                    "y": y,
                    "xanchor": "right",
                    "text": row_name,
                    **annotation_template,
                }
            )

            for score_index, score_str in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": get_x(score_index),
                        "y": y,
                        "xanchor": "center",
                        "text": score_str,
                        **annotation_template,
                    }
                )

            y -= fraction_row

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = tsv_file_path.replace(".tsv", ".html")

        heatmap_trace_template = {
            "type": "heatmap",
            "showscale": False,
        }

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "yaxis": "y2",
                        "x": series_plot.index,
                        "z": series_plot.to_frame().T,
                        "colorscale": DATA_TYPE_COLORSCALE[series_data_type],
                        **heatmap_trace_template,
                    },
                    {
                        "yaxis": "y",
                        "x": dataframe_plot.columns,
                        "y": dataframe_plot.index[::-1],
                        "z": dataframe_plot.iloc[::-1],
                        "colorscale": DATA_TYPE_COLORSCALE[dataframe_data_type],
                        **heatmap_trace_template,
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return scores


def function_heat_map_summary(
    vector,
    matrix_dicts,
    statistics,
    plot_only_shared=False,
    vector_ascending=True,
    vector_data_type="continuous",
    plot_std=nan,
    html_file_path=None,
):

    if plot_only_shared:

        for matrix_dict in matrix_dicts.values():

            vector = vector.loc[vector.index & matrix_dict["matrix"].columns]

    if vector_ascending is not None:

        vector.sort_values(ascending=vector_ascending, inplace=True)

    vector_ = vector.copy()

    if vector_data_type == "continuous":

        vector_ = Series(
            ignore_nan_and_function_1(vector_.values, normalize, "-0-", update=True),
            name=vector_.name,
            index=vector_.index,
        ).clip(lower=-plot_std, upper=plot_std)

    n_space = 2

    n_row = 1

    for matrix_dict in matrix_dicts.values():

        n_row += n_space

        n_row += matrix_dict["matrix"].shape[0]

    layout = {
        "height": max(480, 24 * n_row),
        "width": 800,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "annotations": [],
    }

    fraction_row = 1 / n_row

    yaxis = "yaxis{}".format(len(matrix_dicts) + 1)

    domain = 1 - fraction_row, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "type": "heatmap",
            "x": vector_.index,
            "z": vector_.to_frame().T,
            "colorscale": DATA_TYPE_COLORSCALE[vector_data_type],
            "showscale": False,
        }
    ]

    annotation_ = {
        "xref": "paper",
        "yref": "paper",
        "yanchor": "middle",
        "font": {"size": 10},
        "showarrow": False,
    }

    layout["annotations"].append(
        {
            "x": 0,
            "y": 1 - fraction_row / 2,
            "xanchor": "right",
            "text": "<b>{}</b>".format(vector.name),
            **annotation_,
        }
    )

    for matrix_index, (matrix_name, matrix_dict) in enumerate(matrix_dicts.items()):

        matrix_ = matrix_dict["matrix"].reindex(columns=vector_.index)

        statistics_ = statistics[matrix_name].reindex(index=matrix_.index)

        if "emphasis" in matrix_dict:

            score_ascending = matrix_dict["emphasis"] == "-"

        else:

            score_ascending = False

        statistics_.sort_values("Score", ascending=score_ascending, inplace=True)

        matrix_ = matrix_.loc[statistics_.index]

        if matrix_dict["data_type"] == "continuous":

            matrix_ = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    matrix_.values,
                    normalize,
                    "-0-",
                    update=True,
                ),
                index=matrix_.index,
                columns=matrix_.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        yaxis = "yaxis{}".format(len(matrix_dicts) - matrix_index)

        domain = (
            max(0, domain[0] - fraction_row * (n_space + matrix_.shape[0])),
            domain[0] - fraction_row * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "type": "heatmap",
                "x": matrix_.columns,
                "y": matrix_.index[::-1],
                "z": matrix_.values[::-1],
                "colorscale": DATA_TYPE_COLORSCALE[matrix_dict["data_type"]],
                "showscale": False,
            }
        )

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": domain[1] + fraction_row / 2,
                "xanchor": "center",
                "text": "<b>{}</b>".format(matrix_name),
                **annotation_,
            }
        )

        def get_x(x_index):

            return 1.1 + x_index / 6.4

        if matrix_index == 0:

            for x_index, str_ in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

                layout["annotations"].append(
                    {
                        "x": get_x(x_index),
                        "y": 1 - fraction_row / 2,
                        "xanchor": "center",
                        "text": "<b>{}</b>".format(str_),
                        **annotation_,
                    }
                )

        y = domain[1] - fraction_row / 2

        for row_name, (score, moe, p_value, q_value) in statistics_.iterrows():

            layout["annotations"].append(
                {"x": 0, "y": y, "xanchor": "right", "text": row_name, **annotation_}
            )

            for x_index, str_ in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": get_x(x_index),
                        "y": y,
                        "xanchor": "center",
                        "text": str_,
                        **annotation_,
                    }
                )

            y -= fraction_row

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
