from numpy import full, nan
from numpy.random import seed, shuffle
from pandas import isna
from sklearn.linear_model import LinearRegression

from .COLOR_CATEGORICAL import COLOR_CATEGORICAL
from .compute_empirical_p_value import compute_empirical_p_value
from .plot_and_save import plot_and_save
from .RANDOM_SEED import RANDOM_SEED


def correlate(
    x,
    y,
    n_permutation=0,
    random_seed=RANDOM_SEED,
    plot=True,
    marker_size=16,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    html_file_path=None,
    plotly_html_file_path=None,
):

    model = LinearRegression()

    xs = tuple((x_,) for x_ in x)

    model.fit(xs, y)

    r2 = model.score(xs, y)

    if n_permutation:

        permuted_r2s = full(n_permutation, nan)

        m_ = LinearRegression()

        y_ = y.copy()

        seed(random_seed)

        for i in range(n_permutation):

            shuffle(y_)

            m_.fit(xs, y_)

            permuted_r2s[i] = m_.score(xs, y_)

        p_value = min(
            compute_empirical_p_value(r2, permuted_r2s, "<"),
            compute_empirical_p_value(r2, permuted_r2s, ">"),
        )

    else:

        p_value = nan

    if plot:

        r2_p_value_str = "R^2={:.3f}".format(r2)

        if not isna(p_value):

            r2_p_value_str = "{} & P-Value={:.3e}".format(r2_p_value_str, p_value)

        if title:

            title = "{}\n{}".format(title, r2_p_value_str)

        else:

            title = r2_p_value_str

        plot_and_save(
            {
                "layout": {
                    "title": {"text": title},
                    "xaxis": {"title": xaxis_title},
                    "yaxis": {"title": yaxis_title},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": x,
                        "y": y,
                        "name": "Data",
                        "mode": "markers",
                        "marker": {"size": marker_size, "color": COLOR_CATEGORICAL[0]},
                    },
                    {
                        "type": "scatter",
                        "x": x,
                        "y": model.coef_ * x + model.intercept_,
                        "name": "Fit",
                        "marker": {"color": COLOR_CATEGORICAL[1]},
                    },
                ],
            },
            html_file_path=html_file_path,
            plotly_html_file_path=plotly_html_file_path,
        )

    return r2, p_value
