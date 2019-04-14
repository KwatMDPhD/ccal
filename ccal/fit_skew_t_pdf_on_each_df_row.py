from numpy import full, nan
from pandas import DataFrame, concat

from .fit_skew_t_pdf import fit_skew_t_pdf
from .multiprocess import multiprocess
from .split_df import split_df


def _fit_skew_t_pdf_on_each_df_row(df):

    skew_t_pdf_fit_parameter = full((df.shape[0], 5), nan)

    n = df.shape[0]

    n_per_print = max(1, n // 10)

    for i, (index, series) in enumerate(df.iterrows()):

        if i % n_per_print == 0:

            print("({}/{}) {} ...".format(i + 1, n, index))

        _1d_array = series.values

        skew_t_pdf_fit_parameter[i] = fit_skew_t_pdf(_1d_array)

    return DataFrame(
        skew_t_pdf_fit_parameter,
        index=df.index,
        columns=("N Data", "Location", "Scale", "Degree of Freedom", "Shape"),
    )


def fit_skew_t_pdf_on_each_df_row(df, n_job=1, output_file_path=None):

    skew_t_pdf_fit_parameter = concat(
        multiprocess(
            _fit_skew_t_pdf_on_each_df_row,
            ((df_,) for df_ in split_df(df, 0, min(df.shape[0], n_job))),
            n_job,
        )
    )

    if output_file_path is not None:

        skew_t_pdf_fit_parameter.to_csv(output_file_path, sep="\t")

    return skew_t_pdf_fit_parameter
