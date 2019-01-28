from numpy import array_split, concatenate
from pandas import DataFrame

from ._match_randomly_sampled_target_and_data_to_compute_margin_of_errors import (
    _match_randomly_sampled_target_and_data_to_compute_margin_of_errors,
)
from ._match_target_and_data import _match_target_and_data
from ._permute_target_and_match_target_and_data import (
    _permute_target_and_match_target_and_data,
)
from .compute_empirical_p_values_and_fdrs import compute_empirical_p_values_and_fdrs
from .compute_information_coefficient import compute_information_coefficient
from .make_object_int_mapping import make_object_int_mapping
from .multiprocess import multiprocess
from .select_series_indices import select_series_indices


def _match(
    target,
    data,
    target_ascending=True,
    n_job=1,
    match_function=compute_information_coefficient,
    n_required_for_match_function=2,
    raise_for_n_less_than_required=False,
    n_extreme=8,
    fraction_extreme=None,
    random_seed=20_121_020,
    n_sampling=0,
    n_permutation=0,
):
    
    common_indices = target.index & data.columns

    print(
        "target.index ({}) & data.columns ({}) have {} in common.".format(
            target.index.size, data.columns.size, len(common_indices)
        )
    )

    target = target[common_indices]

    if target.dtype == "O":

        target = target.map(make_object_int_mapping(target)[0])

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    data = data[target.index]

    save_indx = data.index

    target = target.values
    data = data.values

    score_moe_p_value_fdr = DataFrame(columns=("Score", "0.95 MoE", "P-Value", "FDR"))

    n_job = min(data.shape[0], n_job)

    print(
        "Computing score using {} with {} process ...".format(
            match_function.__name__, n_job
        )
    )

    data_split = array_split(data, n_job)

    score_moe_p_value_fdr["Score"] = concatenate(
        multiprocess(
            _match_target_and_data,
            (
                (
                    target,
                    data_,
                    match_function,
                    n_required_for_match_function,
                    raise_for_n_less_than_required,
                )
                for data_ in data_split
            ),
            n_job,
        )
    )



    if n_extreme is None and fraction_extreme is None:

        indices = select_series_indices(
            score_moe_p_value_fdr["Score"],
            "<>",
            n=n_extreme,
            fraction=fraction_extreme,
            plot=False,
        )

        score_moe_p_value_fdr.loc[
            indices, "0.95 MoE"
        ] = _match_randomly_sampled_target_and_data_to_compute_margin_of_errors(
            target,
            data[indices],
            random_seed,
            n_sampling,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        score_moe_p_value_fdr[["P-Value", "FDR"]] = compute_empirical_p_values_and_fdrs(
            score_moe_p_value_fdr["Score"],
            concatenate(
                multiprocess(
                    _permute_target_and_match_target_and_data,
                    (
                        (
                            target,
                            data_,
                            random_seed,
                            n_permutation,
                            match_function,
                            n_required_for_match_function,
                            raise_for_n_less_than_required,
                        )
                        for data_ in data_split
                    ),
                    n_job,
                )
            ).flatten(),
            "less_or_great",
            raise_for_bad=False,
        )

    score_moe_p_value_fdr.index = save_indx

    return score_moe_p_value_fdr
