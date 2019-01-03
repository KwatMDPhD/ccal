from math import ceil

from numpy import apply_along_axis, array_split, concatenate, full, nan
from numpy.random import choice, get_state, seed, set_state, shuffle
from pandas import DataFrame

from .nd_array.nd_array.apply_function_on_2_1d_arrays import (
    apply_function_on_2_1d_arrays,
)
from .nd_array.nd_array.compute_empirical_p_values_and_fdrs import (
    compute_empirical_p_values_and_fdrs,
)
from .nd_array.nd_array.compute_nd_array_margin_of_error import (
    compute_nd_array_margin_of_error,
)
from .support.support.multiprocess import multiprocess
from .support.support.series import get_extreme_series_indices


def _match(
    target,
    features,
    n_job,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
    extreme_feature_threshold,
    random_seed,
    n_sampling,
    n_permutation,
):

    score_moe_p_value_fdr = DataFrame(columns=("Score", "0.95 MoE", "P-Value", "FDR"))

    n_job = min(features.shape[0], n_job)

    print(
        "Computing score using {} with {} process ...".format(
            match_function.__name__, n_job
        )
    )

    features_split = array_split(features, n_job)

    score_moe_p_value_fdr["Score"] = concatenate(
        multiprocess(
            _match_target_and_features,
            (
                (
                    target,
                    features_,
                    match_function,
                    n_required_for_match_function,
                    raise_for_n_less_than_required,
                )
                for features_ in features_split
            ),
            n_job,
        )
    )

    indices = get_extreme_series_indices(
        score_moe_p_value_fdr["Score"], extreme_feature_threshold
    )

    if len(indices):

        moes = _match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
            target,
            features[indices],
            random_seed,
            n_sampling,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        score_moe_p_value_fdr.loc[indices, "0.95 MoE"] = moes

        random_scores = concatenate(
            multiprocess(
                _permute_target_and_match_target_and_features,
                (
                    (
                        target,
                        features_,
                        random_seed,
                        n_permutation,
                        match_function,
                        n_required_for_match_function,
                        raise_for_n_less_than_required,
                    )
                    for features_ in features_split
                ),
                n_job,
            )
        ).flatten()

        p_values, fdrs = compute_empirical_p_values_and_fdrs(
            score_moe_p_value_fdr["Score"],
            random_scores,
            "less_or_great",
            raise_for_bad=False,
        )

        score_moe_p_value_fdr["P-Value"] = p_values

        score_moe_p_value_fdr["FDR"] = fdrs

    return score_moe_p_value_fdr


def _match_randomly_sampled_target_and_features_to_compute_margin_of_errors(
    target,
    features,
    random_seed,
    n_sampling,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    print("Computing MoE with {} sampling ...".format(n_sampling))

    seed(random_seed)

    feature_x_sampling = full((features.shape[0], n_sampling), nan)

    n_sample = ceil(0.632 * target.size)

    for i in range(n_sampling):

        random_indices = choice(target.size, size=n_sample, replace=True)

        sampled_target = target[random_indices]

        sampled_features = features[:, random_indices]

        random_state = get_state()

        feature_x_sampling[:, i] = _match_target_and_features(
            sampled_target,
            sampled_features,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        set_state(random_state)

    return apply_along_axis(
        compute_nd_array_margin_of_error, 1, feature_x_sampling, raise_for_bad=False
    )


def _permute_target_and_match_target_and_features(
    target,
    features,
    random_seed,
    n_permutation,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    print("Computing p-value and FDR with {} permutation ...".format(n_permutation))

    seed(random_seed)

    feature_x_permutation = full((features.shape[0], n_permutation), nan)

    permuted_target = target.copy()

    for i in range(n_permutation):

        shuffle(permuted_target)

        random_state = get_state()

        feature_x_permutation[:, i] = _match_target_and_features(
            permuted_target,
            features,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        set_state(random_state)

    return feature_x_permutation


def _match_target_and_features(
    target,
    features,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    return apply_along_axis(
        apply_function_on_2_1d_arrays,
        1,
        features,
        target,
        match_function,
        n_required=n_required_for_match_function,
        raise_for_n_less_than_required=raise_for_n_less_than_required,
        raise_for_bad=False,
    )
