from numpy import full, nan
from numpy.random import get_state, seed, set_state, shuffle

from ._match_target_and_features import _match_target_and_features


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
