from numpy import nan
from pandas import DataFrame

from .drop_dataframe_slice import drop_dataframe_slice
from .drop_dataframe_slice_greedily import drop_dataframe_slice_greedily
from .log_nd_array import log_nd_array
from .normalize_series_or_dataframe import normalize_series_or_dataframe
from .summarize_feature_x_sample import summarize_feature_x_sample


def process_feature_x_sample(
    feature_x_sample,
    features_to_drop=None,
    samples_to_drop=None,
    nanize=None,
    drop_axis=None,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
    shift_as_necessary_to_achieve_min_before_logging=None,
    log_base=None,
    normalization_axis=None,
    normalization_method=None,
    clip_min=None,
    clip_max=None,
    **summarize_feature_x_sample_keyword_arguments,
):

    summarize_feature_x_sample(
        feature_x_sample, **summarize_feature_x_sample_keyword_arguments
    )

    if feature_x_sample.index.has_duplicates:

        raise ValueError(f"{feature_x_sample.index.name} duplicated.")

    elif feature_x_sample.columns.has_duplicates:

        raise ValueError(f"{feature_x_sample.columns.name} duplicated.")

    shape_before_drop = feature_x_sample.shape

    if features_to_drop is not None:

        features_to_drop = feature_x_sample.index & set(features_to_drop)

        print(f"Dropping {feature_x_sample.index.name}: {features_to_drop} ...")

        feature_x_sample.drop(features_to_drop, inplace=True)

        print(f"Shape: {feature_x_sample.shape}")

    if samples_to_drop is not None:

        samples_to_drop = feature_x_sample.columns & set(samples_to_drop)

        print(f"Dropping {feature_x_sample.columns.name}: {features_to_drop} ...")

        feature_x_sample.drop(samples_to_drop, axis=1, inplace=True)

        print(f"Shape: {feature_x_sample.shape}")

    if feature_x_sample.shape != shape_before_drop:

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if nanize is not None:

        print(f"NANizing <= {nanize} ...")

        feature_x_sample[feature_x_sample <= nanize] = nan

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if (
        max_na is not None
        or min_n_not_na_value is not None
        or min_n_not_na_unique_value is not None
    ):

        print("Dropping slice ...")

        shape_before_drop = feature_x_sample.shape

        if drop_axis is None:

            feature_x_sample = drop_dataframe_slice_greedily(
                feature_x_sample,
                drop_axis,
                max_na=max_na,
                min_n_not_na_value=min_n_not_na_value,
                min_n_not_na_unique_value=min_n_not_na_unique_value,
            )

        else:

            feature_x_sample = drop_dataframe_slice(
                feature_x_sample,
                drop_axis,
                max_na=max_na,
                min_n_not_na_value=min_n_not_na_value,
                min_n_not_na_unique_value=min_n_not_na_unique_value,
            )

        if feature_x_sample.shape != shape_before_drop:

            summarize_feature_x_sample(
                feature_x_sample, **summarize_feature_x_sample_keyword_arguments
            )

    if log_base is not None:

        print(
            f"Logging (shift_as_necessary_to_achieve_min_before_logging={shift_as_necessary_to_achieve_min_before_logging}, log_base={log_base}) ..."
        )

        feature_x_sample = DataFrame(
            log_nd_array(
                feature_x_sample.values,
                raise_for_bad=False,
                shift_as_necessary_to_achieve_min_before_logging=shift_as_necessary_to_achieve_min_before_logging,
                log_base=log_base,
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if normalization_method is not None:

        print(f"Axis-{normalization_axis} {normalization_method} normalizing ...")

        feature_x_sample = normalize_series_or_dataframe(
            feature_x_sample, normalization_axis, normalization_method
        )

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if clip_min is not None:

        print(f"|-{clip_min} clipping  ...")

        feature_x_sample.clip(lower=clip_min, inplace=True)

    if clip_max is not None:

        print(f"{clip_max}-| clipping  ...")

        feature_x_sample.clip(upper=clip_max, inplace=True)

    if clip_min is not None or clip_max is not None:

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    return feature_x_sample
