from numpy import nan
from pandas import DataFrame, read_table

from .nd_array.nd_array.log_nd_array import log_nd_array
from .nd_array.nd_array.normalize_nd_array import normalize_nd_array
from .summarize_feature_x_sample import summarize_feature_x_sample
from .support.support.df import drop_df_slice, drop_df_slice_greedily


def read_and_process_feature_x_sample(
        feature_x_sample_file_path,
        features_to_drop=None,
        samples_to_drop=None,
        nanize=None,
        drop_na_axis=None,
        max_na=None,
        min_n_not_na_unique_value=None,
        log_base=None,
        shift_as_necessary_to_achieve_min_before_logging=None,
        normalization_axis=None,
        normalization_method=None,
        plot=True,
        max_plot_n=int(1e6),
):

    print('Reading and processing {} ...'.format(feature_x_sample_file_path))

    feature_x_sample = read_table(
        feature_x_sample_file_path,
        index_col=0,
    )

    summarize_feature_x_sample(
        feature_x_sample,
        plot=plot,
        max_plot_n=max_plot_n,
    )

    if feature_x_sample.index.has_duplicates:

        raise ValueError('Feature duplicated.')

    elif feature_x_sample.columns.has_duplicates:

        raise ValueError('Sample duplicated.')

    if features_to_drop is not None:

        features_to_drop = feature_x_sample.index & set(features_to_drop)

        print('Dropping features: {} ...'.format(features_to_drop))

        feature_x_sample.drop(
            features_to_drop,
            inplace=True,
        )

        print('Shape: {}'.format(feature_x_sample.shape))

    if samples_to_drop is not None:

        samples_to_drop = feature_x_sample.columns & set(samples_to_drop)

        print('Dropping samples: {} ...'.format(samples_to_drop))

        feature_x_sample.drop(
            samples_to_drop,
            axis=1,
            inplace=True,
        )

        print('Shape: {}'.format(feature_x_sample.shape))

    if features_to_drop is not None or samples_to_drop is not None:

        summarize_feature_x_sample(
            feature_x_sample,
            plot=plot,
            max_plot_n=max_plot_n,
        )

    if nanize is not None:

        print('NANizing <= {} ...'.format(nanize))

        feature_x_sample[feature_x_sample <= nanize] = nan

        summarize_feature_x_sample(
            feature_x_sample,
            plot=plot,
            max_plot_n=max_plot_n,
        )

    if max_na is not None or min_n_not_na_unique_value is not None:

        if min_n_not_na_unique_value == 'max':

            if drop_na_axis is None:

                min_n_not_na_unique_value = min(feature_x_sample.shape)

            else:

                min_n_not_na_unique_value = feature_x_sample.shape[
                    drop_na_axis]

        print(
            'Dropping slice (drop_na_axis={} & max_na={} & min_n_not_na_unique_value={}) ...'.
            format(
                drop_na_axis,
                max_na,
                min_n_not_na_unique_value,
            ))

        if drop_na_axis is None:

            feature_x_sample = drop_df_slice_greedily(
                feature_x_sample,
                max_na=max_na,
                min_n_not_na_unique_value=min_n_not_na_unique_value,
            )

        else:

            feature_x_sample = drop_df_slice(
                feature_x_sample,
                drop_na_axis,
                max_na=max_na,
                min_n_not_na_unique_value=min_n_not_na_unique_value,
            )

        summarize_feature_x_sample(
            feature_x_sample,
            plot=plot,
            max_plot_n=max_plot_n,
        )

    if log_base is not None:

        print(
            'Logging (shift_as_necessary_to_achieve_min_before_logging={} & log_base={}) ...'.
            format(
                shift_as_necessary_to_achieve_min_before_logging,
                log_base,
            ))

        feature_x_sample = DataFrame(
            log_nd_array(
                feature_x_sample.values,
                raise_for_bad_value=False,
                shift_as_necessary_to_achieve_min_before_logging=
                shift_as_necessary_to_achieve_min_before_logging,
                log_base=log_base,
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize_feature_x_sample(
            feature_x_sample,
            plot=plot,
            max_plot_n=max_plot_n,
        )

    if normalization_method is not None:

        print('Axis-{} {} normalizing ...'.format(
            normalization_axis,
            normalization_method,
        ))

        feature_x_sample = DataFrame(
            normalize_nd_array(
                feature_x_sample.values,
                normalization_axis,
                normalization_method,
                raise_for_bad_value=False,
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize_feature_x_sample(
            feature_x_sample,
            plot=plot,
            max_plot_n=max_plot_n,
        )

    return feature_x_sample
