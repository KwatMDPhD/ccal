from numpy import nan
from pandas import DataFrame, read_table

from .nd_array.nd_array.normalize_nd_array import normalize_nd_array
from .nd_array.nd_array.shift_and_log_nd_array import shift_and_log_nd_array
from .plot.plot.plot_distributions import plot_distributions
from .plot.plot.plot_heat_map import plot_heat_map
from .support.support.df import drop_df_slice, drop_df_slice_greedily


def read_and_process_feature_x_sample(feature_x_sample_file_path,
                                      features_to_drop=None,
                                      samples_to_drop=None,
                                      nanize_0=False,
                                      max_na=None,
                                      min_n_not_na_unique_value=None,
                                      drop_na_axis=None,
                                      log=False,
                                      normalization_method=None,
                                      normalization_axis=None):

    print('Reading and processing {} ...'.format(feature_x_sample_file_path))

    feature_x_sample = read_table(feature_x_sample_file_path, index_col=0)

    print('Shape: {}'.format(feature_x_sample.shape))

    if feature_x_sample.index.has_duplicates:

        raise ValueError('Feature duplicated.')

    elif feature_x_sample.columns.has_duplicates:

        raise ValueError('Sample duplicated.')

    if features_to_drop is not None:

        features_to_drop = feature_x_sample.index & set(features_to_drop)

        print('Dropping features: {} ...'.format(features_to_drop))

        feature_x_sample.drop(features_to_drop, inplace=True)

        print('Shape: {}'.format(feature_x_sample.shape))

    if samples_to_drop is not None:

        samples_to_drop = feature_x_sample.columns & set(samples_to_drop)

        print('Dropping samples: {} ...'.format(samples_to_drop))

        feature_x_sample.drop(samples_to_drop, axis=1, inplace=True)

        print('Shape: {}'.format(feature_x_sample.shape))

    _summarize_na(feature_x_sample)

    if nanize_0:

        print('NANizing 0 ...')

        feature_x_sample[feature_x_sample == 0] = nan

        _summarize_na(feature_x_sample, prefix='(After NANizing) ')

    if max_na is not None or min_n_not_na_unique_value is not None:

        if min_n_not_na_unique_value == 'max':

            if drop_na_axis is None:

                min_n_not_na_unique_value = min(feature_x_sample.shape)

            else:

                min_n_not_na_unique_value = feature_x_sample.shape[
                    drop_na_axis]

        print(
            'Dropping slice (max_na={} & min_n_not_na_unique_value={} & drop_na_axis={}) ...'.
            format(max_na, min_n_not_na_unique_value, drop_na_axis))

        if drop_na_axis is None:

            feature_x_sample = drop_df_slice_greedily(
                feature_x_sample,
                max_na=max_na,
                min_n_not_na_unique_value=min_n_not_na_unique_value)

        else:

            feature_x_sample = drop_df_slice(
                feature_x_sample,
                drop_na_axis,
                max_na=max_na,
                min_n_not_na_unique_value=min_n_not_na_unique_value)

        print('Shape: {}'.format(feature_x_sample.shape))

        _summarize_na(feature_x_sample, prefix='(After Dropping Slice) ')

    if log:

        print('Shifting (as needed) and logging ...')

        feature_x_sample = DataFrame(
            shift_and_log_nd_array(
                feature_x_sample.values, raise_for_bad_value=False),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns)

    if normalization_method is not None:

        print('{} normalizing (axis={}) ...'.format(normalization_method,
                                                    normalization_axis))

        feature_x_sample = DataFrame(
            normalize_nd_array(
                feature_x_sample.values,
                normalization_method,
                normalization_axis,
                raise_for_bad_value=False),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns)

    if feature_x_sample.size < 1e6:

        plot_heat_map(
            feature_x_sample, xaxis_title='Sample', yaxis_title='Feature')

    return feature_x_sample


def _summarize_na(feature_x_sample, prefix=''):

    n_0 = feature_x_sample.isna().values.sum()

    percent_0 = n_0 / feature_x_sample.size * 100

    print('{}N NA: {} ({:.2f}%)'.format(prefix, n_0, percent_0))

    isna__feature_x_sample = feature_x_sample.isna()

    if isna__feature_x_sample.values.any():

        plot_distributions(
            ('Feature', 'Sample'),
            (isna__feature_x_sample.sum(axis=1), isna__feature_x_sample.sum()),
            title='{}NA Distribution'.format(prefix),
            xaxis_title='Number of NA')
