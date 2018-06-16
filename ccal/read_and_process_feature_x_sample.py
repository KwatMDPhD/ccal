from numpy import nan
from pandas import DataFrame, read_table

from .nd_array.nd_array.normalize_nd_array import normalize_nd_array
from .nd_array.nd_array.shift_and_log_nd_array import shift_and_log_nd_array
from .plot.plot.plot_distributions import plot_distributions
from .plot.plot.plot_heat_map import plot_heat_map
from .support.support.df import drop_df_slice_greedily


def read_and_process_feature_x_sample(feature_x_sample_file_path,
                                      features_to_drop=None,
                                      samples_to_drop=None,
                                      nanize_0=False,
                                      min_n_not_na_unique_value=None,
                                      log=False,
                                      normalization_method=None,
                                      normalization_axis=None):

    print('\nReading and processing {} ...'.format(feature_x_sample_file_path))

    feature_x_sample = read_table(feature_x_sample_file_path, index_col=0)

    print('\tfeature_x_sample.shape: {}'.format(feature_x_sample.shape))

    _print_n_na(feature_x_sample, prefix='\t')

    if feature_x_sample.index.has_duplicates:

        raise ValueError('Feature duplicated.')

    elif feature_x_sample.columns.has_duplicates:

        raise ValueError('Sample duplicated.')

    if features_to_drop is not None:

        print('\nDropping features: {} ...'.format(features_to_drop))

        features_to_drop = feature_x_sample.index & set(features_to_drop)

        feature_x_sample.drop(features_to_drop, inplace=True)

    if samples_to_drop is not None:

        print('\nDropping samples: {} ...'.format(samples_to_drop))

        samples_to_drop = feature_x_sample.columns & set(samples_to_drop)

        feature_x_sample.drop(samples_to_drop, axis=1, inplace=True)

    if nanize_0:

        print('\nNANizing 0 ...')

        is_0 = feature_x_sample == 0

        feature_x_sample[is_0] = nan

        _print_n_na(feature_x_sample, prefix='\t')

    if min_n_not_na_unique_value is not None:

        print('\nDropping slice (min_n_not_na_unique_value={}) greedily ...'.
              format(min_n_not_na_unique_value))

        feature_x_sample = drop_df_slice_greedily(
            feature_x_sample,
            min_n_not_na_unique_value=min_n_not_na_unique_value)

    if log:

        print('\nShifting (as needed) and logging ...')

        feature_x_sample = DataFrame(
            shift_and_log_nd_array(
                feature_x_sample.values, raise_for_bad_value=False),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns)

    if normalization_method is not None:

        print('\n{} normalizing (axis={}) ...'.format(normalization_method,
                                                      normalization_axis))

        feature_x_sample = DataFrame(
            normalize_nd_array(
                feature_x_sample.values,
                normalization_method,
                normalization_axis,
                raise_for_bad_value=False),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns)

    print('\nShape: {}'.format(feature_x_sample.shape))

    _print_n_na(feature_x_sample, prefix='\n')

    if feature_x_sample.size < 1e6:

        plot_heat_map(
            feature_x_sample, xaxis_title='Sample', yaxis_title='Feature')

    plot_distributions(
        ('Feature', 'Sample'),
        (feature_x_sample.isna().sum(axis=1), feature_x_sample.isna().sum()),
        title='NA Distribution',
        xaxis_title='Number of NA')

    return feature_x_sample


def _print_n_na(feature_x_sample, prefix=''):

    n_0 = feature_x_sample.isna().values.sum()

    percent_0 = n_0 / feature_x_sample.size * 100

    print('{}feature_x_sample N NA: {} ({:.2f}%)'.format(
        prefix, n_0, percent_0))
