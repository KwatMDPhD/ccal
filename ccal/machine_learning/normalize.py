"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from pandas import Series, DataFrame


def normalize_dataframe_or_series(dataframe, method, axis=None, n_ranks=10000):
    """
    Normalize a DataFrame or Series.
    :param dataframe: DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :param axis: int; None for global, 0 for by-column, and 1 for by-row normalization
    :return: DataFrame or Series; normalized DataFrame or Series
    """

    # TODO: make sure the normalization when size == 0 or range == 0 is correct

    if isinstance(dataframe, Series):  # Series
        return normalize_series(dataframe, method, n_ranks=n_ranks)

    elif isinstance(dataframe, DataFrame):
        if axis == 0 or axis == 1:  # Normalize by axis (Series)
            return dataframe.apply(normalize_series, **{'method': method, 'n_ranks': n_ranks}, axis=axis)

        else:  # Normalize globally
            if method == '-0-':
                obj_mean = dataframe.values.mean()
                obj_std = dataframe.values.std()
                if obj_std == 0:
                    # print_log('Not \'-0-\' normalizing (standard deviation is 0), but \'/ size\' normalizing.')
                    return dataframe / dataframe.size
                else:
                    return (dataframe - obj_mean) / obj_std

            elif method == '0-1':
                obj_min = dataframe.values.min()
                obj_max = dataframe.values.max()
                if obj_max - obj_min == 0:
                    # print_log('Not \'0-1\' normalizing (data range is 0), but \'/ size\' normalizing.')
                    return dataframe / dataframe.size
                else:
                    return (dataframe - obj_min) / (obj_max - obj_min)

            elif method == 'rank':
                # TODO: implement global rank normalization
                raise ValueError('Normalizing combination of \'rank\' & axis=\'all\' has not been implemented yet.')


def normalize_series(series, method, n_ranks=10000):
    """
    Normalize a pandas series.
    :param series: pandas Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :return: pandas Series; normalized Series
    """

    if method == '-0-':
        mean = series.mean()
        std = series.std()
        if std == 0:
            # print_log('Not \'-0-\' normalizing (standard deviation is 0), but \'/ size\' normalizing.')
            return series / series.size
        else:
            return (series - mean) / std
    elif method == '0-1':
        series_min = series.min()
        series_max = series.max()
        if series_max - series_min == 0:
            # print_log('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing.')
            return series / series.size
        else:
            return (series - series_min) / (series_max - series_min)
    elif method == 'rank':
        # NaNs are raked lowest in the ascending ranking
        return series.rank(na_option='top') / series.size * n_ranks
