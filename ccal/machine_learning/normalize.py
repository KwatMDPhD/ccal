"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from pandas import Series, DataFrame


#TODO: enable for ndarray
def normalize_dataframe_or_series(dataframe, method, axis=None, n_ranks=10000,
                                  normalizing_mean=None, normalizing_std=None,
                                  normalizing_min=None, normalizing_max=None,
                                  normalizing_size=None):
    """
    Normalize a DataFrame or Series.
    :param dataframe: DataFrame or Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :param axis: int; None for global, 0 for by-column, and 1 for by-row normalization
    :param normalizing_mean:
    :param normalizing_std:
    :param normalizing_min:
    :param normalizing_max:
    :param normalizing_size:
    :return: DataFrame or Series; normalized DataFrame or Series
    """

    if isinstance(dataframe, Series):  # Series
        return normalize_series(dataframe, method, n_ranks=n_ranks,
                                normalizing_mean=normalizing_mean, normalizing_std=normalizing_std,
                                normalizing_min=normalizing_min, normalizing_max=normalizing_max,
                                normalizing_size=normalizing_size)

    elif isinstance(dataframe, DataFrame):

        if axis == 0 or axis == 1:  # Normalize Series by axis
            return dataframe.apply(normalize_series, **{'method': method, 'n_ranks': n_ranks,
                                                        'normalizing_mean': normalizing_mean,
                                                        'normalizing_std': normalizing_std,
                                                        'normalizing_min': normalizing_min,
                                                        'normalizing_max': normalizing_max,
                                                        'normalizing_size': normalizing_size}, axis=axis)

        else:  # Normalize globally

            # Get size
            if normalizing_size is not None:
                size = normalizing_size
            else:
                size = dataframe.values.size

            if method == '-0-':

                # Get mean
                if normalizing_mean is not None:
                    mean = normalizing_mean
                else:
                    mean = dataframe.values.mean()

                # Get STD
                if normalizing_std is not None:
                    std = normalizing_std
                else:
                    std = dataframe.values.std()

                # Normalize
                if std == 0:
                    print('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing ...')
                    return dataframe / size
                else:
                    return (dataframe - mean) / std

            elif method == '0-1':

                # Get min
                if normalizing_min is not None:
                    min_ = normalizing_min
                else:
                    min_ = dataframe.values.min()

                # Get max
                if normalizing_max is not None:
                    max_ = normalizing_max
                else:
                    max_ = dataframe.values.max()

                # Normalize
                if max_ - min_ == 0:
                    print('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing ...')
                    return dataframe / size
                else:
                    return (dataframe - min_) / (max_ - min_)

            elif method == 'rank':
                raise ValueError('Normalizing combination of \'rank\' & axis=\'all\' has not been implemented yet.')


def normalize_series(series, method, n_ranks=10000,
                     normalizing_mean=None, normalizing_std=None,
                     normalizing_min=None, normalizing_max=None,
                     normalizing_size=None):
    """
    Normalize a pandas series.
    :param series: pandas Series;
    :param method: str; normalization type; {'-0-', '0-1', 'rank'}
    :param n_ranks: number; normalization factor for rank normalization: rank / size * n_ranks
    :param normalizing_mean:
    :param normalizing_std:
    :param normalizing_min:
    :param normalizing_max:
    :param normalizing_size:
    :return: pandas Series; normalized Series
    """

    # Get name
    name = series.name

    # Get size
    if normalizing_size is not None:
        size = normalizing_size
    else:
        size = series.size

    if method == '-0-':

        # Get mean
        if isinstance(normalizing_mean, Series):
            mean = normalizing_mean.ix[name]
        elif normalizing_mean is not None:
            mean = normalizing_mean
        else:
            mean = series.mean()

        # Get STD
        if isinstance(normalizing_std, Series):
            std = normalizing_std.ix[name]
        elif normalizing_std is not None:
            std = normalizing_std
        else:
            std = series.std()

        # Normalize
        if std == 0:
            print('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing ...')
            return series / size
        else:
            return (series - mean) / std

    elif method == '0-1':

        # Get min
        if isinstance(normalizing_min, Series):
            min_ = normalizing_min.ix[name]
        elif normalizing_min is not None:
            min_ = normalizing_min
        else:
            min_ = series.min()

        # Get max
        if isinstance(normalizing_max, Series):
            max_ = normalizing_max.ix[name]
        elif normalizing_max is not None:
            max_ = normalizing_max
        else:
            max_ = series.max()

        # Normalize
        if max_ - min_ == 0:
            print('Not \'0-1\' normalizing (data_range is 0), but \'/ size\' normalizing ...')
            return series / size
        else:
            return (series - min_) / (max_ - min_)

    elif method == 'rank':
        # NaNs are raked lowest in the ascending ranking
        return series.rank(na_option='top') / size * n_ranks
