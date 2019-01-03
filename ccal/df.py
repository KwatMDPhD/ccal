from numpy import asarray


def drop_df_slice_greedily(df, max_na=None, min_n_not_na_unique_value=None):

    shift = int(df.shape[1] < df.shape[0])

    for i in range(df.size):

        shape_before = df.shape

        axis = (i + shift) % 2

        df = drop_df_slice(
            df, axis, max_na=max_na, min_n_not_na_unique_value=min_n_not_na_unique_value
        )

        shape_after = df.shape

        print("Shape: {} =(drop axis {})=> {}".format(shape_before, axis, shape_after))

        if 0 < i and shape_before == shape_after:

            return df


def drop_df_slice(df, axis, max_na=None, min_n_not_na_unique_value=None):

    dropped = asarray((False,) * df.shape[(axis + 1) % 2])

    if max_na is not None:

        if max_na < 1:

            max_n_na = max_na * df.shape[axis]

        else:

            max_n_na = max_na

        dropped |= df.apply(lambda series: max_n_na < series.isna().sum(), axis=axis)

    if min_n_not_na_unique_value is not None:

        dropped |= df.apply(
            lambda series: series[~series.isna()].unique().size
            < min_n_not_na_unique_value,
            axis=axis,
        )

    if axis == 0:

        return df.loc[:, ~dropped]

    elif axis == 1:

        return df.loc[~dropped]


def split_df(df, axis, n_split):

    if not (0 < n_split <= df.shape[axis]):

        raise ValueError(
            "Invalid: 0 < n_split ({}) <= n_slices ({})".format(n_split, df.shape[axis])
        )

    n = df.shape[axis] // n_split

    dfs = []

    for i in range(n_split):

        start_i = i * n

        end_i = (i + 1) * n

        if axis == 0:

            dfs.append(df.iloc[start_i:end_i])

        elif axis == 1:

            dfs.append(df.iloc[:, start_i:end_i])

    i = n * n_split

    if i < df.shape[axis]:

        if axis == 0:

            dfs.append(df.iloc[i:])

        elif axis == 1:

            dfs.append(df.iloc[:, i:])

    return dfs
