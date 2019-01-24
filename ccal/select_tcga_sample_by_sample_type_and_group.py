from pandas import concat


def select_tcga_sample_by_sample_type_and_group(df, sample_type="01"):

    print(df.shape)

    df = df.loc[:, df.columns.str[13:15] == sample_type]

    print(df.shape)

    duplicated = df.columns.str[:12].duplicated(keep=False)

    df_not_duplicated = df.loc[:, ~duplicated]

    df_not_duplicated.columns = df_not_duplicated.columns.str[:12]

    print(df_not_duplicated.shape)

    df_duplicated = df.loc[:, duplicated]

    print(df_duplicated.shape)

    if df_duplicated.empty:

        df = df_not_duplicated

    else:

        df_duplicated_grouped = df_duplicated.groupby(
            by=df_duplicated.columns.str[:12], axis=1
        ).mean()

        print(df_duplicated_grouped.shape)

        df = concat((df_not_duplicated, df_duplicated_grouped), axis=1)

        print(df.shape)

    return df
