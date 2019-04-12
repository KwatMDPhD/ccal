from pandas import concat


def select_tcga_sample_by_sample_type_and_group(df, sample_type):

    df_sample_type_selected = df.loc[:, df.columns.str[13:15] == sample_type]

    print("{}: df_sample_type_selected".format(df_sample_type_selected.shape))

    duplicated = df_sample_type_selected.columns.str[:12].duplicated(keep=False)

    df_sample_type_selected_not_duplicated = df_sample_type_selected.loc[:, ~duplicated]

    df_sample_type_selected_not_duplicated.columns = df_sample_type_selected_not_duplicated.columns.str[
        :12
    ]

    print(
        "{}: df_sample_type_selected_not_duplicated".format(
            df_sample_type_selected_not_duplicated.shape
        )
    )

    df_sample_type_selected_duplicated = df_sample_type_selected.loc[:, duplicated]

    print(
        "{}: df_sample_type_selected_duplicated".format(
            df_sample_type_selected_duplicated.shape
        )
    )

    if df_sample_type_selected_duplicated.size:

        df_sample_type_selected_duplicated = df_sample_type_selected_duplicated.groupby(
            by=df_sample_type_selected_duplicated.columns.str[:12], axis=1
        ).median()

    return concat(
        (df_sample_type_selected_not_duplicated, df_sample_type_selected_duplicated),
        axis=1,
    )
