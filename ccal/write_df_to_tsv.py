def write_df_to_tsv(df, index_name, tsv_file_path):

    assert not df.index.hasnans

    assert not df.columns.hasnans

    assert not df.index.has_duplicates

    assert not df.columns.has_duplicates

    df = df.sort_index().sort_index(axis=1)

    df.index.name = index_name

    df.to_csv(tsv_file_path, sep="\t")

    return df
