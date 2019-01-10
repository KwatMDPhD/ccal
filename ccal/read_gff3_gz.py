from pandas import read_table


def read_gff3_gz(gff3_gz_file_path, only_type_to_keep=None):

    gff3_df = read_table(gff3_gz_file_path, comment="#")

    gff3_df.columns = (
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    )

    if only_type_to_keep is not None:

        gff3_df = gff3_df[gff3_df["type"] == only_type_to_keep]

    return gff3_df
