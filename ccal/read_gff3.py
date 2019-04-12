from pandas import read_csv


def read_gff3(gff3_or_gff3_gz_file_path, only_type_to_keep=None):

    df = read_csv(gff3_or_gff3_gz_file_path, sep="\t", comment="#")

    df.columns = (
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

        df = df[df["type"] == only_type_to_keep]

    return df
