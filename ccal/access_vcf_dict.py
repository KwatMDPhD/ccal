from gzip import open as gzip_open

from numpy import nan
from pandas import DataFrame, read_table

from .access_vcf import (
    VCF_COLUMNS,
    get_vcf_sample_format,
    parse_vcf_row_and_make_variant_dict,
)
from .str_ import split_str_ignoring_inside_quotes


def read_vcf_gz_and_make_vcf_dict(vcf_gz_file_path, simplify=True, n_info_ann=1):

    vcf_dict = {
        "meta_information": {},
        "vcf_df": None,
        "variant_dict": [],
        "clean_vcf_df": None,
    }

    print("\nParsing meta-information lines ...")

    with gzip_open(vcf_gz_file_path) as vcf_gz_file:

        n_row_to_skip = 0

        for line in vcf_gz_file:

            line = line.decode()

            if line.startswith("##"):

                n_row_to_skip += 1

                line = line[2:]

                field, value = line.split("=", maxsplit=1)

                if not (value.startswith("<") and value.endswith(">")):

                    vcf_dict["meta_information"][field] = value

                else:

                    value = value.strip("<>")

                    value_split = split_str_ignoring_inside_quotes(value, ",")

                    id_, id_name = value_split.pop(0).split(sep="=")

                    if id_ != "ID":

                        raise ValueError("ID must be the 1st value in {}.".format(line))

                    id_dict = {
                        field: value.strip("'\"")
                        for field, value in (
                            field_value.split("=", maxsplit=1)
                            for field_value in value_split
                        )
                    }

                    if field in vcf_dict["meta_information"]:

                        if id_name in vcf_dict["meta_information"][field]:

                            raise ValueError("Duplicated ID {}.".format(id_name))

                        else:

                            vcf_dict["meta_information"][field][id_name] = id_dict

                    else:

                        vcf_dict["meta_information"][field] = {id_name: id_dict}

            else:

                break

    print("\nReading .vcf DataFrame ...")

    vcf_df = read_table(vcf_gz_file_path, skiprows=n_row_to_skip)

    columns = vcf_df.columns.tolist()

    columns[0] = columns[0][1:]

    vcf_df.columns = columns

    vcf_dict["vcf_df"] = vcf_df

    _describe_vcf_df(vcf_dict["vcf_df"])

    if simplify:

        print("\nMaking variant dicts ...")

        vcf_dict["variant_dict"] = vcf_df.apply(
            parse_vcf_row_and_make_variant_dict, axis=1, n_info_ann=n_info_ann
        ).tolist()

        print("\nMaking clean .vcf DataFrame ...")

        vcf_dict["clean_vcf_df"] = _make_clean_vcf_df(vcf_dict["variant_dict"])

    return vcf_dict


def _describe_vcf_df(vcf_df):

    print("\nCHROM value counts:")

    print(vcf_df["CHROM"].value_counts())

    print("\nREF value counts:")

    print(vcf_df["REF"].value_counts())

    print("\nALT value counts:")

    print(vcf_df["ALT"].value_counts())

    print("\nQUAL description:")

    qual = vcf_df["QUAL"]

    qual = qual[qual.astype(str) != "."]

    print(qual.astype(float).describe())

    for sample in vcf_df.columns[9:]:

        print("\n{} GT value counts:".format(sample))

        try:

            print(
                vcf_df.apply(
                    lambda row: get_vcf_sample_format(row["FORMAT"], row[sample], "GT"),
                    axis=1,
                ).value_counts()
            )

        except ValueError:

            pass

        print("\n{} DP description:".format(sample))

        try:

            print(
                vcf_df.apply(
                    lambda row: get_vcf_sample_format(row["FORMAT"], row[sample], "DP"),
                    axis=1,
                )
                .astype(int)
                .describe()
            )

        except ValueError:

            pass


def _make_clean_vcf_df(variant_dicts):

    vcf_columns = VCF_COLUMNS[:-2]

    info_fields = ("CLNSIG", "CLNDN")

    info_ann_fields = ("gene_name", "transcript_biotype", "effect", "impact")

    columns = vcf_columns + info_fields + info_ann_fields

    vcf_df_rows = []

    for i, variant_dict in enumerate(variant_dicts):

        if variant_dict["FILTER"] == "PASS":

            row = tuple(variant_dict[c] for c in vcf_columns) + tuple(
                variant_dict.get(field, nan) for field in info_fields
            )

            ann_dicts = variant_dict.get("ANN")

            if ann_dicts is not None:

                for ann_i, ann_dict in ann_dicts.items():

                    vcf_df_rows.append(
                        row
                        + tuple(ann_dict[ann_field] for ann_field in info_ann_fields)
                    )

    return (
        DataFrame(vcf_df_rows, columns=columns).drop_duplicates().set_index("gene_name")
    )
