from pandas import DataFrame, read_csv
from os.path import join
from scipy.io import mmread
from .process_feature_x_sample import process_feature_x_sample


def read_gene_x_cell(
    mtx_file_path, index_file_path, column_file_path, output_directory_path
):

    gene_x_cell = DataFrame(
        mmread(mtx_file_path).toarray(),
        index=read_csv(index_file_path, sep="\t", header=None).iloc[:, 1],
        columns=read_csv(column_file_path, sep="\t", header=None, squeeze=True),
    )

    gene_x_cell.index.name = "Gene"

    if gene_x_cell.index.has_duplicates:

        print("Gene duplicated; merging duplicates with median...")

        gene_x_cell = gene_x_cell.groupby(level=0).median()

    gene_x_cell.columns.name = "Cell"

    if gene_x_cell.columns.has_duplicates:

        print("Cell duplicated.")

    gene_x_cell = gene_x_cell.sort_index().sort_index(axis=1)

    output_file_path = join(output_directory_path, "gene_x_cell.tsv")

    gene_x_cell.to_csv(output_file_path, sep="\t")

    gene_x_cell__clean__log = process_feature_x_sample(
        gene_x_cell,
        nanize=0,
        min_n_not_na_value=3,
        shift_as_necessary_to_achieve_min_before_logging=1,
        log_base=2,
        plot=False,
    )

    gene_x_cell__clean__log.to_csv(
        output_file_path.replace(".tsv", ".clean.log.tsv"), sep="\t"
    )

    gene_x_cell__clean__log.fillna(0).to_csv(
        output_file_path.replace(".tsv", ".clean.log.na0.tsv"), sep="\t"
    )