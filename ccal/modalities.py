from ccal.information.information.compute_information_coefficient import \
    compute_information_coefficient
from ccal.match.match.make_match_panel import make_match_panel

RANDOM_SEED = 20121020


def differential_gene_expression(
        phenotypes,
        gene_expression,
        output_filename,
        ranking_method=compute_information_coefficient,
        max_number_of_genes_to_show=20,
        number_of_permutations=10,
        title=None,
        random_seed=RANDOM_SEED):
    """
    Sort genes according to their association with a binary phenotype or class vector.
    :param phenotypes: Series; input binary phenotype/class distinction
    :param gene_expression: Dataframe; data matrix with input gene expression profiles
    :param output_filename: str; output files will have this name plus extensions .txt and .pdf
    :param ranking_method:callable; the function to use to compute similarity between phenotypes and gene_expression.
    :param max_number_of_genes_to_show: int; maximum number of genes to show in the heatmap
    :param number_of_permutations: int; number of random permutations to estimate statistical significance (p-values and FDRs)
    :param title: str;
    :param random_seed: int | array; random number generator seed (can be set to a user supplied integer for reproducibility)
    :return: Dataframe; table of genes ranked by Information Coeff vs. phenotype
    """

    gene_scores = make_match_panel(
        phenotypes,
        gene_expression,
        function=ranking_method,
        target_ascending=False,
        n_features=0.99,
        max_n_features=max_number_of_genes_to_show,
        n_samplings=30,
        n_permutations=number_of_permutations,
        random_seed=random_seed,
        target_type='binary',
        title=title,
        file_path_prefix=output_filename)

    return gene_scores


def match_to_profile(phenotypes,
                     gene_expression,
                     output_filename,
                     ranking_method=compute_information_coefficient,
                     phenotypes_row_label=None,
                     max_number_of_genes_to_show=20,
                     number_of_permutations=10,
                     title=None,
                     random_seed=RANDOM_SEED):
    """
    Sort genes according to their association with a continuous phenotype or class vector.
    :param phenotypes: Series; input binary phenotype/class distinction
    :param gene_expression: DataFrame; data matrix with input gene expression profiles
    :param output_filename: str; output files will have this name plus extensions .txt and .pdf
    :param ranking_method:callable; the function to use to compute similarity between phenotypes and gene_expression.
    :param phenotypes_row_label: str; Name of phenotype row when input_phenotype is an array
    :param max_number_of_genes_to_show: int; maximum number of genes to show in the heatmap
    :param number_of_permutations: int; number of random permutations to estimate statistical significance (p-values and FDRs)
    :param title: str;
    :param random_seed: int | array; random number generator seed (can be set to a user supplied integer for reproducibility)
    :return: DataFrame; table of genes ranked by Information Coeff vs. phenotype
    """

    # TODO: add "if phenotypes_row_label is not None.
    # In this case it would check that Phenotypes is None. Only one can be not None.
    # Use phenotypes if both are provided.
    # Check that phenotypes_row_label actually exist in gene_expression DataFrame

    gene_scores = make_match_panel(
        phenotypes,
        gene_expression,
        # max_n_unique_objects_for_drop_slices=1,
        function=ranking_method,
        target_ascending=False,
        n_features=0.99,
        max_n_features=max_number_of_genes_to_show,
        n_samplings=30,
        n_permutations=number_of_permutations,
        random_seed=random_seed,
        target_type='binary',
        title=title,
        file_path_prefix=output_filename)

    return gene_scores
