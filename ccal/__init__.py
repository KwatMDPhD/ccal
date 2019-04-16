from os.path import abspath

from .add_conda_to_path import add_conda_to_path
from .ALMOST_ZERO import ALMOST_ZERO
from .apply_function_on_2_1d_arrays import apply_function_on_2_1d_arrays
from .apply_function_on_2_2d_arrays_slices import apply_function_on_2_2d_arrays_slices
from .apply_function_on_2_dfs_slices import apply_function_on_2_dfs_slices
from .call_function_with_multiprocess import call_function_with_multiprocess
from .cast_object_to_builtin import cast_object_to_builtin
from .check_nd_array_for_bad import check_nd_array_for_bad
from .clean_and_write_df_to_tsv import clean_and_write_df_to_tsv
from .clip_nd_array_by_standard_deviation import clip_nd_array_by_standard_deviation
from .cluster_2d_array import cluster_2d_array
from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .CODON_AMINO_ACID import CODON_AMINO_ACID
from .COLORS import COLORS
from .compute_1d_array_context import compute_1d_array_context
from .compute_1d_array_entropy import compute_1d_array_entropy
from .compute_bandwidths import compute_bandwidths
from .compute_coclustering_fraction_from_clustering_x_element import (
    compute_coclustering_fraction_from_clustering_x_element,
)
from .compute_correlation_distance_between_2_1d_arrays import (
    compute_correlation_distance_between_2_1d_arrays,
)
from .compute_empirical_p_value import compute_empirical_p_value
from .compute_empirical_p_values_and_fdrs import compute_empirical_p_values_and_fdrs
from .compute_information_coefficient_between_2_1d_arrays import (
    compute_information_coefficient_between_2_1d_arrays,
)
from .compute_information_distance_between_2_1d_arrays import (
    compute_information_distance_between_2_1d_arrays,
)
from .compute_joint_probability import compute_joint_probability
from .compute_kullback_leibler_divergence_between_2_pdfs import (
    compute_kullback_leibler_divergence_between_2_pdfs,
)
from .compute_matrix_norm import compute_matrix_norm
from .compute_nd_array_margin_of_error import compute_nd_array_margin_of_error
from .compute_posterior_probability import compute_posterior_probability
from .correlate_2_1d_arrays import correlate_2_1d_arrays
from .count_gene_impacts_from_variant_dicts import count_gene_impacts_from_variant_dicts
from .create_gitkeep import create_gitkeep
from .DATA_DIRECTORY_PATH import DATA_DIRECTORY_PATH
from .download_and_parse_geo import download_and_parse_geo
from .download_url import download_url
from .drop_df_slice import drop_df_slice
from .drop_df_slice_greedily import drop_df_slice_greedily
from .echo_or_print_str import echo_or_print_str
from .establish_path import establish_path
from .estimate_kernel_density import estimate_kernel_density
from .exit_ import exit_
from .FeatureHDF5 import FeatureHDF5
from .fit_skew_t_pdf_on_1d_array import fit_skew_t_pdf_on_1d_array
from .fit_skew_t_pdf_on_each_df_row import fit_skew_t_pdf_on_each_df_row
from .flatten_nested_iterable import flatten_nested_iterable
from .Genome import Genome
from .get_colormap_colors import get_colormap_colors
from .get_conda_environments import get_conda_environments
from .get_conda_prefix import get_conda_prefix
from .get_gff3_attribute import get_gff3_attribute
from .get_git_versions import get_git_versions
from .get_installed_pip_libraries import get_installed_pip_libraries
from .get_intersections_between_2_1d_arrays import get_intersections_between_2_1d_arrays
from .get_machine import get_machine
from .get_name_within_function import get_name_within_function
from .get_shell_environment import get_shell_environment
from .get_target_grid_indices import get_target_grid_indices
from .get_triangulation_edges_from_point_x_dimension import (
    get_triangulation_edges_from_point_x_dimension,
)
from .get_variant_start_and_end_positions import get_variant_start_and_end_positions
from .get_vcf_allelic_frequencies import get_vcf_allelic_frequencies
from .get_vcf_genotype import get_vcf_genotype
from .get_vcf_info import get_vcf_info
from .get_vcf_info_ann import get_vcf_info_ann
from .get_vcf_population_allelic_frequencies import (
    get_vcf_population_allelic_frequencies,
)
from .get_vcf_sample_format import get_vcf_sample_format
from .get_vcf_variants_by_region import get_vcf_variants_by_region
from .GPSMap import GPSMap
from .group_and_apply_function_on_each_group_in_iterable import (
    group_and_apply_function_on_each_group_in_iterable,
)
from .gsea import gsea
from .hierarchical_consensus_cluster import hierarchical_consensus_cluster
from .hierarchical_consensus_cluster_with_ks import (
    hierarchical_consensus_cluster_with_ks,
)
from .ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays import (
    ignore_bad_and_compute_euclidean_distance_between_2_1d_arrays,
)
from .index_gff3_df_by_name import index_gff3_df_by_name
from .infer import infer
from .infer_assuming_independence import infer_assuming_independence
from .initialize_logger import initialize_logger
from .install_and_activate_conda import install_and_activate_conda
from .install_python_libraries import install_python_libraries
from .is_in_git_repository import is_in_git_repository
from .is_program import is_program
from .is_sorted_nd_array import is_sorted_nd_array
from .is_str_version import is_str_version
from .is_valid_conda_directory_path import is_valid_conda_directory_path
from .log_and_return_response import log_and_return_response
from .log_nd_array import log_nd_array
from .make_binary_df_from_categorical_series import (
    make_binary_df_from_categorical_series,
)
from .make_categorical_colors import make_categorical_colors
from .make_colorscale_from_colors import make_colorscale_from_colors
from .make_context_matrix import make_context_matrix
from .make_coordinates_for_reflection import make_coordinates_for_reflection
from .make_match_panel import make_match_panel
from .make_match_panel_annotations import make_match_panel_annotations
from .make_match_panels import make_match_panels
from .make_mesh_grid_and_ravel import make_mesh_grid_and_ravel
from .make_summary_match_panel import make_summary_match_panel
from .make_variant_dict_consistent import make_variant_dict_consistent
from .make_variant_dict_from_vcf_row import make_variant_dict_from_vcf_row
from .map_iterable_objects_to_ints import map_iterable_objects_to_ints
from .match_randomly_sampled_target_and_data_to_compute_margin_of_errors import (
    match_randomly_sampled_target_and_data_to_compute_margin_of_errors,
)
from .match_target_and_data import match_target_and_data
from .match_target_and_data_and_compute_statistics import (
    match_target_and_data_and_compute_statistics,
)
from .merge_dicts_with_function import merge_dicts_with_function
from .mf_by_multiple_V_and_H import mf_by_multiple_V_and_H
from .mf_by_multiplicative_update import mf_by_multiplicative_update
from .mf_consensus_cluster import mf_consensus_cluster
from .mf_consensus_cluster_with_ks import mf_consensus_cluster_with_ks
from .nmf_by_sklearn import nmf_by_sklearn
from .NONE_STRS import NONE_STRS
from .normalize_cell_line_names import normalize_cell_line_names
from .normalize_contig import normalize_contig
from .normalize_file_name import normalize_file_name
from .normalize_git_url import normalize_git_url
from .normalize_nd_array import normalize_nd_array
from .normalize_path import normalize_path
from .permute_target_and_match_target_and_data import (
    permute_target_and_match_target_and_data,
)
from .pick_nd_array_colors import pick_nd_array_colors
from .plot_and_save import plot_and_save
from .plot_bayesian_nomogram import plot_bayesian_nomogram
from .plot_bubble_map import plot_bubble_map
from .plot_context import plot_context
from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram
from .plot_scatter_and_annotate import plot_scatter_and_annotate
from .process_feature_x_sample import process_feature_x_sample
from .RANDOM_SEED import RANDOM_SEED
from .read_firebrowse_copynumber_gistic2 import read_firebrowse_copynumber_gistic2
from .read_firebrowse_correlate_copynumber_vs_mrnaseq import (
    read_firebrowse_correlate_copynumber_vs_mrnaseq,
)
from .read_firebrowse_mutsignozzlereport2cv import read_firebrowse_mutsignozzlereport2cv
from .read_gff3 import read_gff3
from .read_gmt import read_gmt
from .read_gps_map import read_gps_map
from .read_json import read_json
from .read_matrix_market import read_matrix_market
from .read_where_and_map_column_name_on_hdf5_table import (
    read_where_and_map_column_name_on_hdf5_table,
)
from .reduce_point_x_dimension_dimension import reduce_point_x_dimension_dimension
from .rescale_x_y_coordiantes_in_polar_coordiante import (
    rescale_x_y_coordiantes_in_polar_coordiante,
)
from .reverse_complement_dna_sequence import reverse_complement_dna_sequence
from .reverse_transcribe_rna_sequence import reverse_transcribe_rna_sequence
from .run_command import run_command
from .sample_from_each_series_value import sample_from_each_series_value
from .select_and_group_feature_x_tcga_sample_by_sample_type import (
    select_and_group_feature_x_tcga_sample_by_sample_type,
)
from .select_gene_symbol import select_gene_symbol
from .select_series_indices import select_series_indices
from .shuffle_each_2d_array_slice import shuffle_each_2d_array_slice
from .single_sample_gsea import single_sample_gsea
from .single_sample_gseas import single_sample_gseas
from .skip_quote_and_split_str import skip_quote_and_split_str
from .solve_ax_equal_b import solve_ax_equal_b
from .solve_for_H import solve_for_H
from .split_codons import split_codons
from .split_df import split_df
from .summarize_feature_x_sample import summarize_feature_x_sample
from .title_str import title_str
from .train_and_classify import train_and_classify
from .train_and_regress import train_and_regress
from .transcribe_dna_sequence import transcribe_dna_sequence
from .translate_nucleotide_sequence import translate_nucleotide_sequence
from .untitle_str import untitle_str
from .update_H_by_multiplicative_update import update_H_by_multiplicative_update
from .update_variant_dict import update_variant_dict
from .update_W_by_multiplicative_update import update_W_by_multiplicative_update
from .VariantHDF5 import VariantHDF5
from .VCF_ANN_FIELDS import VCF_ANN_FIELDS
from .VCF_COLUMNS import VCF_COLUMNS
from .write_gps_map import write_gps_map
from .write_json import write_json

VERSION = "1.1.0"
print("CCAL version {} @ {}".format(VERSION, abspath(__file__)))
