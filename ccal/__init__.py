from os.path import abspath, dirname

VERSION = "0.9.0"

print("CCAL version {} @ {}".format(VERSION, abspath(__file__)))

DATA_DIRECTORY_PATH = "{}/../data".format(dirname(__file__))

from .plot_bar import plot_bar
from .dict_ import merge_dicts_with_callable
from .estimate_kernel_density import estimate_kernel_density
from .make_colorscale_from_colors import make_colorscale_from_colors
from .compression import gzip_compress_file
from .compression import gzip_decompress_file
from .compression import gzip_decompress_and_bgzip_compress_file
from .compute_context import compute_context
from .plot_distributions import plot_distributions
from .get_chromosome_size_from_fasta_gz import get_chromosome_size_from_fasta_gz
from .gsea import gsea
from .plot_bayesian_nomogram import plot_bayesian_nomogram
from .nmf_by_multiple_V_and_H import nmf_by_multiple_V_and_H
from .index_gff3_df_by_name import index_gff3_df_by_name
from .bgzip_and_tabix import bgzip_and_tabix
from .compute_nd_array_margin_of_error import compute_nd_array_margin_of_error
from .nmf_by_sklearn import nmf_by_sklearn
from .plot_pie import plot_pie
from .shuffle_each_2d_array_slice import shuffle_each_2d_array_slice
from .nmf_consensus_cluster_with_ks import nmf_consensus_cluster_with_ks
from .compute_joint_probability import compute_joint_probability
from .compute_mutational_signature_enrichment import (
    compute_mutational_signature_enrichment,
)
from .nd_array_is_sorted import nd_array_is_sorted
from .VariantHDF5 import VariantHDF5
from .conda import install_and_activate_conda
from .conda import add_conda_to_path
from .conda import conda_is_installed
from .conda import get_conda_environments
from .conda import get_conda_prefix
from .GPSMap import GPSMap
from .make_case_annotations import make_case_annotations
from .make_summary_match_panel import make_summary_match_panel
from .plot_color_text import plot_color_text
from .exit_ import exit_
from .plot_heat_map import plot_heat_map
from .access_gmt import read_gmts
from .access_gmt import read_gmt
from .access_gmt import write_gmt
from .log_nd_array import log_nd_array
from .select_series_low_and_high_index import select_series_low_and_high_index
from .plot_context import plot_context
from .make_match_panel import make_match_panel
from .series import cast_series_to_builtins
from .series import make_membership_df_from_categorical_series
from .series import get_extreme_series_indices
from .get_gff3_attribute import get_gff3_attribute
from .plot_table import plot_table
from .mds import mds
from .solve_ax_equal_b import solve_ax_equal_b
from .cluster_2d_array_slices import cluster_2d_array_slices
from .process_fasta import faidx_fasta
from .read_where_and_map_column_names import read_where_and_map_column_names
from .access_vcf import get_variants_from_vcf_gz
from .access_vcf import parse_vcf_row_and_make_variant_dict
from .access_vcf import update_variant_dict
from .access_vcf import count_gene_impacts_from_variant_dicts
from .access_vcf import get_vcf_info
from .access_vcf import get_vcf_info_ann
from .access_vcf import get_vcf_sample_format
from .access_vcf import get_variant_start_and_end_positions
from .access_vcf import get_variant_type
from .access_vcf import is_inframe
from .access_vcf import get_maf_variant_classification
from .access_vcf import get_genotype
from .access_vcf import get_allelic_frequencies
from .access_vcf import get_population_allelic_frequencies
from .access_vcf import count_vcf_gz_rows
from .process_sequence import transcribe_dna_sequence
from .process_sequence import reverse_transcribe_rna_sequence
from .process_sequence import reverse_complement_dna_sequence
from .process_sequence import translate_nucleotide_sequence
from .process_sequence import split_codons
from .establish_fai_index import establish_fai_index
from .path import establish_path
from .path import copy_path
from .path import remove_paths
from .path import remove_path
from .path import clean_path
from .path import clean_name
from .read_mutsignozzlereport2cv import read_mutsignozzlereport2cv
from .train_and_classify import train_and_classify
from .access_gct import read_gct
from .access_gct import write_gct
from .download_clinvar_vcf_gz import download_clinvar_vcf_gz
from .clip_nd_array_by_standard_deviation import clip_nd_array_by_standard_deviation
from .normalize_contig import normalize_contig
from .access_vcf_dict import read_vcf_gz_and_make_vcf_dict
from .fit_skew_t_pdfs import fit_skew_t_pdfs
from .infer import infer
from .check_nd_array_for_bad import check_nd_array_for_bad
from .make_categorical_colors import make_categorical_colors
from .volume import mount_volume
from .volume import unmount_volume
from .volume import get_volume_name
from .volume import make_volume_dict
from .FeatureHDF5 import FeatureHDF5
from .get_colormap_colors import get_colormap_colors
from .python import get_installed_pip_libraries
from .python import install_python_libraries
from .python import get_object_reference
from .df import drop_df_slice_greedily
from .df import drop_df_slice
from .df import split_df
from .read_gff3_gz import read_gff3_gz
from .compute_information_coefficient import compute_information_coefficient
from .make_reference_genome import make_reference_genome
from .get_intersections_between_2_1d_arrays import get_intersections_between_2_1d_arrays
from .select_gene_symbol import select_gene_symbol
from .download_and_parse_geo_data import download_and_parse_geo_data
from .plot_points import plot_points
from .plot_violin_or_box import plot_violin_or_box
from .json_ import read_json
from .json_ import write_json
from .nmf_consensus_cluster import nmf_consensus_cluster
from .iterable import group_iterable
from .iterable import flatten_nested_iterable
from .iterable import replace_bad_objects_in_iterable
from .iterable import group_and_apply_function_on_each_group_in_iterable
from .iterable import get_unique_iterable_objects_in_order
from .iterable import make_object_int_mapping
from .cross_validate import cross_validate
from .read_correlate_copynumber_vs_mrnaseq import read_correlate_copynumber_vs_mrnaseq
from .make_colorscale import make_colorscale
from .math import rescale_x_y_coordiantes_in_polar_coordiante
from .make_mesh_grid_coordinates_per_axis import make_mesh_grid_coordinates_per_axis
from .hierarchical_consensus_cluster_with_ks import (
    hierarchical_consensus_cluster_with_ks,
)
from .multiprocess import multiprocess
from .compute_empirical_p_values_and_fdrs import compute_empirical_p_values_and_fdrs
from .access_gps_map import dump_gps_map
from .access_gps_map import load_gps_map
from .process_bam import sort_and_index_bam_using_samtools_sort_and_index
from .process_bam import index_bam_using_samtools_index
from .process_bam import mark_duplicates_in_bam_using_picard_markduplicates
from .process_bam import check_bam_using_samtools_flagstat
from .process_bam import get_variants_from_bam_using_freebayes_and_multiprocess
from .process_bam import get_variants_from_bam_using_freebayes
from .process_bam import get_variants_from_bam_using_strelka
from .get_function_name import get_function_name
from .compute_correlation_distance import compute_correlation_distance
from .get_1d_array_unique_objects_in_order import get_1d_array_unique_objects_in_order
from .summarize_feature_x_sample import summarize_feature_x_sample
from .process_vcf_gz import concatenate_vcf_gzs_using_bcftools_concat
from .process_vcf_gz import rename_chromosome_of_vcf_gz_using_bcftools_annotate
from .process_vcf_gz import annotate_vcf_gz_using_snpeff
from .process_vcf_gz import annotate_vcf_gz_using_bcftools_annotate
from .process_vcf_gz import filter_vcf_gz_using_bcftools_view
from .network import download
from .network import get_open_port
from .compute_bandwidths import compute_bandwidths
from .single_sample_gseas import single_sample_gseas
from .hierarchical_consensus_cluster import hierarchical_consensus_cluster
from .nmf_by_multiplicative_update import nmf_by_multiplicative_update
from .make_comparison_panel import make_comparison_panel
from .compute_empirical_p_value import compute_empirical_p_value
from .train_and_regress import train_and_regress
from .compute_entropy import compute_entropy
from .infer_assuming_independence import infer_assuming_independence
from .correlate import correlate
from .subprocess_ import run_command
from .subprocess_ import run_command_and_monitor
from .read_matrix_market import read_matrix_market
from .fit_skew_t_pdf import fit_skew_t_pdf
from .plot_bubble_map import plot_bubble_map
from .solve_for_H import solve_for_H
from .make_match_panels import make_match_panels
from .str_ import cast_str_to_builtins
from .str_ import title_str
from .str_ import untitle_str
from .str_ import split_str_ignoring_inside_quotes
from .str_ import str_is_version
from .str_ import make_file_name_from_str
from .read_copynumber_gistic2 import read_copynumber_gistic2
from .process_feature_x_sample import process_feature_x_sample
from .access_maf import split_maf_by_tumor_sample_barcode
from .access_maf import make_maf_from_vcf
from .write_dict import write_dict
from .git import create_gitkeep
from .git import get_git_versions
from .git import clean_git_url
from .git import in_git_repository
from .make_context_matrix import make_context_matrix
from .plot_and_save import plot_and_save
from .compute_information_distance import compute_information_distance
from .make_random_color import make_random_color
from .simulate_sequences_using_dwgsim import simulate_sequences_using_dwgsim
from .compute_posterior_probability import compute_posterior_probability
from .normalize_nd_array import normalize_nd_array
from .get_sequence_from_fasta_gz import get_sequence_from_fasta_gz
from .apply_function_on_2_1d_arrays import apply_function_on_2_1d_arrays
from .log import get_now
from .log import initialize_logger
from .log import echo_or_print
from .log import log_and_return_response
from .single_sample_gsea import single_sample_gsea
from .machine import get_machine
from .machine import get_shell_environment
from .machine import have_program
from .machine import shutdown_machine
from .machine import reboot_machine
from .process_fastq_gz import check_fastq_gzs_using_fastqc
from .process_fastq_gz import trim_fastq_gzs_using_skewer
from .process_fastq_gz import align_fastq_gzs_using_bwa_mem
from .process_fastq_gz import align_fastq_gzs_using_hisat2
from .process_fastq_gz import count_transcripts_using_kallisto_quant
from .make_coordinates_for_reflection import make_coordinates_for_reflection
from .Genome import is_valid_vcf_gz
from .Genome import Genome
from .apply_function_on_2_2d_arrays_slices import apply_function_on_2_2d_arrays_slices
