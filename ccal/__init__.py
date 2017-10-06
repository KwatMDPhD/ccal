from .classification.classification.classify import classify
from .cluster.cluster.consensus_cluster import consensus_cluster
from .cluster.cluster.count_coclusterings import count_coclusterings
from .cluster.cluster.hierarchical_cluster_distances_and_compute_ccc import \
    hierarchical_cluster_distances_and_compute_ccc
from .cluster.cluster.hierarchical_consensus_cluster import \
    hierarchical_consensus_cluster
from .cluster.cluster.nmf_consensus_cluster import nmf_consensus_cluster
from .dimension_reduction.dimension_reduction.mds import mds
from .gsea.gsea.compute_enrichment_score import compute_enrichment_score
from .gsea.gsea.permute_and_compute_enrichment_score import \
    permute_and_compute_enrichment_score
from .gsea.gsea.run_single_sample_gsea import run_single_sample_gsea
from .information.information.compute_brier_entropy import \
    compute_brier_entropy
from .information.information.compute_entropy import compute_entropy
from .information.information.compute_information_coefficient import \
    compute_information_coefficient
from .kernel_density.kernel_density.compute_bandwidths import \
    compute_bandwidths
from .kernel_density.kernel_density.kde import kde
from .linear_algebra.linear_algebra.solve_ax_equal_b import solve_ax_equal_b
from .linear_model.linear_model.correlate import correlate
from .match.match.make_match_panel import make_match_panel
from .match.match.make_summary_match_panel import make_summary_match_panel
from .matrix_decomposition.matrix_decomposition.nmf import nmf
from .nd_array.nd_array.apply_function_on_2_1d_arrays_and_compute_empirical_p_value import \
    apply_function_on_2_1d_arrays_and_compute_empirical_p_value
from .nd_array.nd_array.apply_function_on_2_2d_arrays_slices import \
    apply_function_on_2_2d_arrays_slices
from .nd_array.nd_array.cluster_2d_array_rows_and_columns import \
    cluster_2d_array_rows_and_columns
from .nd_array.nd_array.cluster_2d_array_slices_by_group import \
    cluster_2d_array_slices_by_group
from .nd_array.nd_array.compute_empirical_p_value import \
    compute_empirical_p_value
from .nd_array.nd_array.compute_empirical_p_values_and_fdrs import \
    compute_empirical_p_values_and_fdrs
from .nd_array.nd_array.compute_log2_fold_ratios import compute_log2_ratios
from .nd_array.nd_array.compute_margin_of_error import compute_margin_of_error
from .nd_array.nd_array.define_exponential_function import \
    define_exponential_function
from .nd_array.nd_array.drop_nan_and_apply_function_on_2_1d_arrays import \
    drop_nan_and_apply_function_on_2_1d_arrays
from .nd_array.nd_array.fit_function_on_each_2d_array_slice import \
    fit_function_on_each_2d_array_slice
from .nd_array.nd_array.get_1d_array_unique_objects_in_order import \
    get_1d_array_unique_objects_in_order
from .nd_array.nd_array.make_index_and_fraction_grid_coordinates_pair import \
    make_index_and_fraction_grid_coordinates_pair
from .nd_array.nd_array.make_nd_grid_coordinates import \
    make_nd_grid_coordinates
from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array
from .nd_array.nd_array.shuffle_each_2d_array_slice import \
    shuffle_each_2d_array_slice
from .onco_gps.onco_gps.define_components import define_components
from .onco_gps.onco_gps.define_states import define_states
from .onco_gps.onco_gps.GPSMap import GPSMap
from .plot.plot.plot_heatmap import plot_heatmap
from .probability.probability.compute_joint_probability import \
    compute_joint_probability
from .probability.probability.compute_posterior_probability import \
    compute_posterior_probability
from .probability.probability.get_target_grid_indices import \
    get_target_grid_indices
from .probability.probability.infer import infer
from .probability.probability.infer_assuming_independence import \
    infer_assuming_independence
from .probability.probability.plot_bayesian_nomogram import \
    plot_bayesian_nomogram
from .regression.regression.regress import regress
from .skew.skew.skew import (define_x_coordinates_for_reflection,
                             fit_essentiality, make_essentiality_matrix,
                             plot_essentiality)
from .support.support.compression import (extract_tar, gzip_compress,
                                          gzip_decompress,
                                          gzip_decompress_and_bgzip_compress,
                                          unzip)
from .support.support.df import drop_df_slices, simulate_df, split_df
from .support.support.dict import merge_dicts_with_function, write_dict
from .support.support.environment import (get_reference, install_libraries,
                                          source_environment)
from .support.support.iterable import (flatten_nested,
                                       get_unique_objects_in_order, group,
                                       group_and_apply_function_on_each_group,
                                       integize, replace_bad_objects)
from .support.support.multiprocess import multiprocess
from .support.support.network import download, get_open_port
from .support.support.path import (copy_path, establish_path,
                                   get_home_directory_path)
from .support.support.series import (cast_series_to_builtins,
                                     get_top_and_bottom_series_indices,
                                     make_membership_df_from_categorical_series,
                                     simulate_series)
from .support.support.str_ import (cast_str_to_builtins,
                                   split_str_ignoring_inside_quotes, title_str,
                                   untitle_str)
from .support.support.subprocess_ import run_command
from .support.support.system import reboot, shutdown
from .support.support.volume import (get_volume_name, make_volume_dict, mount,
                                     unmount)
from .support.suppport.log import get_now
from .tcga.tcga.access_tcga import (make_case_annotations,
                                    read_CopyNumber_Gistic2,
                                    read_CopyNumberLowPass_Gistic2,
                                    read_MutSigNozzleReport2CV)
