"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from colorsys import rgb_to_hsv, hsv_to_rgb
from os.path import join

import matplotlib.pyplot as plt
from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from numpy import asarray, zeros, zeros_like, ones, empty, linspace, nansum, ma, sqrt
from pandas import DataFrame, Series, read_csv, isnull
from scipy.spatial import Delaunay, ConvexHull

from .association import make_association_panel
from .. import RANDOM_SEED
from ..machine_learning.cluster import hierarchical_consensus_cluster, nmf_consensus_cluster
from ..machine_learning.fit import fit_matrix
from ..machine_learning.multidimentional_scale import mds
from ..machine_learning.normalize import normalize_dataframe_or_series
from ..machine_learning.score import compute_association_and_pvalue
from ..machine_learning.solve import solve_matrix_linear_equation
from ..machine_learning.classify import classify
from ..mathematics.equation import define_exponential_function
from ..mathematics.information import EPS, kde2d, bcv, information_coefficient
from ..support.d2 import drop_uniform_slice_from_dataframe, drop_na_2d
from ..support.file import read_gct, establish_filepath, load_gct, write_gct, write_dict
from ..support.log import print_log
from ..support.plot import FIGURE_SIZE, CMAP_CONTINUOUS, CMAP_CATEGORICAL_2, CMAP_BINARY, plot_heatmap, plot_points, \
    plot_nmf, assign_colors_to_states, save_plot


# ======================================================================================================================
# Define components
# ======================================================================================================================
def define_components(matrix, ks, how_to_drop_na='all', n_jobs=1, n_clusterings=40, random_seed=RANDOM_SEED,
                      directory_path=None):
    """
    NMF-consensus cluster samples (matrix columns) and compute cophenetic-correlation coefficients, and save 1 NMF
    results for each k.
    :param matrix: DataFrame or str; (n_rows, n_columns) or filepath to a .gct file
    :param how_to_drop_na: str; 'all' or 'any'
    :param ks: iterable or int; iterable of int k used for NMF
    :param n_jobs: int;
    :param n_clusterings: int; number of NMF for consensus clustering
    :param random_seed: int;
    :param directory_path: str; directory path where
            cophenetic_correlation_coefficients{.pdf, .gct}
            matrices/nmf_k{k}_{w, h}.gct
            figures/nmf_k{k}_{w, h}.pdf
        will be saved.
    :return: dict and dict; {k: {w: W matrix (n_rows, k), h: H matrix (k, n_columns), e: Reconstruction Error}} and
                            {k: Cophenetic Correlation Coefficient}
    """

    if isinstance(matrix, str):  # Read form a .gct file
        matrix = read_gct(matrix)

    # Drop na rows & columns
    matrix = drop_na_2d(matrix, how=how_to_drop_na)

    # Rank normalize the input matrix by column
    # TODO: try changing n_ranks (choose automatically)
    matrix = normalize_dataframe_or_series(matrix, 'rank', n_ranks=10000, axis=0)
    plot_heatmap(matrix, title='(Rank-Normalized) Matrix to be Decomposed', xlabel='Sample', ylabel='Feature',
                 xticklabels=False, yticklabels=False, cluster=True)

    # NMF-consensus cluster (while saving 1 NMF result per k)
    nmf_results, cophenetic_correlation_coefficient = nmf_consensus_cluster(matrix, ks,
                                                                            n_jobs=n_jobs, n_clusterings=n_clusterings,
                                                                            random_seed=random_seed)

    # Make NMF directory, where
    #     cophenetic_correlation_coefficients{.pdf, .gct}
    #     matrices/nmf_k{k}_{w, h}.gct
    #     figures/nmf_k{k}_{w, h}.pdf
    # will be saved
    if directory_path:
        # Make NMF parent directory
        directory_path = join(directory_path, 'nmf', '')
        establish_filepath(directory_path)

        print_log('Saving NMF decompositions and cophenetic correlation coefficients ...')
        # Save NMF decompositions
        _save_nmf(nmf_results, join(directory_path, 'matrices', ''))
        # Save cophenetic correlation coefficients
        write_dict(cophenetic_correlation_coefficient,
                   join(directory_path, 'cophenetic_correlation_coefficients.txt'),
                   key_name='k', value_name='cophenetic_correlation_coefficient')

        # Saving filepath for cophenetic correlation coefficients figure
        filepath_ccc_pdf = join(directory_path, 'cophenetic_correlation_coefficients.pdf')

    else:
        # Not saving cophenetic correlation coefficients figure
        filepath_ccc_pdf = None

    print_log('Plotting NMF decompositions and cophenetic correlation coefficients ...')
    # Plot cophenetic correlation coefficients
    plot_points(sorted(cophenetic_correlation_coefficient.keys()),
                [cophenetic_correlation_coefficient[k] for k in sorted(cophenetic_correlation_coefficient.keys())],
                title='NMF Cophenetic Correlation Coefficient vs. k',
                xlabel='k', ylabel='NMF Cophenetic Correlation Coefficient',
                filepath=filepath_ccc_pdf)

    if isinstance(ks, int):
        ks = [ks]

    # Plot NMF decompositions
    for k in ks:
        print_log('\tPlotting k={} ...'.format(k))
        if directory_path:
            filepath_nmf = join(directory_path, 'figures', 'nmf_k{}.pdf'.format(k))
        else:
            filepath_nmf = None

        plot_nmf(nmf_results, k, filepath=filepath_nmf)

    return nmf_results, cophenetic_correlation_coefficient


def _save_nmf(nmf_results, filepath_prefix):
    """
    Save NMF decompositions.
    :param nmf_results: dict; {k: {w: W matrix, h: H matrix, e: Reconstruction Error}} and
                              {k: Cophenetic Correlation Coefficient}
    :param filepath_prefix: str; filepath_prefix_nmf_k{k}_{w, h}.gct and will be saved
    :return: None
    """

    for k, v in nmf_results.items():
        write_gct(v['w'], filepath_prefix + 'nmf_k{}_w.gct'.format(k))
        write_gct(v['h'], filepath_prefix + 'nmf_k{}_h.gct'.format(k))


def get_w_or_h_matrix(nmf_results, k, w_or_h):
    """
    Get W or H matrix from nmf_results.
    :param nmf_results: dict;
    :param k: int;
    :param w_or_h: str; 'w', 'W', 'H', or 'h'
    :return: DataFrame; W or H matrix for this k
    """

    w_or_h = w_or_h.strip()
    if w_or_h not in ('w', 'W', 'H', 'h'):
        raise TypeError('w_or_h must be one of {w, W, H, h}.')

    return nmf_results[k][w_or_h.lower()]


def solve_for_components(w_matrix, a_matrix, method='nnls', average_duplicated_rows_of_a_matrix=True,
                         filepath_prefix=None):
    """
    Get H matrix of a_matrix in the space of w_matrix by solving W * H = A for H.
    :param w_matrix: str or DataFrame; (n_rows, k)
    :param a_matrix: str or DataFrame; (n_rows, n_columns)
    :param method: str; {nnls, pinv}
    :param average_duplicated_rows_of_a_matrix: bool; Average duplicate rows of the A matrix or not
    :param filepath_prefix: str; filepath_prefix_solved_nmf_h_k{}.{gct, pdf} will be saved
    :return: DataFrame; (k, n_columns)
    """

    # Load A and W matrices
    w_matrix = load_gct(w_matrix)
    a_matrix = load_gct(a_matrix)
    if average_duplicated_rows_of_a_matrix:  # Average duplicate rows of the A matrix
        a_matrix = a_matrix.groupby(level=0).mean()

    # Keep only indices shared by both
    common_indices = set(a_matrix.index) & set(w_matrix.index)
    w_matrix = w_matrix.ix[common_indices, :]
    a_matrix = a_matrix.ix[common_indices, :]

    # Rank normalize the A matrix by column
    # TODO: try changing n_ranks (choose automatically)
    a_matrix = normalize_dataframe_or_series(a_matrix, 'rank', n_ranks=10000, axis=0)

    # Normalize the W matrix by column
    # TODO: improve the normalization (why this normalization?)
    w_matrix = w_matrix.apply(lambda c: c / c.sum() * 1000)

    # Solve W * H = A
    print_log('Solving for components: W({}x{}) * H = A({}x{}) ...'.format(*w_matrix.shape, *a_matrix.shape))
    h_matrix = solve_matrix_linear_equation(w_matrix, a_matrix, method=method)

    if filepath_prefix:  # Save H matrix
        write_gct(h_matrix, filepath_prefix + '_solved_nmf_h_k{}.gct'.format(h_matrix.shape[0]))
        plot_filepath = filepath_prefix + '_solved_nmf_h_k{}.pdf'.format(h_matrix.shape[0])
    else:
        plot_filepath = None

    plot_nmf(w_matrix=w_matrix, h_matrix=h_matrix, filepath=plot_filepath)

    return h_matrix


def select_features_and_nmf(testing, training,
                            target, target_type='categorical', feature_scores=None,
                            testing_name='Testing', training_name='Training', row_name='Feature', column_name='Sample',
                            n_jobs=1, n_samplings=30, n_permutations=30,
                            n_top_features=0.05, n_bottom_features=0.05,
                            ks=(), n_clusterings=100, random_seed=RANDOM_SEED,
                            directory_path=None, feature_scores_filename_prefix='feature_scores'):
    """
    Select features from training based on their association with target. Keep only those features from testing, and
    perform NMF on testing, which has only the selected features.
    :param testing: DataFrame;
    :param training: DataFrame;
    :param target: Series;
    :param target_type: str;
    :param feature_scores: DataFrame or str;
    :param testing_name:
    :param training_name:
    :param row_name:
    :param column_name:
    :param n_jobs:
    :param n_samplings:
    :param n_permutations:
    :param n_top_features:
    :param n_bottom_features:
    :param ks:
    :param n_clusterings:
    :param random_seed: int;
    :param directory_path:
    :param feature_scores_filename_prefix:
    :return:
    """

    # Plot training
    plot_heatmap(training, normalization_method='-0-', normalization_axis=1,
                 column_annotation=target,
                 title=training_name, xlabel=column_name, ylabel=row_name, yticklabels=False)

    if not feature_scores:  # Compute feature scores
        feature_scores = make_association_panel(target, training,
                                                target_type=target_type,
                                                n_jobs=n_jobs, n_samplings=n_samplings, n_permutations=n_permutations,
                                                random_seed=random_seed,
                                                filepath_prefix=join(directory_path, feature_scores_filename_prefix))
    else:  # Read feature scores from a file
        if not isinstance(feature_scores, DataFrame):
            feature_scores = read_csv(feature_scores, sep='\t', index_col=0)

    # Select features
    if n_top_features < 1:  # Fraction
        n_top_features = n_top_features * feature_scores.shape[0]
    if n_bottom_features < 1:  # Fraction
        n_bottom_features = n_bottom_features * feature_scores.shape[0]
    features = feature_scores.index[:n_top_features] | feature_scores.index[-n_bottom_features:]

    # Plot training with selected features
    plot_heatmap(training.ix[features, :], normalization_method='-0-', normalization_axis=1,
                 column_annotation=target,
                 title='{} with Selected {}s'.format(training_name, row_name),
                 xlabel=column_name, ylabel=row_name, yticklabels=False)

    # Plot testing with selected features
    testing = testing.ix[testing.index & features, :]
    print_log('Selected {} testing features.'.format(testing.shape[0]))
    plot_heatmap(testing, normalization_method='-0-', normalization_axis=1,
                 title='{} with Selected {}'.format(testing_name, row_name),
                 xlabel=column_name, ylabel=row_name, xticklabels=False, yticklabels=False)

    # NMF
    nmf_results, cophenetic_correlation_coefficients = define_components(testing, ks,
                                                                         n_jobs=n_jobs, n_clusterings=n_clusterings,
                                                                         random_seed=random_seed,
                                                                         directory_path=directory_path)

    return nmf_results, cophenetic_correlation_coefficients


# ======================================================================================================================
# Define states
# ======================================================================================================================
def define_states(matrix, ks, distance_matrix=None, max_std=3, n_clusterings=40, random_seed=RANDOM_SEED,
                  directory_path=None):
    """
    Hierarchical-consensus cluster samples (matrix columns) and compute cophenetic correlation coefficients.
    :param matrix: DataFrame or str; (n_rows, n_columns); filepath to a .gct
    :param ks: iterable; iterable of int k used for hierarchical clustering
    :param distance_matrix: str or DataFrame; (n_columns, n_columns); distance matrix to hierarchical cluster
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of hierarchical clusterings for consensus clustering
    :param random_seed: int;
    :param directory_path: str; directory path where
            clusterings/distance_matrix.{txt, pdf}
            clusterings/clusterings.{gct, pdf}
            clusterings/cophenetic_correlation_coefficients.{txt, pdf}
        will be saved
    :return: DataFrame, DataFrame, and Series;
        distance_matrix (n_samples, n_samples),
        clusterings (n_ks, n_columns), and
        cophenetic correlation coefficients (n_ks)
    """

    if isinstance(matrix, str):  # Read form a .gct file
        matrix = read_gct(matrix)

    # '-0-' normalize by rows and clip values max_std standard deviation away; then '0-1' normalize by rows
    matrix = normalize_dataframe_or_series(normalize_dataframe_or_series(matrix, '-0-', axis=1).clip(-max_std, max_std),
                                           method='0-1', axis=1)

    # Hierarchical-consensus cluster
    distance_matrix, clusterings, cophenetic_correlation_coefficients = \
        hierarchical_consensus_cluster(matrix, ks, distance_matrix=distance_matrix, n_clusterings=n_clusterings,
                                       random_seed=random_seed)

    if directory_path:  # Save and plot distance matrix, clusterings, and cophenetic correlation coefficients
        directory_path = join(directory_path, 'clusterings', '')
        establish_filepath(directory_path)

        # Save results
        distance_matrix.to_csv(join(directory_path, 'distance_matrix.txt'), sep='\t')
        write_gct(clusterings, join(directory_path, 'clusterings.gct'))
        write_dict(cophenetic_correlation_coefficients,
                   join(directory_path, 'cophenetic_correlation_coefficients.txt'),
                   key_name='k', value_name='cophenetic_correlation_coefficient')

        # Set up filepath to save plots
        filepath_distance_matrix_plot = join(directory_path, 'distance_matrix.pdf')
        filepath_clusterings_plot = join(directory_path, 'clusterings.pdf')
        filepath_cophenetic_correlation_coefficients_plot = join(directory_path,
                                                                 'cophenetic_correlation_coefficients.pdf')

    else:  # Don't save results and plots
        filepath_distance_matrix_plot = None
        filepath_clusterings_plot = None
        filepath_cophenetic_correlation_coefficients_plot = None

    # Plot distance matrix
    plot_heatmap(distance_matrix, cluster=True,
                 title='Distance Matrix', xlabel='Sample', ylabel='Sample',
                 xticklabels=False, yticklabels=False, filepath=filepath_distance_matrix_plot)

    # Plot clusterings
    plot_heatmap(clusterings, sort_axis=1, data_type='categorical', normalization_method=None,
                 title='Clustering per k', xticklabels=False, filepath=filepath_clusterings_plot)

    # Plot cophenetic correlation coefficients
    plot_points(sorted(cophenetic_correlation_coefficients.keys()),
                [cophenetic_correlation_coefficients[k] for k in sorted(cophenetic_correlation_coefficients.keys())],
                title='Consensus-Clustering-Cophenetic-Correlation Coefficients vs. k',
                xlabel='k', ylabel='Cophenetic Score', filepath=filepath_cophenetic_correlation_coefficients_plot)

    return distance_matrix, clusterings, cophenetic_correlation_coefficients


def get_state_labels(clusterings, k):
    """
    Get state labels from clusterings.
    :param clusterings: DataFrame;
    :param k: int;
    :return: Series;
    """

    return clusterings.ix[k, :].tolist()


# TODO: use explode function, and remove this function
def define_binary_state_labels(clusterings, k, state_relabeling=None):
    """
    Get state labels from clusterings and create binary variable for each state
    :param clusterings: DataFrame;
    :param k: int;
    :param state_relabeling: int; [1, n_states]; new labels for the states or None
    :return: DataFrame;
    """

    k_labels = clusterings.ix[k, :]

    if state_relabeling:
        for i in range(len(k_labels)):
            k_labels.ix[i] = state_relabeling[k_labels.ix[i] - 1]

    u_labels = k_labels.unique().tolist()
    max_index = len(u_labels) + 1
    u_labels.extend([max_index])

    binary_labels = DataFrame(index=u_labels, columns=clusterings.columns)
    for state in k_labels.unique():
        binary_labels.ix[state, :] = (k_labels == state).astype('int')

    binary_labels.ix[max_index, :] = k_labels

    return binary_labels


# ======================================================================================================================
# Make Onco-GPS map
# ======================================================================================================================
def make_oncogps(training_h,
                 training_states,
                 std_max=3,

                 testing_h=None,
                 testing_states=(),
                 testing_h_normalization='using_training_h',

                 components=None,
                 equilateral=False,
                 informational_mds=True,
                 mds_seed=RANDOM_SEED,

                 n_pulls=None,
                 power=None,
                 fit_min=0,
                 fit_max=2,
                 power_min=1,
                 power_max=5,

                 n_grids=256,
                 kde_bandwidths_factor=2,

                 samples_to_plot=None,
                 component_ratio=0,

                 annotation=(),
                 annotation_name='',
                 annotation_type='continuous',
                 annotation_ascending=True,
                 highlight_high_magnitude=True,

                 title='Onco-GPS Map',
                 title_fontsize=26,
                 title_fontcolor='#3326C0',

                 subtitle_fontsize=20,
                 subtitle_fontcolor='#FF0039',

                 component_marker='o',
                 component_markersize=26,
                 component_markerfacecolor='#000726',
                 component_markeredgewidth=2.6,
                 component_markeredgecolor='#FFFFFF',
                 component_names=(),
                 component_fontsize=26,

                 delaunay_linewidth=0.7,
                 delaunay_linecolor='#000000',

                 colors=(),
                 bad_color='wheat',
                 max_background_saturation=1,

                 n_contours=26,
                 contour_linewidth=0.60,
                 contour_linecolor='#262626',
                 contour_alpha=0.80,

                 sample_markersize=23,
                 sample_markeredgewidth=0.92,
                 sample_markeredgecolor='#000000',
                 sample_name_size=16,
                 sample_name_color=None,

                 legend_markersize=16,
                 legend_fontsize=16,

                 filepath=None):
    """
    :param training_h: DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param training_states: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values

    :param testing_h: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param testing_states: iterable of int; (n_samples); sample states
    :param testing_h_normalization: str or None; {'using_training_h', 'using_testing_h', None}

    :param components: DataFrame; (n_components, 2 [x, y]); component coordinates
    :param equilateral: bool;
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling

    :param n_pulls: int; [1, n_components]; number of components influencing a sample's coordinate
    :param power: str or number; power to raise components' influence on each sample
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;

    :param n_grids: int; number of grids; larger the n_grids, higher the resolution
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths

    :param samples_to_plot: indexer; (n_training_samples), (n_testing_samples), or (n_sample_indices)
    :param component_ratio: number; number if int; percentile if float & < 1

    :param annotation: pandas Series; (n_samples); sample annotation; will color samples based on annotation
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param annotation_ascending: bool;
    :param highlight_high_magnitude: bool;

    :param title: str;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib color;

    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib color;

    :param component_marker: str;
    :param component_markersize: number;
    :param component_markerfacecolor: matplotlib color;
    :param component_markeredgewidth: number;
    :param component_markeredgecolor: matplotlib color;
    :param component_names: iterable; (n_components)
    :param component_fontsize: number;

    :param delaunay_linewidth: number;
    :param delaunay_linecolor: matplotlib color;

    :param colors: matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap, or iterable;
    :param bad_color: matplotlib color;
    :param max_background_saturation: float; [0, 1]

    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]

    :param sample_markersize: number;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param sample_name_size: number;
    :param sample_name_color: matplotlib color; not plotting sample if None

    :param legend_markersize: number;
    :param legend_fontsize: number;

    :param filepath: str;

    :return: None
    """

    # Make sure the index is str (better for .ix)
    # TODO: enforece
    training_h.index = training_h.index.astype(str)

    # For normalization
    dummy = training_h.copy()

    # Preprocess training-H matrix and training states
    training_h, training_states = _process_h_and_states(training_h, training_states, std_max)
    print_log('Training Onco-GPS with {} components, {} samples, & {} states ...'.format(*training_h.shape,
                                                                                         len(set(training_states))))
    print_log('\tComponents: {}.'.format(set(training_h.index)))
    print_log('\tTraining states: {}.'.format(set(training_states)))

    if isinstance(testing_h, DataFrame):

        # Make sure the index is str (better for .ix)
        # TODO: enforce
        testing_h.index = testing_h.index.astype(str)

        if testing_h_normalization == 'using_training_h':
            normalize_testing_h = True
            normalizing_size = dummy.shape[1]
            normalizing_mean = dummy.mean(axis=1)
            normalizing_std = dummy.std(axis=1)
            dummy = normalize_dataframe_or_series(dummy, '-0-', axis=1).clip(-std_max, std_max)
            normalizing_min = dummy.min(axis=1)
            normalizing_max = dummy.max(axis=1)

        elif testing_h_normalization == 'using_testing_h':
            normalize_testing_h = True
            normalizing_size = None
            normalizing_mean = None
            normalizing_std = None
            normalizing_min = None
            normalizing_max = None

        elif testing_h_normalization is None:
            normalize_testing_h = False
            normalizing_size = None
            normalizing_mean = None
            normalizing_std = None
            normalizing_min = None
            normalizing_max = None

        else:
            raise ValueError('testing_h_normalization must be \'using_training_h\', \'using_testing_h\', or None.')
        testing_h, testing_states, = _process_h_and_states(testing_h, testing_states, std_max,
                                                           normalize=normalize_testing_h,
                                                           normalizing_size=normalizing_size,
                                                           normalizing_mean=normalizing_mean,
                                                           normalizing_std=normalizing_std,
                                                           normalizing_min=normalizing_min,
                                                           normalizing_max=normalizing_max)

        if not any(testing_states):  # Predict state labels for the testing samples
            testing_states = classify(training_h.T, training_states, testing_h.T)

        print_log('Testing Onco-GPS with {} components, {} samples, & {} states ...'.format(*testing_h.shape,
                                                                                            len(set(testing_states))))
        print_log('\tComponents: {}.'.format(set(testing_h.index)))
        print_log('\tTraining states: {}.'.format(set(testing_states)))
        print_log('\tNormalization: {}'.format(testing_h_normalization))

    # Compute component coordinates
    if equilateral and training_h.shape[0] == 3:
        print_log('Using equilateral component coordinates ...'.format(components))
        components = DataFrame(index=['Vertex 1', 'Vertex 2', 'Vertex 3'], columns=['x', 'y'])
        components.iloc[0, :] = [0.5, sqrt(3) / 2]
        components.iloc[1, :] = [1, 0]
        components.iloc[2, :] = [0, 0]

    if isinstance(components, DataFrame):
        print_log('Using predefined component coordinates ...'.format(components))
        components.index = training_h.index
    else:
        if informational_mds:
            print_log('Computing component coordinates using informational distance ...')
            distance_function = information_coefficient
        else:
            print_log('Computing component coordinates using Euclidean distance ...')
            distance_function = None
        components = mds(training_h, distance_function=distance_function, random_seed=mds_seed, standardize=True)

    if not n_pulls:  # n_pulls = number of all components
        n_pulls = training_h.shape[0]

    if not power:
        print_log('Computing power ...')
        if training_h.shape[0] < 4:
            print_log('\tCould\'t model with Ae^(kx) + C; too few data points.')
            power = 1
        else:
            try:
                power = _compute_component_power(training_h, fit_min, fit_max, power_min, power_max)
            except RuntimeError as e:
                print_log('\tCould\'t model with Ae^(kx) + C; {}.'.format(e))
                power = 1

    # Process samples
    training_samples = _process_samples(training_h, training_states, components, n_pulls, power, component_ratio)

    print_log('Computing grid probabilities and states ...')
    grid_probabilities, grid_states = _compute_grid_probabilities_and_states(training_samples, n_grids,
                                                                             kde_bandwidths_factor)

    if isinstance(testing_h, DataFrame):
        testing_samples = _process_samples(testing_h, testing_states, components, n_pulls, power, component_ratio)

        samples = testing_samples
        if any(annotation):  # Make sure annotation is Series and keep selected samples
            annotation = _process_annotation(annotation, testing_h.columns)

    else:
        samples = training_samples
        if any(annotation):  # Make sure annotation is Series and keep selected samples
            annotation = _process_annotation(annotation, training_h.columns)

    if samples_to_plot:  # Limit samples to be plotted
        samples = samples.ix[samples_to_plot, :]

    print_log('Plotting ...')
    _plot_onco_gps(components=components,
                   samples=samples,
                   grid_probabilities=grid_probabilities,
                   grid_states=grid_states,
                   n_training_states=len(set(training_states)),

                   annotation=annotation,
                   annotation_name=annotation_name,
                   annotation_type=annotation_type,
                   annotation_ascending=annotation_ascending,
                   highlight_high_magnitude=highlight_high_magnitude,

                   std_max=std_max,

                   title=title,
                   title_fontsize=title_fontsize,
                   title_fontcolor=title_fontcolor,

                   subtitle_fontsize=subtitle_fontsize,
                   subtitle_fontcolor=subtitle_fontcolor,

                   component_marker=component_marker,
                   component_markersize=component_markersize,
                   component_markerfacecolor=component_markerfacecolor,
                   component_markeredgewidth=component_markeredgewidth,
                   component_markeredgecolor=component_markeredgecolor,
                   component_names=component_names,
                   component_fontsize=component_fontsize,

                   delaunay_linewidth=delaunay_linewidth,
                   delaunay_linecolor=delaunay_linecolor,

                   colors=colors,
                   bad_color=bad_color,
                   max_background_saturation=max_background_saturation,

                   n_contours=n_contours,
                   contour_linewidth=contour_linewidth,
                   contour_linecolor=contour_linecolor,
                   contour_alpha=contour_alpha,

                   sample_markersize=sample_markersize,
                   sample_markeredgewidth=sample_markeredgewidth,
                   sample_markeredgecolor=sample_markeredgecolor,
                   sample_name_size=sample_name_size,
                   sample_name_color=sample_name_color,

                   legend_markersize=legend_markersize,
                   legend_fontsize=legend_fontsize,

                   filepath=filepath)


# ======================================================================================================================
# Process H matrix and states
# ======================================================================================================================
def _process_h_and_states(h, states, std_max, normalize=True,
                          normalizing_size=None,
                          normalizing_mean=None, normalizing_std=None,
                          normalizing_min=None, normalizing_max=None):
    """
    Process H matrix and states.
    :param h: DataFrame; (n_components, n_samples); H matrix
    :param states: iterable of ints;
    :param std_max: number;
    :param normalizing_size:
    :param normalizing_mean:
    :param normalizing_std:
    :param normalizing_min:
    :param normalizing_max:
    :return: DataFrame and Series; processed H matrix and states
    """

    # TODO: improve logic
    # Convert sample-state labels, which match sample, into Series
    if not any(states):
        states = zeros(h.shape[1])
    states = Series(states, index=h.columns)

    # Normalize H matrix and drop all-0 samples
    h = _process_h(h, std_max, normalize=normalize,
                   normalizing_size=normalizing_size,
                   normalizing_mean=normalizing_mean, normalizing_std=normalizing_std,
                   normalizing_min=normalizing_min, normalizing_max=normalizing_max)

    # Drop all-0 samples from states too
    states = states.ix[h.columns]

    return h, states


def _process_h(h, std_max, normalize=True,
               normalizing_size=None,
               normalizing_mean=None, normalizing_std=None,
               normalizing_min=None, normalizing_max=None):
    """
    Normalize H matrix and drop all-0 samples.
    :param h: DataFrame; (n_components, n_samples); H matrix
    :param std_max: number;
    :param normalizing_size:
    :param normalizing_mean:
    :param normalizing_std:
    :param normalizing_min:
    :param normalizing_max:
    :return: DataFrame; (n_components, n_samples); Normalized H matrix
    """

    # TODO: refactor 2 dropping all-0 samples

    # Drop all-0 samples
    h = drop_uniform_slice_from_dataframe(h, 0)

    if normalize:
        # Clip by standard deviation and 0-1 normalize
        h = _normalize_h(h, std_max,
                         normalizing_size=normalizing_size,
                         normalizing_mean=normalizing_mean, normalizing_std=normalizing_std,
                         normalizing_min=normalizing_min, normalizing_max=normalizing_max)

        # Drop all-0 samples
        h = drop_uniform_slice_from_dataframe(h, 0)

    return h


def _normalize_h(h, std_max,
                 normalizing_size=None,
                 normalizing_mean=None, normalizing_std=None,
                 normalizing_min=None, normalizing_max=None):
    """
    Clip by standard deviation and 0-1 normalize the rows of H matrix.
    :param h: DataFrame; (n_components, n_samples); H matrix
    :param std_max: number;
    :param normalizing_size:
    :param normalizing_mean:
    :param normalizing_std:
    :param normalizing_min:
    :param normalizing_max:
    :return: DataFrame; (n_components, n_samples); Normalized H matrix
    """

    h = normalize_dataframe_or_series(h, '-0-', axis=1,
                                      normalizing_size=normalizing_size,
                                      normalizing_mean=normalizing_mean, normalizing_std=normalizing_std)
    h = h.clip(-std_max, std_max)
    h = normalize_dataframe_or_series(h, '0-1', axis=1,
                                      normalizing_size=normalizing_size,
                                      normalizing_min=normalizing_min, normalizing_max=normalizing_max)

    return h


def _compute_component_power(h, fit_min, fit_max, power_min, power_max):
    """
    Compute component power by fitting component magnitudes of samples to the exponential function.
    :param h: DataFrame;
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;
    :return: float; power
    """

    fit_parameters = fit_matrix(h, define_exponential_function, sort_matrix=True)
    k = fit_parameters[1]

    # Linear transform
    k_zero_to_one = (k - fit_min) / (fit_max - fit_min)
    k_rescaled = k_zero_to_one * (power_max - power_min) + power_min

    return k_rescaled


# ======================================================================================================================
# Process samples
# ======================================================================================================================
def _process_samples(h, states, components, n_pulls, power, component_ratio):
    """

    :param h:
    :param states:
    :param components:
    :param n_pulls:
    :param power:
    :param component_ratio:
    :return: DataFrame; (n_samples, 3 (or 4) [x, y, state, (component_ratio)])
    """

    samples = DataFrame(index=h.columns, columns=['x', 'y', 'state'])
    samples.ix[:, 'state'] = states

    print_log('Computing sample coordinates using {} components and {:.3f} power ...'.format(n_pulls, power))
    samples.ix[:, ['x', 'y']] = _compute_sample_coordinates(components, h, n_pulls, power)

    if component_ratio and 0 < component_ratio:
        print_log('Computing component ratios ...')
        component_ratios = _compute_component_ratios(h, component_ratio)
        if 1 < len(set(component_ratios)):
            samples.ix[:, 'component_ratio'] = component_ratios

    return samples


def _compute_sample_coordinates(component_x_coordinates, component_x_samples, n_influencing_components, power):
    """
    Compute sample coordinates based on component coordinates (components pull samples).
    :param component_x_coordinates: DataFrame; (n_points, 2 [x, y])
    :param component_x_samples: DataFrame; (n_points, n_samples)
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param power: number; power to raise components' influence on each sample
    :return: DataFrame; (n_samples, 2 [x, y]); sample_coordinates
    """

    component_x_coordinates = asarray(component_x_coordinates)

    sample_coordinates = empty((component_x_samples.shape[1], 2))

    for i, (_, c) in enumerate(component_x_samples.iteritems()):
        c = asarray(c)

        # Silence components that are not pulling
        threshold = sorted(c)[-n_influencing_components]
        c[c < threshold] = 0

        x = nansum(c ** power * component_x_coordinates[:, 0]) / nansum(c ** power)
        y = nansum(c ** power * component_x_coordinates[:, 1]) / nansum(c ** power)

        sample_coordinates[i] = x, y
    return sample_coordinates


def _compute_component_ratios(h, n):
    """
    Compute the ratio between the sum of the top-n component values and the sum of the rest of the component values.
    :param h: DataFrame;
    :param n: number;
    :return: array; ratios
    """

    ratios = zeros(h.shape[1])

    if n and n < 1:  # If n is a fraction, compute its respective number
        n = int(h.shape[0] * n)

    # Compute pull ratio for each sample (column)
    for i, (c_idx, c) in enumerate(h.iteritems()):
        c_sorted = c.sort_values(ascending=False)

        ratios[i] = c_sorted[:n].sum() / max(c_sorted[n:].sum(), EPS) * c.sum()

    return ratios


# ======================================================================================================================
# Compute grid probabilities and states
# ======================================================================================================================
def _compute_grid_probabilities_and_states(samples, n_grids, kde_bandwidths_factor):
    """

    :param samples:
    :param n_grids:
    :param kde_bandwidths_factor:
    :return:
    """

    grid_probabilities = zeros((n_grids, n_grids), dtype=float)
    grid_states = zeros((n_grids, n_grids), dtype=int)

    # Compute bandwidths created from all states' x & y coordinates and rescale them
    bandwidths = asarray([bcv(asarray(samples.ix[:, 'x'].tolist()))[0],
                          bcv(asarray(samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor

    # KDE for each state using bandwidth created from all states' x & y coordinates
    kdes = {}
    for s in samples.ix[:, 'state'].unique():
        coordinates = samples.ix[samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = kde2d(asarray(coordinates.ix[:, 'x'], dtype=float), asarray(coordinates.ix[:, 'y'], dtype=float),
                    bandwidths, n=asarray([n_grids]), lims=asarray([0, 1, 0, 1]))
        kdes[s] = asarray(kde[2])

    # Assign the best KDE probability and state for each grid
    for i in range(n_grids):
        for j in range(n_grids):

            # Find the maximum probability and its state
            grid_probability = 0
            grid_state = None
            for s, kde in kdes.items():
                a_probability = kde[i, j]
                if a_probability > grid_probability:
                    grid_probability = a_probability
                    grid_state = s

            # Assign the maximum probability and its state
            grid_probabilities[i, j] = grid_probability
            grid_states[i, j] = grid_state

    return grid_probabilities, grid_states


# ======================================================================================================================
# Process annotations
# ======================================================================================================================
def _process_annotation(annotation, ordered_indices):
    """

    :param annotation:
    :param ordered_indices:
    :return:
    """

    if isinstance(annotation, Series):
        annotation = annotation.ix[ordered_indices]
    else:
        annotation = Series(annotation)
        annotation.index = ordered_indices

    return annotation


# ======================================================================================================================
# Plot Onco-GPS map
# ======================================================================================================================
def _plot_onco_gps(components,
                   samples,
                   grid_probabilities,
                   grid_states,
                   n_training_states,

                   annotation,
                   annotation_name,
                   annotation_type,
                   annotation_ascending,
                   highlight_high_magnitude,

                   std_max,

                   title,
                   title_fontsize,
                   title_fontcolor,

                   subtitle_fontsize,
                   subtitle_fontcolor,

                   component_marker,
                   component_markersize,
                   component_markerfacecolor,
                   component_markeredgewidth,
                   component_markeredgecolor,
                   component_names,
                   component_fontsize,

                   delaunay_linewidth,
                   delaunay_linecolor,

                   colors,
                   bad_color,
                   max_background_saturation,

                   n_contours,
                   contour_linewidth,
                   contour_linecolor,
                   contour_alpha,

                   sample_markersize,
                   sample_markeredgewidth,
                   sample_markeredgecolor,
                   sample_name_size,
                   sample_name_color,

                   legend_markersize,
                   legend_fontsize,

                   filepath):
    """
    Plot Onco-GPS map.
    :param components: DataFrame; (n_components, 2 [x, y]);
    :param samples: DataFrame; (n_samples, 3 [x, y, state, component_ratio]);
    :param grid_probabilities: numpy 2D array; (n_grids, n_grids)
    :param grid_states: numpy 2D array; (n_grids, n_grids)
    :param n_training_states: int; number of training-sample states
    :param annotation: Series; (n_samples); sample annotation; will color samples based on annotation
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param annotation_ascending: logical True or False
    :param highlight_high_magnitude: bool;
    :param violin_or_box: str;
    :param std_max: number; threshold to clip standardized values
    :param title: str;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib color;
    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib color;
    :param component_marker;
    :param component_markersize: number;
    :param component_markerfacecolor: matplotlib color;
    :param component_markeredgewidth: number;
    :param component_markeredgecolor: matplotlib color;
    :param component_names: iterable;
    :param component_fontsize: number;
    :param delaunay_linewidth: number;
    :param delaunay_linecolor: matplotlib color;
    :param colors: matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap, or iterable;
    :param bad_color: matplotlib color;
    :param max_background_saturation: float; [0, 1]
    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]
    :param sample_markersize: number;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param sample_name_size: number;
    :param sample_name_color: None or matplotlib color; not plotting sample if None
    :param legend_markersize: number;
    :param legend_fontsize: number;
    :param filepath: str;
    :return: None
    """

    # Set up grids
    x_grids = linspace(0, 1, grid_probabilities.shape[0])
    y_grids = linspace(0, 1, grid_probabilities.shape[1])
    # Set up figure
    plt.figure(figsize=FIGURE_SIZE)
    gridspec = GridSpec(10, 16)
    # Set up title ax
    ax_title = plt.subplot(gridspec[0, :])
    ax_title.axis([0, 1, 0, 1])
    ax_title.axis('off')
    # Set up map ax
    ax_map = plt.subplot(gridspec[0:, :12])
    ax_map.axis([0, 1, 0, 1])
    ax_map.axis('off')
    # Set up legend ax
    ax_legend = plt.subplot(gridspec[1:, 14:])
    ax_legend.axis([0, 1, 0, 1])
    ax_legend.axis('off')

    # Plot title
    ax_map.text(0, 1.16, title,
                fontsize=title_fontsize, weight='bold', color=title_fontcolor, horizontalalignment='left')
    ax_map.text(0, 1.12, '{} samples, {} components, & {} states'.format(samples.shape[0], components.shape[0],
                                                                         n_training_states),
                fontsize=subtitle_fontsize, weight='bold', color=subtitle_fontcolor, horizontalalignment='left')

    # Plot component markers
    ax_map.plot(components.ix[:, 'x'], components.ix[:, 'y'], linestyle='', marker=component_marker,
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor,
                aa=True, clip_on=False, zorder=6)
    # Compute convexhull
    convexhull = ConvexHull(components)
    convexhull_region = Path(convexhull.points[convexhull.vertices])
    # Plot component labels
    if any(component_names):
        components.index = component_names
    for i in components.index:
        # Get x & y coordinates
        x = components.ix[i, 'x']
        y = components.ix[i, 'y']
        # Shift
        if x < 0.5:
            h_shift = -0.0475
        elif 0.5 < x:
            h_shift = 0.0475
        else:
            h_shift = 0
        if y < 0.5:
            v_shift = -0.0475
        elif 0.5 < y:
            v_shift = 0.0475
        else:
            v_shift = 0
        if convexhull_region.contains_point((components.ix[i, 'x'] + h_shift, components.ix[i, 'y'] + v_shift)):  # Flip
            h_shift *= -1
            v_shift *= -1
        x += h_shift
        y += v_shift
        # Plot
        ax_map.text(x, y, i, horizontalalignment='center', verticalalignment='center',
                    fontsize=component_fontsize, weight='bold', color=component_markerfacecolor, zorder=6)

    # Plot Delaunay triangulation
    delaunay = Delaunay(components)
    ax_map.triplot(delaunay.points[:, 0], delaunay.points[:, 1], delaunay.simplices.copy(),
                   linewidth=delaunay_linewidth, color=delaunay_linecolor, aa=True, clip_on=False, zorder=4)

    # Assign colors to states
    state_colors = assign_colors_to_states(n_training_states, colors=colors)

    # Plot background
    grid_probabilities_min = grid_probabilities.min()
    grid_probabilities_max = grid_probabilities.max()
    grid_probabilities_range = grid_probabilities_max - grid_probabilities_min
    image = ones((*grid_probabilities.shape, 3))
    for i in range(grid_probabilities.shape[0]):
        for j in range(grid_probabilities.shape[1]):
            if convexhull_region.contains_point((x_grids[i], y_grids[j])):
                rgba = state_colors[grid_states[i, j]]
                hsv = rgb_to_hsv(*rgba[:3])
                a = (grid_probabilities[i, j] - grid_probabilities_min) / grid_probabilities_range
                image[j, i] = hsv_to_rgb(hsv[0], a * max_background_saturation, hsv[2] * a + (1 - a))
    ax_map.imshow(image, origin='lower', aspect='auto', extent=ax_map.axis(), clip_on=False, zorder=1)
    mask = zeros_like(grid_probabilities, dtype=bool)
    for i in range(grid_probabilities.shape[0]):
        for j in range(grid_probabilities.shape[1]):
            if not convexhull_region.contains_point((x_grids[i], y_grids[j])):
                mask[i, j] = True
    z = ma.array(grid_probabilities, mask=mask)

    # Plot contours
    ax_map.contour(z.transpose(), n_contours,
                   origin='lower', aspect='auto', extent=ax_map.axis(),
                   corner_mask=True,
                   linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha,
                   aa=True, clip_on=False, zorder=2)

    # Plot sample legends
    for i, s in enumerate(range(1, n_training_states + 1)):
        y = 1 - float(1 / (n_training_states + 1)) * (i + 1)
        c = state_colors[s]
        ax_legend.plot(-0.05, y, marker='s',
                       markersize=legend_markersize, markerfacecolor=c,
                       aa=True, clip_on=False)
        ax_legend.text(0.16, y, 'State {} (n={})'.format(s, (samples.ix[:, 'state'] == s).sum()),
                       fontsize=legend_fontsize, weight='bold', verticalalignment='center')

    if isinstance(annotation, Series):  # Plot samples, annotation, sample legends, and annotation legends
        samples.ix[:, 'annotation'] = annotation

        # Set up annotation min, mean, max, colormap, and range
        if annotation_type == 'continuous':
            if annotation.dtype == object:
                raise TypeError('Continuous annotation values must be numbers (float, int, etc).')
            # Normalize annotation
            samples.ix[:, 'annotation_value'] = normalize_dataframe_or_series(samples.ix[:, 'annotation'], '-0-').clip(
                -std_max,
                std_max)
            # Get annotation statistics
            annotation_min = -std_max
            annotation_mean = samples.ix[:, 'annotation_value'].mean()
            annotation_max = std_max
            # Set color map
            cmap = CMAP_CONTINUOUS
        else:  # Annotation is categorical or binary
            if annotation.dtype == object:  # Convert str annotation to value
                a_to_value = {}
                value_to_a = {}
                for a_i, a in enumerate(annotation.dropna().sort_values().unique()):
                    # 1-to-1 map
                    a_to_value[a] = a_i
                    value_to_a[a_i] = a
                samples.ix[:, 'annotation_value'] = annotation.apply(a_to_value.get)
            else:
                samples.ix[:, 'annotation_value'] = samples.ix[:, 'annotation']
            # Get annotation statistics
            annotation_min = 0
            annotation_mean = int(samples.ix[:, 'annotation_value'].mean())
            annotation_max = int(samples.ix[:, 'annotation_value'].max())
            # Set color map
            if annotation_type == 'categorical':
                cmap = CMAP_CATEGORICAL_2
            elif annotation_type == 'binary':
                cmap = CMAP_BINARY
            else:
                raise ValueError('annotation_type must be one of {continuous, categorical, binary}.')
        # Get annotation range
        annotation_range = annotation_max - annotation_min

        # Plot IC score
        score, p_val = compute_association_and_pvalue(samples.ix[:, 'annotation_value'], samples.ix[:, 'state'])
        ax_legend.text(0.5, 1, '{}\nIC={:.3f} (p-val={:.3f})'.format(annotation_name, score, p_val),
                       fontsize=legend_fontsize * 1.26, weight='bold', horizontalalignment='center')

        # Set plotting order and plot
        if highlight_high_magnitude:
            samples = samples.reindex_axis(
                samples.ix[:, 'annotation_value'].abs().sort_values(na_position='first').index)
        else:
            samples.sort_values('annotation_value', ascending=annotation_ascending, inplace=True)

        for idx, s in samples.iterrows():
            x = s.ix['x']
            y = s.ix['y']
            if isnull(s.ix['annotation_value']):
                c = bad_color
            else:
                if annotation_type == 'continuous':
                    c = cmap((s.ix['annotation_value'] - annotation_min) / annotation_range)
                elif annotation_type in ('categorical', 'binary'):
                    if annotation_range:
                        c = cmap((s.ix['annotation_value'] - annotation_min) / annotation_range)
                    else:
                        c = cmap(0)
                else:
                    raise ValueError('annotation_type must be one of {continuous, categorical, binary}.')
            ax_map.plot(x, y, marker='o',
                        markersize=sample_markersize, markerfacecolor=c,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor,
                        aa=True, clip_on=False, zorder=5)

        if annotation.dtype == object:  # Plot categorical legends below the map
            for i, a in enumerate(sorted(a_to_value, reverse=True)):
                v = a_to_value.get(a)
                x = 1 - float(1 / (len(a_to_value) + 1)) * (i + 1)
                y = -0.1
                if annotation_range:
                    c = cmap((v - annotation_min) / annotation_range)
                else:
                    c = cmap(0)
                if 5 < len(a):
                    rotation = 90
                else:
                    rotation = 0
                ax_map.plot(x, y, marker='o',
                            markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)
                ax_map.text(x, y - 0.03, a, fontsize=legend_fontsize, weight='bold', color=title_fontcolor,
                            rotation=rotation, horizontalalignment='center', verticalalignment='top')

        if annotation_type == 'continuous':  # Plot color bar
            cax, kw = make_axes(ax_legend, location='bottom', fraction=0.1, shrink=1, aspect=8,
                                cmap=cmap, norm=Normalize(vmin=annotation_min, vmax=annotation_max),
                                ticks=[annotation_min, annotation_mean, annotation_max])
            ColorbarBase(cax, **kw)
            cax.set_title('Normalized Values', **{'fontsize': 16, 'weight': 'bold'})

    else:  # Plot samples using state colors
        if 'component_ratio' in samples:
            samples.ix[:, 'component_ratio'] = normalize_dataframe_or_series(samples.ix[:, 'component_ratio'], '0-1')
        for idx, s in samples.iterrows():
            x = s.ix['x']
            y = s.ix['y']
            c = state_colors[s.ix['state']]
            if 'component_ratio' in samples:
                a = s.ix['component_ratio']
            else:
                a = 1
            ax_map.plot(x, y, marker='o',
                        markersize=sample_markersize, markerfacecolor=c, alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor,
                        aa=True, clip_on=False, zorder=5)

    if sample_name_color:  # Plot sample names
        for idx, s in samples.iterrows():
            x = s.ix['x']
            y = s.ix['y']
            ax_map.text(x, y + 0.03, idx,
                        fontsize=sample_name_size, weight='bold', color=sample_name_color, horizontalalignment='center',
                        zorder=7)

    if filepath:
        save_plot(filepath)
