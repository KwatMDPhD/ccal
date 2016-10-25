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

from os.path import join

from colorsys import rgb_to_hsv, hsv_to_rgb
import numpy as np
from numpy import asarray, zeros, zeros_like, ones, empty, linspace, nansum
from pandas import DataFrame, Series, isnull
from scipy.spatial import Delaunay, ConvexHull
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.colorbar import make_axes, ColorbarBase
from seaborn import violinplot, boxplot

from . import SEED
from .support import EPS, print_log, establish_filepath, load_gct, write_gct, write_dictionary, fit_matrix, \
    nmf_consensus_cluster, information_coefficient, normalize_pandas, hierarchical_consensus_cluster, \
    exponential_function, mds, compute_association_and_pvalue, solve_matrix_linear_equation, \
    drop_value_from_dataframe, \
    FIGURE_SIZE, CMAP_CONTINUOUS, CMAP_CATEGORICAL, CMAP_BINARY, save_plot, plot_clustermap, plot_heatmap, plot_nmf, \
    plot_x_vs_y

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


# ======================================================================================================================
# Define components
# ======================================================================================================================
def define_components(matrix, ks, n_jobs=1, n_clusterings=100, random_state=SEED, directory_path=None):
    """
    NMF-consensus cluster samples (matrix columns) and compute cophenetic-correlation coefficients, and save 1 NMF
    results for each k.
    :param matrix: DataFrame; (n_rows, n_columns)
    :param ks: iterable; iterable of int k used for NMF
    :param n_jobs: int;
    :param n_clusterings: int; number of NMF for consensus clustering
    :param random_state: int;
    :param directory_path: str; directory path where
            cophenetic_correlation_coefficients{.pdf, .gct}
            matrices/nmf_k{k}_{w, h}.gct
            figures/nmf_k{k}_{w, h}.pdf
        will be saved.
    :return: dict and dict; {k: {w: W matrix (n_rows, k), h: H matrix (k, n_columns), e: Reconstruction Error}} and
                            {k: Cophenetic Correlation Coefficient}
    """

    # Rank normalize the input matrix by column
    # TODO: try changing n_ranks (choose automatically)
    matrix = normalize_pandas(matrix, 'rank', n_ranks=10000, axis=0)
    plot_clustermap(matrix, title='(Rank-normalized) Matrix to be Decomposed', xlabel='Sample', ylabel='Feature',
                    xticklabels=False, yticklabels=False)

    # NMF-consensus cluster (while saving 1 NMF result per k)
    nmf_results, cophenetic_correlation_coefficient = nmf_consensus_cluster(matrix, ks,
                                                                            n_jobs=n_jobs, n_clusterings=n_clusterings,
                                                                            random_state=random_state)

    # Make NMF directory, where
    #     cophenetic_correlation_coefficients{.pdf, .gct}
    #     matrices/nmf_k{k}_{w, h}.gct
    #     figures/nmf_k{k}_{w, h}.pdf
    # will be saved
    directory_path = join(directory_path, 'nmf/')
    establish_filepath(directory_path)

    # Save and plot NMF cophenetic correlation coefficients
    print_log('Saving and plotting cophenetic correlation coefficients ...')
    write_dictionary(cophenetic_correlation_coefficient,
                     join(directory_path, 'cophenetic_correlation_coefficients.txt'),
                     key_name='k', value_name='cophenetic_correlation_coefficient')
    plot_x_vs_y(sorted(cophenetic_correlation_coefficient.keys()),
                [cophenetic_correlation_coefficient[k] for k in sorted(cophenetic_correlation_coefficient.keys())],
                title='NMF Cophenetic Correlation Coefficient vs. k',
                xlabel='k', ylabel='NMF Cophenetic Correlation Coefficient',
                filepath=join(directory_path, 'cophenetic_correlation_coefficients.pdf'))

    # Save and plot NMF results
    print_log('Saving and plotting NMF results ...')
    _save_nmf(nmf_results, join(directory_path, 'matrices', ''))

    # Save NMF figures
    for k in ks:
        print_log('\tPlotting k={} ...'.format(k))
        plot_nmf(nmf_results, k, filepath=join(directory_path, 'figures', 'nmf_k{}.pdf'.format(k)))

    return nmf_results, cophenetic_correlation_coefficient


def _save_nmf(nmf_results, filepath_prefix):
    """
    Save NMF results.
    :param nmf_results: dict; {k: {w: W matrix, h: H matrix, e: Reconstruction Error}} and
                              {k: Cophenetic Correlation Coefficient}
    :param filepath_prefix: str; filepath_prefix_nmf_k{k}_{w, h}.gct and will be saved
    :return: None
    """

    for k, v in nmf_results.items():
        write_gct(v['w'], filepath_prefix + 'nmf_k{}_w.gct'.format(k))
        write_gct(v['h'], filepath_prefix + 'nmf_k{}_h.gct'.format(k))


def solve_for_components(w_matrix, a_matrix, filepath_prefix=None):
    """
    Get H matrix of a_matrix in the space of w_matrix by solving W * H = A for H.
    :param w_matrix: str or DataFrame; (n_rows, k)
    :param a_matrix: str or DataFrame; (n_rows, n_columns)
    :param filepath_prefix: str; filepath_prefix_solved_nmf_h_k{}.{gct, pdf} will be saved
    :return: DataFrame; (k, n_columns)
    """

    # Load A and W matrices
    w_matrix = load_gct(w_matrix)
    a_matrix = load_gct(a_matrix)

    # Keep only indices shared by both
    common_indices = set(a_matrix.index) & set(w_matrix.index)
    w_matrix = w_matrix.ix[common_indices, :]
    a_matrix = a_matrix.ix[common_indices, :]

    # Average duplicates in A matrix
    a_matrix = a_matrix.groupby(level=0).mean()

    # Rank normalize the A matrix by column
    # TODO: try changing n_ranks (choose automatically)
    a_matrix = normalize_pandas(a_matrix, 'rank', n_ranks=10000, axis=0)

    # Normalize the W matrix by column
    # TODO: improve the normalization (why this normalization?)
    w_matrix = w_matrix.apply(lambda c: c / sum(c) * 1000)

    # Solve W * H = A
    print_log('Solving for components: W({}x{}) * H = A({}x{}) ...'.format(*w_matrix.shape, *a_matrix.shape))
    h_matrix = solve_matrix_linear_equation(w_matrix, a_matrix, method='pinv')

    if filepath_prefix:  # Save H matrix
        write_gct(h_matrix, filepath_prefix + '_solved_nmf_h_k{}.gct'.format(h_matrix.shape[0]))
        plot_filepath = filepath_prefix + '_solved_nmf_h_k{}.pdf'.format(h_matrix.shape[0])
    else:
        plot_filepath = None

    plot_nmf(w_matrix=w_matrix, h_matrix=h_matrix, filepath=plot_filepath)

    return h_matrix


# ======================================================================================================================
# Define states
# ======================================================================================================================
def define_states(matrix, ks, distance_matrix=None, max_std=3, n_clusterings=100, directory_path=None):
    """
    Hierarchical-consensus cluster samples (matrix columns) and compute cophenetic correlation coefficients.
    :param matrix: DataFrame; (n_rows, n_columns);
    :param ks: iterable; iterable of int k used for hierarchical clustering
    :param distance_matrix: str or DataFrame; (n_columns, n_columns); distance matrix to hierarchical cluster
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of hierarchical clusterings for consensus clustering
    :param directory_path: str; directory path where
            distance_matrix.{txt, pdf}
            clusterings.{gct, pdf}
            cophenetic_correlation_coefficients.{txt, pdf}
        will be saved
    :return: DataFrame, DataFrame, and Series;
        distance_matrix (n_samples, n_samples),
        clusterings (n_ks, n_columns), and
        cophenetic correlation coefficients (n_ks)
    """

    # '-0-' normalize by rows and clip values max_std standard deviation away; then '0-1' normalize by rows
    matrix = normalize_pandas(normalize_pandas(matrix, '-0-', axis=1).clip(-max_std,
                                                                           max_std),
                              method='0-1', axis=1)

    # Hierarchical-consensus cluster
    distance_matrix, clusterings, cophenetic_correlation_coefficients = \
        hierarchical_consensus_cluster(matrix, ks, distance_matrix=distance_matrix, n_clusterings=n_clusterings)

    if directory_path:  # Save and plot distance matrix, clusterings, and cophenetic correlation coefficients
        establish_filepath(directory_path)

        # Save results
        distance_matrix.to_csv(join(directory_path, 'distance_matrix.txt'), sep='\t')
        write_gct(clusterings, join(directory_path, 'clusterings.gct'))
        write_dictionary(cophenetic_correlation_coefficients,
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
    plot_clustermap(distance_matrix, title='Distance Matrix', xlabel='Sample', ylabel='Sample',
                    xticklabels=False, yticklabels=False,
                    filepath=filepath_distance_matrix_plot)

    # Plot clusterings
    plot_heatmap(clusterings, sort_axis=1, data_type='categorical', title='Clustering per k', xticklabels=False,
                 filepath=filepath_clusterings_plot)

    # Plot cophenetic correlation coefficients
    plot_x_vs_y(sorted(cophenetic_correlation_coefficients.keys()),
                [cophenetic_correlation_coefficients[k] for k in sorted(cophenetic_correlation_coefficients.keys())],
                title='Consensus Clustering Cophenetic Correlation Coefficients vs. k',
                xlabel='k', ylabel='Cophenetic Score',
                filepath=filepath_cophenetic_correlation_coefficients_plot)

    return distance_matrix, clusterings, cophenetic_correlation_coefficients


# ======================================================================================================================
# Make Onco-GPS map
# ======================================================================================================================
def make_oncogps_map(training_h, training_states, std_max=3, components=None,
                     testing_h=None, testing_states=None, testing_h_normalization='using_training',
                     informational_mds=True, mds_seed=SEED,
                     n_pulls=None, power=None, fit_min=0, fit_max=2, power_min=1, power_max=5,
                     component_ratio=0, n_grids=256, kde_bandwidths_factor=1,
                     annotation=(), annotation_name='', annotation_type='continuous',
                     title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
                     subtitle_fontsize=16, subtitle_fontcolor='#FF0039',
                     colors=None, component_markersize=16, component_markerfacecolor='#000726',
                     component_markeredgewidth=1.69, component_markeredgecolor='#FFFFFF',
                     component_text_position='auto', component_fontsize=22,
                     delaunay_linewidth=1.26, delaunay_linecolor='#000000',
                     n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
                     sample_markersize=19, sample_without_annotation_markerfacecolor='#999999',
                     sample_markeredgewidth=0.92, sample_markeredgecolor='#000000',
                     legend_markersize=22, legend_fontsize=16, effectplot_type='violine',
                     effectplot_mean_markerfacecolor='#FFFFFF', effectplot_mean_markeredgecolor='#FF0082',
                     effectplot_median_markeredgecolor='#FF0082', filepath=None):
    """
    :param training_h: DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param training_states: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values
    :param components: DataFrame; (n_components, 2 [x, y]); component coordinates
    :param testing_h: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param testing_states: iterable of int; (n_samples); sample states
    :param testing_h_normalization: str or None; {using_training, as_training, None}
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param power: str or number; power to raise components' influence on each sample
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param n_pulls: int; [1, n_components]; number of components influencing a sample's coordinate
    :param power_max: number;
    :param component_ratio: number; number if int; percentile if float & < 1
    :param n_grids: int; number of grids; larger the n_grids, higher the resolution
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :param annotation: pandas Series; (n_samples); sample annotation; will color samples based on annotation
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param title: str;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib color;
    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib color;
    :param colors: matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap, or list;
    :param component_markersize: number;
    :param component_markerfacecolor: matplotlib color;
    :param component_markeredgewidth: number;
    :param component_markeredgecolor: matplotlib color;
    :param component_text_position: str; {'auto', 'top', 'bottom'}
    :param component_fontsize: number;
    :param delaunay_linewidth: number;
    :param delaunay_linecolor: matplotlib color;
    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]
    :param background_markersize: number; set to 0 to disable drawing backgrounds
    :param background_mask_markersize: number; set to 0 to disable masking
    :param background_max_alpha: float; [0, 1]; the maximum background alpha (transparency)
    :param sample_markersize: number;
    :param sample_without_annotation_markerfacecolor: matplotlib color;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param legend_markersize: number;
    :param legend_fontsize: number;
    :param effectplot_type: str; {'violine', 'box'}
    :param effectplot_mean_markerfacecolor: matplotlib color;
    :param effectplot_mean_markeredgecolor: matplotlib color;
    :param effectplot_median_markeredgecolor: matplotlib color;
    :param filepath: str;
    :return: DataFrame and DataFrame; components and samples
    """

    # Make sure the index is str (better for .ix)
    training_h.index = training_h.index.astype(str)

    if isinstance(testing_h, DataFrame):
        print_log('\tTesting normalization: {}'.format(testing_h_normalization))
        if not testing_h_normalization:
            normalize_training_h = False
            normalizing_h = None
        elif testing_h_normalization == 'using_training':
            normalize_training_h = True
            normalizing_h = training_h.copy()
        elif testing_h_normalization == 'as_training':
            normalize_training_h = True
            normalizing_h = None
        else:
            raise ValueError('testing_h_normalization must be one of {using_training, as_training, None}.')

    # Preprocess training-H matrix and training states
    training_h, training_states = _process_h_and_states(training_h, training_states, std_max)

    print_log('Training Onco-GPS with {} components, {} samples, and {} states ...'.format(*training_h.shape,
                                                                                           len(set(training_states))))
    print_log('\tComponents: {}.'.format(set(training_h.index)))
    print_log('\tTraining states: {}.'.format(set(training_states)))

    # Compute component coordinates
    if isinstance(components, DataFrame):
        print_log('Using predefined component coordinates ...'.format(components))
        # TODO: enforce matched index
        components.index = training_h.index
    else:
        if informational_mds:
            print_log('Computing component coordinates using informational distance ...')
            distance_function = information_coefficient
        else:
            print_log('Computing component coordinates using Euclidean distance ...')
            distance_function = None
        components = mds(training_h, distance_function=distance_function, mds_seed=mds_seed, standardize=True)

    if not n_pulls:  # n_pulls = number of all components
        n_pulls = training_h.shape[0]

    if not power:
        print_log('Computing component power ...')
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
    # TODO: refactor meaningfully
    training_samples = _process_samples(training_h, training_states, components, n_pulls, power, component_ratio)

    print_log('Computing grid probabilities and states ...')
    grid_probabilities, grid_states = _compute_grid_probabilities_and_states(training_samples, n_grids,
                                                                             kde_bandwidths_factor)

    if isinstance(testing_h, DataFrame):
        # Make sure the index is str (better for .ix)
        testing_h.index = testing_h.index.astype(str)

        print_log('Testing Onco-GPS with {} samples and {} states ...'.format(testing_h.shape[1],
                                                                              len(set(testing_states))))
        print_log('\tTesting states: {}'.format(set(training_states)))

        testing_h, testing_states, = _process_h_and_states(testing_h, testing_states, std_max,
                                                           normalize=normalize_training_h, normalizing_h=normalizing_h)
        testing_samples = _process_samples(testing_h, testing_states, components, n_pulls, power, component_ratio)
        samples = testing_samples
    else:
        samples = training_samples

    print_log('Plotting ...')
    _plot_onco_gps(components, samples, grid_probabilities, grid_states, len(set(training_states)),
                   annotation=annotation, annotation_name=annotation_name, annotation_type=annotation_type,
                   std_max=std_max,
                   title=title, title_fontsize=title_fontsize, title_fontcolor=title_fontcolor,
                   subtitle_fontsize=subtitle_fontsize, subtitle_fontcolor=subtitle_fontcolor,
                   colors=colors,
                   component_markersize=component_markersize, component_markerfacecolor=component_markerfacecolor,
                   component_markeredgewidth=component_markeredgewidth,
                   component_markeredgecolor=component_markeredgecolor,
                   component_text_position=component_text_position, component_fontsize=component_fontsize,
                   delaunay_linewidth=delaunay_linewidth, delaunay_linecolor=delaunay_linecolor,
                   n_contours=n_contours,
                   contour_linewidth=contour_linewidth, contour_linecolor=contour_linecolor,
                   contour_alpha=contour_alpha,
                   sample_markersize=sample_markersize,
                   sample_without_annotation_markerfacecolor=sample_without_annotation_markerfacecolor,
                   sample_markeredgewidth=sample_markeredgewidth, sample_markeredgecolor=sample_markeredgecolor,
                   legend_markersize=legend_markersize, legend_fontsize=legend_fontsize,
                   effectplot_type=effectplot_type, effectplot_mean_markerfacecolor=effectplot_mean_markerfacecolor,
                   effectplot_mean_markeredgecolor=effectplot_mean_markeredgecolor,
                   effectplot_median_markeredgecolor=effectplot_median_markeredgecolor,
                   filepath=filepath)
    return training_h, testing_h, components, samples, grid_probabilities, grid_states


# ======================================================================================================================
# Process H matrix and states
# ======================================================================================================================
def _process_h_and_states(h, states, std_max, normalize=True, normalizing_h=None):
    """
    Process H matrix and states.
    :param h: DataFrame; (n_components, n_samples); H matrix
    :param states: iterable of ints;
    :param std_max: number;
    :param normalizing_h: DataFrame; (n_components, m_samples);
    :return: DataFrame and Series; processed H matrix and states
    """

    # Convert sample-state labels, which match sample, into Series
    states = Series(states, index=h.columns)

    # Normalize H matrix and drop all-0 samples
    h = _process_h(h, std_max, normalize=normalize, normalizing_h=normalizing_h)

    # Drop all-0 samples from states too
    states = states.ix[h.columns]

    return h, states


def _process_h(h, std_max, normalize=True, normalizing_h=None):
    """
    Normalize H matrix and drop all-0 samples.
    :param h: DataFrame; (n_components, n_samples); H matrix
    :param std_max: number;
    :param normalizing_h: DataFrame; (n_components, m_samples);
    :return: DataFrame; (n_components, n_samples); Normalized H matrix
    """

    # Drop all-0 samples
    h = drop_value_from_dataframe(h, 0)

    if normalize:
        # Clip by standard deviation and 0-1 normalize
        h = _normalize_h(h, std_max, normalizing_h=normalizing_h)

        # Drop all-0 samples
        h = drop_value_from_dataframe(h, 0)

    return h


# TODO: consider making a general function in support.py that normalizes with other matrix's values
def _normalize_h(h, std_max, normalizing_h=None):
    """
    Clip by standard deviation and 0-1 normalize the rows of H matrix.
    :param h: DataFrame; (n_components, n_samples); H matrix
    :param std_max: number;
    :param normalizing_h: DataFrame; (n_components, m_samples);
    :return: DataFrame; (n_components, n_samples); Normalized H matrix
    """

    if isinstance(normalizing_h, DataFrame):  # Normalize using statistics from normalizing-H matrix
        # -0-
        for r_i, r in normalizing_h.iterrows():
            mean = r.mean()
            std = r.std()
            if std == 0:
                h.ix[r_i, :] = h.ix[r_i, :] / r.size
                normalizing_h.ix[r_i, :] = r / r.size
            else:
                h.ix[r_i, :] = (h.ix[r_i, :] - mean) / std
                normalizing_h.ix[r_i, :] = (r - mean) / std

        # Clip
        h = h.clip(-std_max, std_max)
        # 0-1
        for r_i, r in normalizing_h.iterrows():
            r_min = r.min()
            r_max = r.max()
            if r_max - r_min == 0:
                h.ix[r_i, :] = h.ix[r_i, :] / r.size
            else:
                h.ix[r_i, :] = (h.ix[r_i, :] - r_min) / (r_max - r_min)

    else:  # Normalize using statistics from H matrix
        # -0-
        h = normalize_pandas(h, '-0-', axis=1)
        # Clip
        h = h.clip(-std_max, std_max)
        # 0-1
        h = normalize_pandas(h, '0-1', axis=1)

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

    fit_parameters = fit_matrix(h, exponential_function, sort_matrix=True)
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
    :return: DataFrame; (n_samples, 4 [x, y, state, component_ratio])
    """
    samples = DataFrame(index=h.columns, columns=['x', 'y', 'state', 'component_ratio'])
    samples.ix[:, 'state'] = states

    print_log('Computing sample coordinates using {} components and {:.3f} power ...'.format(n_pulls, power))
    samples.ix[:, ['x', 'y']] = _compute_sample_coordinates(components, h, n_pulls, power)

    if component_ratio and 0 < component_ratio:
        print_log('Computing component ratios ...')
        samples.ix[:, 'component_ratio'] = _compute_component_ratios(h, component_ratio)
    else:
        samples.ix[:, 'component_ratio'] = 1

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
        n = h.shape[0] * n

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
# Plot Onco-GPS map
# ======================================================================================================================
def _plot_onco_gps(components, samples, grid_probabilities, grid_states, n_training_states,
                   annotation=(), annotation_name='', annotation_type='continuous', std_max=3,
                   title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
                   subtitle_fontsize=16, subtitle_fontcolor='#FF0039', colors=None,
                   component_markersize=13, component_markerfacecolor='#000726', component_markeredgewidth=1.69,
                   component_markeredgecolor='#FFFFFF', component_text_position='auto', component_fontsize=16,
                   delaunay_linewidth=1, delaunay_linecolor='#000000', max_background_saturation=0.66,
                   n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
                   sample_markersize=12, sample_without_annotation_markerfacecolor='#999999',
                   sample_markeredgewidth=0.81, sample_markeredgecolor='#000000', plot_sample_names=False,
                   legend_markersize=10, legend_fontsize=11,
                   effectplot_type='violine', effectplot_mean_markerfacecolor='#FFFFFF',
                   effectplot_mean_markeredgecolor='#FF0082', effectplot_median_markeredgecolor='#FF0082',
                   filepath=None):
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
    :param std_max: number; threshold to clip standardized values
    :param title: str;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib color;
    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib color;
    :param colors: matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap, or list;
    :param component_markersize: number;
    :param component_markerfacecolor: matplotlib color;
    :param component_markeredgewidth: number;
    :param component_markeredgecolor: matplotlib color;
    :param component_text_position: str; {'auto', 'top', 'bottom'}
    :param component_fontsize: number;
    :param delaunay_linewidth: number;
    :param delaunay_linecolor: matplotlib color;
    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]
    :param sample_markersize: number;
    :param sample_without_annotation_markerfacecolor: matplotlib color;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param plot_sample_names: bool; plot sample names or not
    :param legend_markersize: number;
    :param legend_fontsize: number;
    :param effectplot_type: str; {'violine', 'box'}
    :param effectplot_mean_markerfacecolor: matplotlib color;
    :param effectplot_mean_markeredgecolor: matplotlib color;
    :param effectplot_median_markeredgecolor: matplotlib color;
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
    ax_title = plt.subplot(gridspec[0, :7])
    ax_title.axis([0, 1, 0, 1])
    ax_title.axis('off')

    # Set up colorbar ax
    ax_colorbar = plt.subplot(gridspec[0, 7:12])
    ax_colorbar.axis([0, 1, 0, 1])
    ax_colorbar.axis('off')

    # Set up map ax
    ax_map = plt.subplot(gridspec[1:, :12])
    ax_map.axis([0, 1, 0, 1])
    ax_map.axis('off')

    # Set up legend ax
    ax_legend = plt.subplot(gridspec[1:, 14:])
    ax_legend.axis('off')

    # Plot title
    ax_title.text(0, 1, title,
                  fontsize=title_fontsize, weight='bold', color=title_fontcolor)
    ax_title.text(0, 0.6, '{} samples, {} components, and {} states'.format(samples.shape[0],
                                                                            components.shape[0],
                                                                            n_training_states),
                  fontsize=subtitle_fontsize, weight='bold', color=subtitle_fontcolor)

    # Plot components and their labels
    ax_map.plot(components.ix[:, 'x'], components.ix[:, 'y'], marker='D', linestyle='',
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor,
                aa=True, clip_on=False, zorder=6)

    # Compute convexhull
    convexhull = ConvexHull(components)
    convexhull_region = Path(convexhull.points[convexhull.vertices])

    # Put labels on top or bottom of the component markers
    vertical_shift = -0.03
    for i in components.index:
        # Compute vertical shift
        if component_text_position == 'auto':
            if convexhull_region.contains_point((components.ix[i, 'x'], components.ix[i, 'y'] + vertical_shift)):
                vertical_shift *= -1
        elif component_text_position == 'top':
            vertical_shift *= -1
        elif component_text_position == 'bottom':
            pass

        x = components.ix[i, 'x']
        y = components.ix[i, 'y'] + vertical_shift
        ax_map.text(x, y, i,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=component_fontsize, weight='bold', color=component_markerfacecolor,
                    zorder=6)

    # Plot Delaunay triangulation
    delaunay = Delaunay(components)
    ax_map.triplot(delaunay.points[:, 0], delaunay.points[:, 1], delaunay.simplices.copy(),
                   linewidth=delaunay_linewidth, color=delaunay_linecolor,
                   aa=True, clip_on=False, zorder=4)

    # Assign colors to states
    states_color = {}
    # TODO: remove
    if colors == 'paper':
        colors = [[0.80392157, 0.58823529, 0.80392157, 1.],
                  [0.36078431, 0.6745098, 0.93333333, 1.],
                  [0.2627451, 0.80392157, 0.50196078, 1.],
                  [1., 0.64705882, 0., 1.],
                  [0.80392157, 0.33333333, 0.33333333, 1.],
                  [0.94117647, 0.64705882, 0.67058824, 1.],
                  [0.60392157, 0.78039216, 0.9372549, 1.],
                  [0.83921569, 0.63921569, 0.98823529, 1.],
                  [1., 0.88235294, 0.8627451, 1.],
                  [0.98039216, 0.94901961, 0.74509804, 1.],
                  [0.95294118, 0.78039216, 0.94901961, 1.],
                  [0.77647059, 0.98039216, 0.37647059, 1.],
                  [0.97647059, 0.43921569, 0.97647059, 1.],
                  [0.98823529, 0.5372549, 0.38431373, 1.],
                  [0.96470588, 0.89019608, 0.43921569, 1.],
                  [0.94117647, 0.95686275, 0.25882353, 1.],
                  [0.68235294, 0.83137255, 0.92941176, 1.],
                  [0.85098039, 0.85098039, 0.85098039, 1.],
                  [0.99215686, 0.60784314, 0.52156863, 1.],
                  [0.49803922, 1., 0., 1.],
                  [1., 0.7254902, 0.05882353, 1.],
                  [0.43137255, 0.54509804, 0.23921569, 1.],
                  [0.54509804, 0.53333333, 0.47058824, 1.],
                  [0.49803922, 1., 0.83137255, 1.],
                  [0., 0., 0.54509804, 1.],
                  [0.82352941, 0.70588235, 0.54901961, 1.],
                  [0., 0.39215686, 0., 1.]]
    for i, s in enumerate(range(1, n_training_states + 1)):
        if colors:
            if isinstance(colors, ListedColormap) or isinstance(colors, LinearSegmentedColormap):
                states_color[s] = colors(s)
            else:
                states_color[s] = colors[i]
        else:
            states_color[s] = CMAP_CATEGORICAL(int(s / n_training_states * CMAP_CATEGORICAL.N))

    # Plot background
    grid_probabilities_min = grid_probabilities.min()
    grid_probabilities_max = grid_probabilities.max()
    grid_probabilities_range = grid_probabilities_max - grid_probabilities_min

    image = ones((*grid_probabilities.shape, 3))
    for i in range(grid_probabilities.shape[0]):
        for j in range(grid_probabilities.shape[1]):
            if convexhull_region.contains_point((x_grids[i], y_grids[j])):
                rgba = states_color[grid_states[i, j]]
                hsv = rgb_to_hsv(*rgba[:3])
                a = (grid_probabilities[i, j] - grid_probabilities_min) / grid_probabilities_range
                image[j, i] = hsv_to_rgb(hsv[0], a * max_background_saturation, hsv[2])
    ax_map.imshow(image,
                  origin='lower', aspect='auto', extent=ax_map.axis(),
                  clip_on=False, zorder=1)

    mask = zeros_like(grid_probabilities, dtype=bool)
    for i in range(grid_probabilities.shape[0]):
        for j in range(grid_probabilities.shape[1]):
            if not convexhull_region.contains_point((x_grids[i], y_grids[j])):
                mask[i, j] = True
    z = np.ma.array(grid_probabilities, mask=mask)

    # Plot contours
    ax_map.contour(z.transpose(), n_contours,
                   origin='lower', aspect='auto', extent=ax_map.axis(),
                   corner_mask=True,
                   linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha,
                   aa=True, clip_on=False, zorder=2)

    if any(annotation):  # Plot samples, annotation, sample legends, and annotation legends
        # Set up annotation
        a = Series(annotation)
        a.index = samples.index

        # Set up annotation min, mean, max, and colormap
        if annotation_type == 'continuous':

            # Normalize annotation
            samples.ix[:, 'annotation'] = normalize_pandas(a, '-0-').clip(-std_max, std_max)

            # Get annotation statistics
            annotation_min = max(-std_max, samples.ix[:, 'annotation'].min())
            annotation_mean = samples.ix[:, 'annotation'].mean()
            annotation_max = min(std_max, samples.ix[:, 'annotation'].max())

            # Set color map
            cmap = CMAP_CONTINUOUS

        else:  # Annotation is categorical or binary
            samples.ix[:, 'annotation'] = annotation

            # Get annotation statistics
            annotation_min = 0
            annotation_mean = int(samples.ix[:, 'annotation'].mean())
            annotation_max = int(samples.ix[:, 'annotation'].max())

            # Set color map
            if annotation_type == 'categorical':
                cmap = CMAP_CATEGORICAL
            elif annotation_type == 'binary':
                cmap = CMAP_BINARY
            else:
                raise ValueError('Annotation type must be one of {continuous, categorical, binary}.')

        # Get annotation range
        annotation_range = annotation_max - annotation_min

        # Plot annotated samples
        # TODO: add component_ratio logic
        for idx, s in samples.iterrows():
            x = s.ix['x']
            y = s.ix['y']
            if isnull(s.ix['annotation']):
                c = sample_without_annotation_markerfacecolor
            else:
                if annotation_type == 'continuous':
                    c = cmap(s.ix['annotation'])
                elif annotation_type in ('categorical', 'binary'):
                    c = cmap((s.ix['annotation'] - annotation_min) / annotation_range)
                else:
                    raise ValueError('Annotation type must be one of {continuous, categorical, binary}.')

            ax_map.plot(x, y, marker='o',
                        markersize=sample_markersize, markerfacecolor=c,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor,
                        aa=True, clip_on=False, zorder=5)

        # Plot sample legends
        ax_legend.axis('on')
        ax_legend.patch.set_visible(False)
        score, p_val = compute_association_and_pvalue(annotation, samples.ix[:, 'state'])
        ax_legend.set_title('{}\nIC={:.3f} (p-val={:.3f})'.format(annotation_name, score, p_val),
                            fontsize=legend_fontsize * 1.26, weight='bold')

        # Plot effect plot
        if effectplot_type == 'violine':
            violinplot(x=samples.ix[:, 'annotation'], y=samples.ix[:, 'state'], palette=states_color, scale='count',
                       inner=None, orient='h', ax=ax_legend, clip_on=False)
            boxplot(x=samples.ix[:, 'annotation'], y=samples.ix[:, 'state'], showbox=False, showmeans=True,
                    medianprops={'marker': 'o',
                                 'markerfacecolor': effectplot_mean_markerfacecolor,
                                 'markeredgewidth': 0.9,
                                 'markeredgecolor': effectplot_mean_markeredgecolor},
                    meanprops={'color': effectplot_median_markeredgecolor}, orient='h', ax=ax_legend)

        elif effectplot_type == 'box':
            boxplot(x=samples.ix[:, 'annotation'], y=samples.ix[:, 'state'], palette=states_color, showmeans=True,
                    medianprops={'marker': 'o',
                                 'markerfacecolor': effectplot_mean_markerfacecolor,
                                 'markeredgewidth': 0.9,
                                 'markeredgecolor': effectplot_mean_markeredgecolor},
                    meanprops={'color': effectplot_median_markeredgecolor}, orient='h', ax=ax_legend)
        else:
            raise ValueError('effectplot_type must be one of {violine, box}.')

        # Set up x label, ticks, and lines
        ax_legend.set_xlabel('')
        ax_legend.set_xticks([annotation_min, annotation_mean, annotation_max])
        for t in ax_legend.get_xticklabels():
            t.set(rotation=90, size=legend_fontsize * 0.9, weight='bold')
        ax_legend.axvline(annotation_min, color='#000000', ls='-', alpha=0.16, aa=True, clip_on=False)
        ax_legend.axvline(annotation_mean, color='#000000', ls='-', alpha=0.39, aa=True, clip_on=False)
        ax_legend.axvline(annotation_max, color='#000000', ls='-', alpha=0.16, aa=True, clip_on=False)

        # Set up y label, ticks, and lines
        ax_legend.set_ylabel('')
        ax_legend.set_yticklabels(
            ['State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)) for s in range(1, n_training_states + 1)],
            fontsize=legend_fontsize, weight='bold')
        ax_legend.yaxis.tick_right()

        # Plot sample markers
        l, r = ax_legend.axis()[:2]
        x = l - float((r - l) / 5)
        for i, s in enumerate(samples.ix[:, 'state'].unique()):
            c = states_color[s]
            ax_legend.plot(x, i, marker='o', markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)

        # Plot color bar
        if annotation_type == 'continuous':
            cax, kw = make_axes(ax_colorbar, location='top', fraction=0.39, shrink=1, aspect=16,
                                cmap=cmap, norm=Normalize(vmin=annotation_min, vmax=annotation_max),
                                ticks=[annotation_min, annotation_mean, annotation_max])
            ColorbarBase(cax, **kw)

    else:  # Plot samples and sample legends
        # TODO: add component_ratio logic
        # Plot samples
        for idx, s in samples.iterrows():
            x = s.ix['x']
            y = s.ix['y']
            c = states_color[s.ix['state']]
            ax_map.plot(x, y, marker='o',
                        markersize=sample_markersize, markerfacecolor=c,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor,
                        aa=True, clip_on=False, zorder=5)

        # Plot sample legends
        ax_legend.axis([0, 1, 0, 1])
        for i, s in enumerate(range(1, n_training_states + 1)):
            y = 1 - float(1 / (n_training_states + 1)) * (i + 1)
            c = states_color[s]
            ax_legend.plot(0.12, y, marker='o',
                           markersize=legend_markersize, markerfacecolor=c,
                           aa=True, clip_on=False)
            ax_legend.text(0.26, y, 'State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)),
                           fontsize=legend_fontsize, weight='bold', verticalalignment='center')

    if plot_sample_names:  # Plot sample names
        for idx, s in samples.iterrows():
            x = s.ix['x']
            y = s.ix['y']
            ax_map.text(x, y + 0.03, idx,
                        fontsize=legend_fontsize, weight='bold', color=title_fontcolor, horizontalalignment='center',
                        zorder=7)

    if filepath:
        save_plot(filepath)
