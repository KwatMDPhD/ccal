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
from numpy import asarray, zeros, empty, linspace
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
    nmf_consensus_cluster, information_coefficient, normalize_pandas_object, hierarchical_consensus_cluster, \
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
    matrix = normalize_pandas_object(matrix, 'rank', n_ranks=10000, axis=0)
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


def solve_for_components(a_matrix, w_matrix, filepath_prefix=None):
    """
    Get H matrix of a_matrix in the space of w_matrix by solving W * H = A for H.
    :param a_matrix: str or DataFrame; (n_rows, n_columns)
    :param w_matrix: str or DataFrame; (n_rows, k)
    :param filepath_prefix: str; filepath_prefix_solved_nmf_h_k{}.{gct, pdf} will be saved
    :return: DataFrame; (k, n_columns)
    """

    # Load A and W matrices
    a_matrix = load_gct(a_matrix)
    w_matrix = load_gct(w_matrix)

    # Keep only indices shared by both
    common_indices = set(a_matrix.index) & set(w_matrix.index)
    a_matrix = a_matrix.ix[common_indices, :]
    w_matrix = w_matrix.ix[common_indices, :]

    # Rank normalize the A matrix by column
    # TODO: try changing n_ranks (choose automatically)
    a_matrix = normalize_pandas_object(a_matrix, 'rank', n_ranks=10000, axis=0)

    # Normalize the W matrix by column
    # TODO: improve the normalization (why this normalization?)
    w_matrix = w_matrix.apply(lambda c: c / sum(c) * 1000)

    # Solve W * H = A
    h_matrix = solve_matrix_linear_equation(w_matrix, a_matrix)

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
    :return: DataFrame and Series; assignment matrix (n_ks, n_columns) and cophenetic correlation coefficients (n_ks)
    """

    # '-0-' normalize by rows and clip values max_std standard deviation away; then '0-1' normalize by rows
    matrix = normalize_pandas_object(normalize_pandas_object(matrix, '-0-', axis=1).clip(-max_std,
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
def make_oncogps_map(training_h, training_states, components=None, std_max=3,
                     testing_h=None, testing_h_normalization='exact_as_training', testing_states=None,
                     informational_mds=True, mds_seed=SEED,
                     power=None, fit_min=0, fit_max=2, power_min=1, power_max=5, n_pulls=None,
                     component_ratio=0,
                     kde_bandwidths_factor=1,
                     annotation=(), annotation_name='', annotation_type='continuous',
                     title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
                     subtitle_fontsize=16, subtitle_fontcolor='#FF0039',
                     colors=None, component_markersize=13, component_markerfacecolor='#000726',
                     component_markeredgewidth=1.69, component_markeredgecolor='#FFFFFF',
                     component_text_position='auto', component_fontsize=16,
                     delaunay_linewidth=1, delaunay_linecolor='#000000',
                     n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
                     background_markersize=5.55, background_mask_markersize=7, background_max_alpha=0.9,
                     sample_markersize=16, sample_without_annotation_markerfacecolor='#999999',
                     sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
                     legend_markersize=10, legend_fontsize=11, effectplot_type='violine',
                     effectplot_mean_markerfacecolor='#FFFFFF', effectplot_mean_markeredgecolor='#FF0082',
                     effectplot_median_markeredgecolor='#FF0082', filepath=None):
    """
    :param training_h: DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param training_states: iterable of int; (n_samples); sample states
    :param components: DataFrame; (n_components, 2 [x, y]); component coordinates
    :param std_max: number; threshold to clip standardized values
    :param testing_h: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param testing_h_normalization: str or None; {'as_train', 'clip_and_0-1', None}
    :param testing_states: iterable of int; (n_samples); sample states
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param power: str or number; power to raise components' influence on each sample
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;
    :param n_pulls: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_ratio: number; number if int; percentile if float & < 1
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

    # Compute coordinates of components and samples and compute sample probabilities and states at each grid
    components, samples, gp, gs = _make_onco_gps_elements(training_h, training_states, std_max,
                                                          components, informational_mds, mds_seed,
                                                          power, fit_min, fit_max, power_min, power_max, n_pulls,
                                                          component_ratio,
                                                          128, kde_bandwidths_factor)
    if isinstance(testing_h, DataFrame):
        testing_h = _normalize_testing_h(testing_h, testing_h_normalization, training_h, std_max)
        samples = _load_samples(testing_h, testing_states,
                                power, fit_min, fit_max, power_min, power_max, n_pulls,
                                components,
                                component_ratio)

    _plot_onco_gps(components, samples, gp, gs, len(set(training_states)),
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
                   background_markersize=background_markersize, background_mask_markersize=background_mask_markersize,
                   background_max_alpha=background_max_alpha,
                   sample_markersize=sample_markersize,
                   sample_without_annotation_markerfacecolor=sample_without_annotation_markerfacecolor,
                   sample_markeredgewidth=sample_markeredgewidth, sample_markeredgecolor=sample_markeredgecolor,
                   legend_markersize=legend_markersize, legend_fontsize=legend_fontsize,
                   effectplot_type=effectplot_type, effectplot_mean_markerfacecolor=effectplot_mean_markerfacecolor,
                   effectplot_mean_markeredgecolor=effectplot_mean_markeredgecolor,
                   effectplot_median_markeredgecolor=effectplot_median_markeredgecolor,
                   filepath=filepath)

    return components, samples


# TODO: allow non-int state labels
def _make_onco_gps_elements(h, states, std_max,
                            components, informational_mds, mds_seed,
                            power, fit_min, fit_max, power_min, power_max, n_pulls,
                            component_ratio,
                            n_grids, kde_bandwidths_factor):
    """
    Compute coordinates of components and samples and compute sample probabilities and states at each grid.
    :param h: pandas DataFrame; (n_nmf_components, n_samples); NMF H matrix
    :param states: iterable of int; (n_samples); sample states
    :param components: DataFrame; (n_nmf_components, 2 ('x', 'y')); component coordinates
    :param std_max: number; threshold to clip standardized values
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param power: str or number; power to raise components' influence on each sample
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;
    :param n_pulls: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_ratio: number; number if int; percentile if < 1
    :param n_grids: int;
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :return: DataFrame, DataFrame, array, and array;
                 components (n_components, 2 [x, y]),
                 samples (n_samples, 4 [x, y, state, component_ratio]),
                 grid_probabilities (n_grids, n_grids),
                 and grid_states (n_grids, n_grids)
    """

    # Preprocess
    h, states = _process_h_and_states(h, states, std_max)
    print_log('Making Onco-GPS with {} components, {} samples, and {} states ...'.format(*h.shape, len(set(states))))
    print_log('\tComponents: {}'.format(set(h.index)))
    print_log('\tStates: {}'.format(set(states)))

    # Compute component coordinates
    if isinstance(components, DataFrame):
        print_log('Using predefined component coordinates ...'.format(components))
        # TODO: enforce matched index
        components.index = h.index
    else:
        if informational_mds:
            print_log('Computing component coordinates using informational distance ...')
            distance_function = information_coefficient
        else:
            print_log('Computing component coordinates using Euclidean distance ...')
            distance_function = None
        components = mds(h, distance_function=distance_function, mds_seed=mds_seed, standardize=True)

    samples = _load_samples(h, states, power, fit_min, fit_max, power_min, power_max, n_pulls, components,
                            component_ratio)

    print_log('Computing grid probabilities and states ...')
    grid_probabilities, grid_states = _compute_grid_probabilities_and_states(samples, n_grids, kde_bandwidths_factor)

    return components, samples, grid_probabilities, grid_states


def _process_h_and_states(h, states, std_max):
    """
    Make sure states is Series.
    Normalize h.
    Drop 0 samples.
    :param h: DataFrame;
    :param states: iterable; iterable of int
    :param std_max: number;
    :return: DataFrame, Series; h and states
    """

    # Convert sample-state labels into Series matching corresponding sample
    states = Series(states, index=h.columns)

    # Drop columns with all-0 values
    h = drop_value_from_dataframe(h, 0)

    # Clip by standard deviation and 0-1 normalize the data
    h = normalize_pandas_object(h, '-0-', axis=1).clip(-std_max, std_max)
    h = normalize_pandas_object(h, '0-1', axis=1)

    # Drop columns with all-0 values
    h = drop_value_from_dataframe(h, 0)

    states = states.ix[h.columns]

    return h, states


def _load_samples(h, states, power, fit_min, fit_max, power_min, power_max, n_pulls, components, component_ratio):
    """

    :param h:
    :param states:
    :param power:
    :param fit_min:
    :param fit_max:
    :param power_min:
    :param power_max:
    :param n_pulls:
    :param components:
    :param component_ratio:
    :return: DataFrame;
    """

    samples = DataFrame(index=h.columns, columns=['x', 'y', 'state', 'component_ratio'])
    samples.ix[:, 'state'] = states

    # Compute sample coordinates
    if not power:
        print_log('Computing component power ...')
        if h.shape[0] < 4:
            print_log('\tCould\'t model with Ae^(kx) + C; too few data points.')
            power = 1
        else:
            power = _compute_component_power(h, fit_min, fit_max, power_min, power_max)

    print_log('Computing sample coordinates using {} components and {:.3f} power ...'.format(n_pulls, power))
    samples.ix[:, ['x', 'y']] = _compute_sample_coordinates(components, h, n_pulls, power)

    if component_ratio and 0 < component_ratio:
        print_log('Computing component ratios ...')
        samples.ix[:, 'component_ratio'] = _compute_component_ratio(h, n_pulls)
    else:
        samples.ix[:, 'component_ratio'] = 1

    return samples


def _compute_component_power(h, fit_min, fit_max, power_min, power_max):
    """
    Compute component power by fitting component magnitudes of samples to exponential function.
    :param h: DataFrame;
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;
    :return: float
    """

    fit_parameters = fit_matrix(h, exponential_function, sort_matrix=True)
    k = fit_parameters[1]

    # Linear transform
    k_zero_to_one = (k - fit_min) / (fit_max - fit_min)
    k_rescaled = k_zero_to_one * (power_max - power_min) + power_min

    return k_rescaled


def _compute_sample_coordinates(component_x_coordinates, component_x_samples, n_influencing_components, power):
    """
    Compute sample coordinates based on component coordinates (components pull samples).
    :param component_x_coordinates: DataFrame; (n_points, 2 [x, y])
    :param component_x_samples: DataFrame; (n_points, n_samples)
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param power: number; power to raise components' influence on each sample
    :return: DataFrame; (n_samples, 2 [x, y])
    """

    component_x_coordinates = asarray(component_x_coordinates)

    sample_coordinates = empty((component_x_samples.shape[1], 2))

    if not n_influencing_components:  # n_influencing_components = number of all components
        n_influencing_components = component_x_samples.shape[0]

    for i, (_, c) in enumerate(component_x_samples.iteritems()):
        c = asarray(c)

        # Silence components that are not pulling
        threshold = sorted(c)[-n_influencing_components]
        c[c < threshold] = 0

        x = sum(c ** power * component_x_coordinates[:, 0]) / sum(c ** power)
        y = sum(c ** power * component_x_coordinates[:, 1]) / sum(c ** power)

        sample_coordinates[i] = x, y

    return sample_coordinates


def _compute_component_ratio(h, n):
    """
    Compute the ratio between the sum of the top-n component values and the sum of the rest of the component values.
    :param h: DataFrame;
    :param n: number;
    :return: array;
    """

    ratios = zeros(h.shape[1])

    if n and n < 1:  # If n is a fraction, compute its respective number
        n = h.shape[0] * n

    # Compute pull ratio for each sample (column)
    for i, (c_idx, c) in enumerate(h.iteritems()):
        c_sorted = c.sort_values(ascending=False)
        ratios[i] = c_sorted[:n].sum() / max(c_sorted[n:].sum(), EPS) * c.sum()

    return ratios


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


def _normalize_testing_h(testing_h, normalization, training_h, std_max):
    """

    :param testing_h:
    :param normalization:
    :param training_h:
    :param std_max:
    :return:
    """

    if normalization == 'exact_as_training':  # Normalize as done on training H using the same normalizing factors
        for r_i, r in training_h.iterrows():
            if r.std() == 0:
                testing_h.ix[r_i, :] = testing_h.ix[r_i, :] / r.size()
            else:
                testing_h.ix[r_i, :] = (testing_h.ix[r_i, :] - r.mean()) / r.std()

    elif normalization == 'as_training':  # Normalize as done on training H
        testing_h = normalize_pandas_object(testing_h, '-0-', axis=1).clip(-std_max, std_max)
        testing_h = normalize_pandas_object(testing_h, '0-1', axis=1)

    return testing_h


def _plot_onco_gps(components, samples, grid_probabilities, grid_states, n_training_states,
                   annotation=(), annotation_name='', annotation_type='continuous', std_max=3,
                   title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
                   subtitle_fontsize=16, subtitle_fontcolor='#FF0039', colors=None,
                   component_markersize=13, component_markerfacecolor='#000726', component_markeredgewidth=1.69,
                   component_markeredgecolor='#FFFFFF', component_text_position='auto', component_fontsize=16,
                   delaunay_linewidth=1, delaunay_linecolor='#000000',
                   n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
                   background_markersize=5.55, background_mask_markersize=7, background_max_alpha=0.9,
                   sample_markersize=12, sample_without_annotation_markerfacecolor='#999999',
                   sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
                   legend_markersize=10, legend_fontsize=11,
                   effectplot_type='violine', effectplot_mean_markerfacecolor='#FFFFFF',
                   effectplot_mean_markeredgecolor='#FF0082', effectplot_median_markeredgecolor='#FF0082',
                   filepath=None):
    """
    Plot Onco-GPS map.
    :param components: DataFrame; (n_components, [x, y]);
        output from _make_onco_gps_elements
    :param samples: DataFrame; (n_samples, [x, y, state])
    :param grid_probabilities: numpy 2D array; (n_grids, n_grids)
    :param grid_states: numpy 2D array; (n_grids, n_grids)
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
    :return: None
    """

    x_grids = linspace(0, 1, grid_probabilities.shape[0])
    y_grids = linspace(0, 1, grid_probabilities.shape[1])

    #
    # Figure and axes
    #
    # Set up figure and axes
    plt.figure(figsize=FIGURE_SIZE)
    gridspec = GridSpec(10, 16)
    ax_title = plt.subplot(gridspec[0, :7])
    ax_title.axis([0, 1, 0, 1])
    ax_title.axis('off')
    ax_colorbar = plt.subplot(gridspec[0, 7:12])
    ax_colorbar.axis([0, 1, 0, 1])
    ax_colorbar.axis('off')
    ax_map = plt.subplot(gridspec[1:, :12])
    ax_map.axis([0, 1, 0, 1])
    ax_map.axis('off')
    ax_legend = plt.subplot(gridspec[1:, 14:])
    ax_legend.axis('off')

    #
    # Title
    #
    ax_title.text(0, 0.9, title, fontsize=title_fontsize, color=title_fontcolor, weight='bold')
    ax_title.text(0, 0.39, '{} samples, {} components, and {} states'.format(samples.shape[0], components.shape[0],
                                                                             n_training_states),
                  fontsize=subtitle_fontsize, color=subtitle_fontcolor, weight='bold')

    #
    # Components
    #
    # Plot components and their labels
    ax_map.plot(components.ix[:, 'x'], components.ix[:, 'y'], marker='D', linestyle='',
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor,
                clip_on=False, aa=True, zorder=6)

    # Compute convexhull
    convexhull = ConvexHull(components)
    convexhull_region = Path(convexhull.points[convexhull.vertices])

    # Put labels on top or bottom of the component markers
    component_text_verticalshift = -0.03
    for i in components.index:
        if component_text_position == 'auto':
            if convexhull_region.contains_point((components.ix[i, 'x'],
                                                 components.ix[i, 'y'] + component_text_verticalshift)):
                component_text_verticalshift *= -1

        elif component_text_position == 'top':
            component_text_verticalshift *= -1

        elif component_text_position == 'bottom':
            pass

        x, y = components.ix[i, 'x'], components.ix[i, 'y'] + component_text_verticalshift
        ax_map.text(x, y, i, fontsize=component_fontsize, color=component_markerfacecolor, weight='bold',
                    horizontalalignment='center', verticalalignment='center', zorder=6)

    # Plot Delaunay triangulation
    delaunay = Delaunay(components)
    ax_map.triplot(delaunay.points[:, 0], delaunay.points[:, 1], delaunay.simplices.copy(),
                   linewidth=delaunay_linewidth, color=delaunay_linecolor, aa=True, zorder=4)

    #
    # Contours
    #
    # Plot contours
    if n_contours > 0:
        ax_map.contour(x_grids, y_grids, grid_probabilities, n_contours, corner_mask=True,
                       linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha, aa=True, zorder=2)

    #
    # State colors
    #
    # Assign colors to states
    states_color = {}
    if colors == 'paper':
        colors = ['#cd96cd', '#5cacee', '#43cd80', '#ffa500', '#cd5555', '#F0A5AB', '#9AC7EF', '#D6A3FC', '#FFE1DC',
                  '#FAF2BE', '#F3C7F2', '#C6FA60', '#F970F9', '#FC8962', '#F6E370', '#F0F442', '#AED4ED', '#D9D9D9',
                  '#FD9B85', '#7FFF00', '#FFB90F', '#6E8B3D', '#8B8878', '#7FFFD4', '#00008b', '#d2b48c', '#006400']
    for i, s in enumerate(range(1, n_training_states + 1)):
        if colors:
            if isinstance(colors, ListedColormap) or isinstance(colors, LinearSegmentedColormap):
                states_color[s] = colors(s)
            else:
                states_color[s] = colors[i]
        else:
            states_color[s] = CMAP_CATEGORICAL(int(s / n_training_states * CMAP_CATEGORICAL.N))

    #
    # Background
    #
    # Plot background
    if background_markersize > 0:
        grid_probabilities_min = grid_probabilities.min()
        grid_probabilities_max = grid_probabilities.max()
        grid_probabilities_range = grid_probabilities_max - grid_probabilities_min
        for i in range(grid_probabilities.shape[0]):
            for j in range(grid_probabilities.shape[1]):
                if convexhull_region.contains_point((x_grids[i], y_grids[j])):
                    c = states_color[grid_states[i, j]]
                    a = min(background_max_alpha,
                            (grid_probabilities[i, j] - grid_probabilities_min) / grid_probabilities_range)
                    ax_map.plot(x_grids[i], y_grids[j], marker='s', markersize=background_markersize, markerfacecolor=c,
                                alpha=a, aa=True, zorder=1)
    # Plot background mask
    if background_mask_markersize > 0:
        for i in range(grid_probabilities.shape[0]):
            for j in range(grid_probabilities.shape[1]):
                if not convexhull_region.contains_point((x_grids[i], y_grids[j])):
                    ax_map.plot(x_grids[i], y_grids[j], marker='s', markersize=background_mask_markersize,
                                markerfacecolor='w', aa=True, zorder=3)

    #
    # Annotation
    #
    if any(annotation):  # Plot samples, annotation, sample legends, and annotation legends
        # Set up annotation
        a = Series(annotation)
        a.index = samples.index

        # Set up annotation min, mean, max, and colormap
        if annotation_type == 'continuous':
            samples.ix[:, 'annotation'] = normalize_pandas_object(a, '-0-').clip(-std_max, std_max)
            annotation_min = max(-std_max, samples.ix[:, 'annotation'].min())
            annotation_mean = samples.ix[:, 'annotation'].mean()
            annotation_max = min(std_max, samples.ix[:, 'annotation'].max())
            cmap = CMAP_CONTINUOUS
        else:
            samples.ix[:, 'annotation'] = annotation
            annotation_min = 0
            annotation_mean = int(samples.ix[:, 'annotation'].mean())
            annotation_max = int(samples.ix[:, 'annotation'].max())
            if annotation_type == 'categorical':
                cmap = CMAP_CATEGORICAL
            elif annotation_type == 'binary':
                cmap = CMAP_BINARY
            else:
                raise ValueError('Unknown annotation_type {}.'.format(annotation_type))
        annotation_range = annotation_max - annotation_min
        # Plot annotated samples
        for idx, s in samples.iterrows():
            if isnull(s.ix['annotation']):
                c = sample_without_annotation_markerfacecolor
            else:
                if annotation_type == 'continuous':
                    c = cmap(s.ix['annotation'])
                elif annotation_type in ('categorical', 'binary'):
                    c = cmap((s.ix['annotation'] - annotation_min) / annotation_range)
                else:
                    raise ValueError('Unknown annotation_type {}.'.format(annotation_type))
            if 'pullratio' in samples.columns:
                a = samples.ix[idx, 'pullratio']
            else:
                a = 1
            ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c,
                        alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                        clip_on=False, zorder=5)
            if a < 1:
                ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize,
                            markerfacecolor='none',
                            markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                            clip_on=False, zorder=5)
        # Plot sample legends
        ax_legend.axis('on')
        ax_legend.patch.set_visible(False)
        score, p_val = compute_association_and_pvalue(samples.ix[:, 'state'], annotation)
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
            raise ValueError('Unknown effectplot_type {}. effectplot_type = [\'violine\', \'box\'].')
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
        # Plot colorbar
        if annotation_type == 'continuous':
            cax, kw = make_axes(ax_colorbar, location='top', fraction=0.39, shrink=1, aspect=16,
                                cmap=cmap, norm=Normalize(vmin=annotation_min, vmax=annotation_max),
                                ticks=[annotation_min, annotation_mean, annotation_max])
            ColorbarBase(cax, **kw)

    else:  # Plot samples and sample legends
        # Plot samples
        for idx, s in samples.iterrows():
            c = states_color[s.ix['state']]
            if 'pullratio' in samples.columns:
                a = samples.ix[idx, 'pullratio']
            else:
                a = 1
            ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c,
                        alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                        clip_on=False, zorder=5)
            if a < 1:
                ax_map.plot(s.ix['x'], s.ix['y'], marker='o', markersize=sample_markersize,
                            markerfacecolor='none',
                            markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                            clip_on=False, zorder=5)

        # Plot sample legends
        ax_legend.axis([0, 1, 0, 1])
        for i, s in enumerate(samples.ix[:, 'state'].unique()):
            y = 1 - float(1 / (n_training_states + 1)) * (i + 1)
            c = states_color[s]
            ax_legend.plot(0.16, y, marker='o', markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)
            ax_legend.text(0.26, y, 'State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)),
                           fontsize=legend_fontsize, weight='bold', verticalalignment='center')

    if filepath:
        save_plot(filepath)
