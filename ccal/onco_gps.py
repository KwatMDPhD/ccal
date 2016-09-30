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
from numpy import asarray, zeros, argmax, linspace
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
from .support import EPS, print_log, establish_filepath, write_gct, write_dictionary, fit_matrix, nmf_and_score, \
    information_coefficient, normalize_pandas_object, consensus_cluster, exponential_function, mds, \
    compute_score_and_pvalue, FIGURE_SIZE, DPI, CMAP_CONTINUOUS, CMAP_CATEGORICAL, CMAP_BINARY, save_plot, \
    plot_clustermap, plot_clustering_per_k, plot_nmf_result, plot_x_vs_y

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


# ======================================================================================================================
# Define components
# ======================================================================================================================
def define_components(matrix, ks, n_clusterings=30, random_state=SEED,
                      figure_size=FIGURE_SIZE, dpi=DPI, directory_path=None):
    """
    NMF matrix into W and H matrices using k from ks and calculate cophenetic correlation by consensus clustering.
    :param matrix: pandas DataFrame;
    :param ks: iterable; iterable of int k used for NMF
    :param n_clusterings: int; number of NMF for consensus clustering
    :param random_state: int;
    :param figure_size: tuple;
    :param dpi: int;
    :param directory_path: str; directory where nmf_k{k}_{w, h}.gct and nmf_scores.pdf will be saved
    :return: dict and dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}} and {k: cophenetic correlation}
    """

    # Rank normalize the input matrix by column
    matrix = normalize_pandas_object(matrix, method='rank', n_ranks=10000, axis=0)
    plot_clustermap(matrix, figure_size=figure_size,
                    title='A Matrix', xlabel='Sample', ylabel='Gene', xticklabels=False, yticklabels=False)

    # NMF and score, while saving a NMF result for each k
    nmf_results, nmf_scores = nmf_and_score(matrix=matrix, ks=ks, n_clusterings=n_clusterings,
                                            random_state=random_state)

    # Make nmf directory
    directory_path = join(directory_path, 'nmf/')
    establish_filepath(directory_path)

    # Save NMF scores @ nmf/scores{.pdf, .gct}
    print_log('Saving and plotting NMF scores ...')
    write_dictionary(nmf_scores, join(directory_path, 'scores.txt'), key_name='k', value_name='cophenetic_correlation')
    plot_x_vs_y(sorted(nmf_scores.keys()), [nmf_scores[k] for k in sorted(nmf_scores.keys())],
                figure_size=figure_size, title='NMF Cophenetic Score vs. k', xlabel='k', ylabel='NMF Cophenetic Score',
                dpi=dpi, filepath=join(directory_path, 'scores.pdf'))

    # Save NMF results @ nmf/matrices/nmf_k{...}_{w, h}.gct
    print_log('Saving and plotting NMF results ...')
    _save_nmf_results(nmf_results, join(directory_path, 'matrices', ''))

    # Save NMF figures @ nmf/figures/nmf_k{...}.pdf
    for k in ks:
        print_log('\tPlotting k={} ...'.format(k))
        plot_nmf_result(nmf_results, k, figure_size=figure_size, dpi=dpi,
                        filepath=join(directory_path, 'figures', 'nmf_k{}.pdf'.format(k)))

    return nmf_results, nmf_scores


def _save_nmf_results(nmf_results, filepath_prefix):
    """
    Save nmf_results.
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param filepath_prefix: str; filepath_prefix_nmf_k{k}_{w, h}.gct and  will be saved
    :return: None
    """

    establish_filepath(filepath_prefix)
    for k, v in nmf_results.items():
        write_gct(v['W'], filepath_prefix + 'nmf_k{}_w.gct'.format(k))
        write_gct(v['H'], filepath_prefix + 'nmf_k{}_h.gct'.format(k))


# ======================================================================================================================
# Define states
# ======================================================================================================================
def define_states(h, ks, max_std=3, n_clusterings=50,
                  figure_size=FIGURE_SIZE, title='Clustering Labels', dpi=DPI, filepath_prefix=None):
    """
    Cluster samples using k from ks and calculate cophenetic correlation by consensus clustering.
    :param h: pandas DataFrame; (n_features, m_samples); H matrix from NMF
    :param ks: iterable; iterable of int k used for NMF
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for consensus clustering
    :param figure_size: tuple;
    :param title: str; plot title
    :param dpi: int;
    :param filepath_prefix: str; filepath_prefix_labels.gct and filepath_prefix_labels.pdf will be saved
    :return: pandas DataFrame and Series; assignment matrix (n_ks, n_samples) and the cophenetic correlations (n_ks)
    """

    # Cluster
    clusterings, scores = consensus_cluster(h, ks, max_std=max_std, n_clusterings=n_clusterings)

    # Save
    if filepath_prefix:
        establish_filepath(filepath_prefix)
        write_gct(clusterings, filepath_prefix + '_labels.gct')
        write_dictionary(scores, filepath_prefix + '_clustering_scores.txt',
                         key_name='k', value_name='cophenetic_correlation')

    # Plot
    plot_clustering_per_k(clusterings, title=title, filepath=filepath_prefix + '_labels.pdf')
    plot_x_vs_y(sorted(scores.keys()), [scores[k] for k in sorted(scores.keys())],
                figure_size=figure_size, title='Consensus Clustering Cophenetic Score vs. k',
                xlabel='k', ylabel='Consensus Clustering Cophenetic Score',
                dpi=dpi, filepath=filepath_prefix + '_clustering_scores.pdf')

    return clusterings, scores


# ======================================================================================================================
# Make Onco-GPS map
# ======================================================================================================================
def make_map(h_train, states_train, std_max=3, h_test=None, h_test_normalization='clip_and_0-1', states_test=None,
             informational_mds=True, mds_seed=SEED,
             fit_min=0, fit_max=2, pull_power_min=1, pull_power_max=5,
             n_pulling_components='all', component_pull_power='auto', n_pullratio_components=0, pullratio_factor=5,
             n_grids=128, kde_bandwidths_factor=1,
             annotations=(), annotation_name='', annotation_type='continuous',
             figure_size=FIGURE_SIZE, title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
             subtitle_fontsize=16, subtitle_fontcolor='#FF0039',
             colors=None, component_markersize=13, component_markerfacecolor='#000726', component_markeredgewidth=1.69,
             component_markeredgecolor='#FFFFFF', component_text_position='auto', component_fontsize=16,
             delaunay_linewidth=1, delaunay_linecolor='#000000',
             n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
             background_markersize=5.55, background_mask_markersize=7, background_max_alpha=0.9,
             sample_markersize=12, sample_without_annotation_markerfacecolor='#999999',
             sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
             legend_markersize=10, legend_fontsize=11, effectplot_type='violine',
             effectplot_mean_markerfacecolor='#FFFFFF', effectplot_mean_markeredgecolor='#FF0082',
             effectplot_median_markeredgecolor='#FF0082',
             dpi=DPI, filepath=None):
    """
    :param h_train: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param states_train: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values
    :param h_test: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param h_test_normalization: str or None; {'as_train', 'clip_and_0-1', None}
    :param states_test: iterable of int; (n_samples); sample states
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param fit_min: number;
    :param fit_max: number;
    :param pull_power_min: number;
    :param pull_power_max: number;
    :param n_pulling_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_pull_power: str or number; power to raise components' influence on each sample
    :param n_pullratio_components: number; number if int; percentile if float & < 1
    :param pullratio_factor: number;
    :param n_grids: int;
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :param annotations: pandas Series; (n_samples); sample annotations; will color samples based on annotations
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param figure_size: tuple;
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
    :param dpi: int;
    :param filepath: str;
    :return: None
    """

    cc, s, gp, gs = _make_onco_gps_elements(h_train, states_train, std_max=std_max,
                                            h_test=h_test, h_test_normalization=h_test_normalization,
                                            states_test=states_test,
                                            informational_mds=informational_mds, mds_seed=mds_seed,
                                            fit_min=fit_min, fit_max=fit_max,
                                            pull_power_min=pull_power_min, pull_power_max=pull_power_max,
                                            n_pulling_components=n_pulling_components,
                                            component_pull_power=component_pull_power,
                                            n_pullratio_components=n_pullratio_components,
                                            pullratio_factor=pullratio_factor,
                                            n_grids=n_grids, kde_bandwidths_factor=kde_bandwidths_factor)
    _plot_onco_gps(cc, s, gp, gs, len(set(states_train)),
                   annotations=annotations, annotation_name=annotation_name, annotation_type=annotation_type,
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
                   figure_size=figure_size, dpi=dpi, filepath=filepath)


def _make_onco_gps_elements(h_train, states_train, std_max=3, h_test=None, h_test_normalization='as_train',
                            states_test=None,
                            informational_mds=True, mds_seed=SEED, mds_n_init=1000, mds_max_iter=1000,
                            function_to_fit=exponential_function, fit_maxfev=1000,
                            fit_min=0, fit_max=2, pull_power_min=1, pull_power_max=3,
                            n_pulling_components='all', component_pull_power='auto', n_pullratio_components=0,
                            pullratio_factor=5,
                            n_grids=128, kde_bandwidths_factor=1):
    """
    Compute component and sample coordinates. And compute grid probabilities and states.
    :param h_train: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param states_train: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values
    :param h_test: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param h_test_normalization: str or None; {'as_train', 'clip_and_0-1', None}
    :param states_test: iterable of int; (n_samples); sample states
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param mds_n_init: int;
    :param mds_max_iter: int;
    :param function_to_fit: function;
    :param fit_maxfev: int;
    :param fit_min: number;
    :param fit_max: number;
    :param pull_power_min: number;
    :param pull_power_max: number;
    :param n_pulling_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_pull_power: str or number; power to raise components' influence on each sample
    :param n_pullratio_components: number; number if int; percentile if float & < 1
    :param pullratio_factor: number;
    :param n_grids: int;
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :return: pandas DataFrame, DataFrame, numpy array, and numpy array;
             component_coordinates (n_components, [_nmf_and_score, y]),
             samples (n_samples, [_nmf_and_score, y, state, annotation]),
             grid_probabilities (n_grids, n_grids),
             and grid_states (n_grids, n_grids)
    """

    print_log('Making Onco-GPS with {} components, {} samples, and {} states {} ...'.format(*h_train.shape,
                                                                                            len(set(states_train)),
                                                                                            set(states_train)))

    # clip and 0-1 normalize the data
    training_h = normalize_pandas_object(normalize_pandas_object(h_train, method='-0-', axis=1).clip(-std_max, std_max),
                                         method='0-1', axis=1)

    # Compute component coordinates
    if informational_mds:
        distance_function = information_coefficient
    else:
        distance_function = None
    component_coordinates = mds(training_h, distance_function=distance_function,
                                mds_seed=mds_seed, n_init=mds_n_init, max_iter=mds_max_iter, standardize=True)

    # Compute component pulling power
    if component_pull_power == 'auto':
        fit_parameters = fit_matrix(training_h, function_to_fit, maxfev=fit_maxfev)
        print_log('Modeled columns by {}e^({}x) + {}.'.format(*fit_parameters))
        k = fit_parameters[1]
        # Linear transform
        k_normalized = (k - fit_min) / (fit_max - fit_min)
        component_pull_power = k_normalized * (pull_power_max - pull_power_min) + pull_power_min
        print_log('component_pulling_power = {:.3f}.'.format(component_pull_power))

    # Compute sample coordinates
    training_samples = _get_sample_coordinates_via_pulling(component_coordinates, training_h,
                                                           n_influencing_components=n_pulling_components,
                                                           component_pulling_power=component_pull_power)

    # Compute pulling ratios
    ratios = zeros(training_h.shape[1])
    if 0 < n_pullratio_components:
        if n_pullratio_components < 1:
            n_pullratio_components = training_h.shape[0] * n_pullratio_components
        for i, (c_idx, c) in enumerate(training_h.iteritems()):
            c_sorted = c.sort_values(ascending=False)
            ratio = float(
                c_sorted[:n_pullratio_components].sum() / max(c_sorted[n_pullratio_components:].sum(), EPS)) * c.sum()
            ratios[i] = ratio
        normalized_ratios = (ratios - ratios.min()) / (ratios.max() - ratios.min()) * pullratio_factor
        training_samples.ix[:, 'pullratio'] = normalized_ratios.clip(0, 1)

    # Load sample states
    training_samples.ix[:, 'state'] = states_train

    # Compute grid probabilities and states
    grid_probabilities = zeros((n_grids, n_grids))
    grid_states = zeros((n_grids, n_grids), dtype=int)
    # Get KDE for each state using bandwidth created from all states' scores & y coordinates; states have 1 based-index
    kdes = zeros((training_samples.ix[:, 'state'].unique().size + 1, n_grids, n_grids))
    bandwidths = asarray([bcv(asarray(training_samples.ix[:, '_nmf_and_score'].tolist()))[0],
                          bcv(asarray(training_samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor
    for s in sorted(training_samples.ix[:, 'state'].unique()):
        coordinates = training_samples.ix[training_samples.ix[:, 'state'] == s, ['_nmf_and_score', 'y']]
        kde = kde2d(asarray(coordinates.ix[:, '_nmf_and_score'], dtype=float),
                    asarray(coordinates.ix[:, 'y'], dtype=float),
                    bandwidths, n=asarray([n_grids]), lims=asarray([0, 1, 0, 1]))
        kdes[s] = asarray(kde[2])
    # Assign the best KDE probability and state for each grid
    for i in range(n_grids):
        for j in range(n_grids):
            grid_probabilities[i, j] = max(kdes[:, j, i])
            grid_states[i, j] = argmax(kdes[:, i, j])

    if isinstance(h_test, DataFrame):
        print_log('Focusing on samples from testing H matrix ...')
        # Normalize testing H
        if h_test_normalization == 'as_train':
            testing_h = h_test
            for r_idx, r in h_train.iterrows():
                if r.std() == 0:
                    testing_h.ix[r_idx, :] = testing_h.ix[r_idx, :] / r.size()
                else:
                    testing_h.ix[r_idx, :] = (testing_h.ix[r_idx, :] - r.mean()) / r.std()
        elif h_test_normalization == 'clip_and_0-1':
            testing_h = normalize_pandas_object(
                normalize_pandas_object(h_test, method='-0-', axis=1).clip(-std_max, std_max),
                method='0-1', axis=1)
        elif not h_test_normalization:
            testing_h = h_test
        else:
            raise ValueError('Unknown normalization method for testing H {}.'.format(h_test_normalization))

        # Compute testing-sample coordinates
        testing_samples = _get_sample_coordinates_via_pulling(component_coordinates, testing_h,
                                                              n_influencing_components=n_pulling_components,
                                                              component_pulling_power=component_pull_power)
        testing_samples.ix[:, 'state'] = states_test
        return component_coordinates, testing_samples, grid_probabilities, grid_states
    else:
        return component_coordinates, training_samples, grid_probabilities, grid_states


def _get_sample_coordinates_via_pulling(component_x_coordinates, component_x_samples,
                                        n_influencing_components='all', component_pulling_power=1):
    """
    Compute sample coordinates based on component coordinates, which pull samples.
    :param component_x_coordinates: pandas DataFrame; (n_points, [_nmf_and_score, y])
    :param component_x_samples: pandas DataFrame; (n_points, n_samples)
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_pulling_power: str or number; power to raise components' influence on each sample
    :return: pandas DataFrame; (n_samples, [_nmf_and_score, y])
    """

    sample_coordinates = DataFrame(index=component_x_samples.columns, columns=['_nmf_and_score', 'y'])
    for sample in sample_coordinates.index:
        c = component_x_samples.ix[:, sample]
        if n_influencing_components == 'all':
            n_influencing_components = component_x_samples.shape[0]
        c = c.mask(c < c.sort_values().tolist()[-n_influencing_components], other=0)
        x = sum(c ** component_pulling_power * component_x_coordinates.ix[:, '_nmf_and_score']) / sum(
            c ** component_pulling_power)
        y = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'y']) / sum(c ** component_pulling_power)
        sample_coordinates.ix[sample, ['_nmf_and_score', 'y']] = x, y
    return sample_coordinates


def _plot_onco_gps(component_coordinates, samples, grid_probabilities, grid_states, n_states_train,
                   annotations=(), annotation_name='', annotation_type='continuous', std_max=3,
                   figure_size=FIGURE_SIZE, title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
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
                   dpi=DPI, filepath=None):
    """
    Plot Onco-GPS map.
    :param component_coordinates: pandas DataFrame; (n_components, [_nmf_and_score, y]);
        output from _make_onco_gps_elements
    :param samples: pandas DataFrame; (n_samples, [_nmf_and_score, y, state])
    :param grid_probabilities: numpy 2D array; (n_grids, n_grids)
    :param grid_states: numpy 2D array; (n_grids, n_grids)
    :param n_states_train: int; number of states used to create Onco-GPS
    :param annotations: pandas Series; (n_samples); sample annotations; will color samples based on annotations
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param std_max: number; threshold to clip standardized values
    :param figure_size: tuple;
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
    :param dpi: int;
    :param filepath: str;
    :return: None
    """

    x_grids = linspace(0, 1, grid_probabilities.shape[0])
    y_grids = linspace(0, 1, grid_probabilities.shape[1])

    # Set up figure and axes
    plt.figure(figsize=figure_size)
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

    # Plot title
    ax_title.text(0, 0.9, title, fontsize=title_fontsize, color=title_fontcolor, weight='bold')
    ax_title.text(0, 0.39,
                  '{} samples, {} components, and {} states'.format(samples.shape[0], component_coordinates.shape[0],
                                                                    n_states_train),
                  fontsize=subtitle_fontsize, color=subtitle_fontcolor, weight='bold')

    # Plot components and their labels
    ax_map.plot(component_coordinates.ix[:, '_nmf_and_score'], component_coordinates.ix[:, 'y'], marker='D',
                linestyle='',
                markersize=component_markersize, markerfacecolor=component_markerfacecolor,
                markeredgewidth=component_markeredgewidth, markeredgecolor=component_markeredgecolor, clip_on=False,
                aa=True, zorder=6)
    # Compute convexhull
    convexhull = ConvexHull(component_coordinates)
    convexhull_region = Path(convexhull.points[convexhull.vertices])
    # Put labels on top or bottom of the component markers
    component_text_verticalshift = -0.03
    for i in component_coordinates.index:
        if component_text_position == 'auto':

            if convexhull_region.contains_point((component_coordinates.ix[i, '_nmf_and_score'],
                                                 component_coordinates.ix[i, 'y'] + component_text_verticalshift)):
                component_text_verticalshift *= -1
        elif component_text_position == 'top':
            component_text_verticalshift *= -1
        elif component_text_position == 'bottom':
            pass
        x, y = component_coordinates.ix[i, '_nmf_and_score'], component_coordinates.ix[
            i, 'y'] + component_text_verticalshift

        ax_map.text(x, y, i,
                    fontsize=component_fontsize, color=component_markerfacecolor, weight='bold',
                    horizontalalignment='center', verticalalignment='center', zorder=6)

    # Plot Delaunay triangulation
    delaunay = Delaunay(component_coordinates)
    ax_map.triplot(delaunay.points[:, 0], delaunay.points[:, 1], delaunay.simplices.copy(),
                   linewidth=delaunay_linewidth, color=delaunay_linecolor, aa=True, zorder=4)

    # Plot contours
    if n_contours > 0:
        ax_map.contour(x_grids, y_grids, grid_probabilities, n_contours, corner_mask=True,
                       linewidths=contour_linewidth, colors=contour_linecolor, alpha=contour_alpha, aa=True, zorder=2)

    # Assign colors to states
    if colors:
        if not (isinstance(colors, ListedColormap) and isinstance(colors, LinearSegmentedColormap)):
            colors = ListedColormap(colors)
    states_color = {}
    for s in range(1, n_states_train + 1):
        if colors:
            states_color[s] = colors(s)
        else:
            states_color[s] = CMAP_CATEGORICAL(int(s / n_states_train * CMAP_CATEGORICAL.N))

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

    if any(annotations):  # Plot samples, annotations, sample legends, and annotation legends
        # Set up annotations
        a = Series(annotations)
        a.index = samples.index
        # Set up annotation min, mean, max, and colormap.
        if annotation_type == 'continuous':
            samples.ix[:, 'annotation'] = normalize_pandas_object(a, method='-0-').clip(-std_max, std_max)
            annotation_min = max(-std_max, samples.ix[:, 'annotation'].min())
            annotation_mean = samples.ix[:, 'annotation'].mean()
            annotation_max = min(std_max, samples.ix[:, 'annotation'].max())
            cmap = CMAP_CONTINUOUS
        else:
            samples.ix[:, 'annotation'] = annotations
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
            ax_map.plot(s.ix['_nmf_and_score'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c,
                        alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                        zorder=5)
            if a < 1:
                ax_map.plot(s.ix['_nmf_and_score'], s.ix['y'], marker='o', markersize=sample_markersize,
                            markerfacecolor='none',
                            markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                            zorder=5)
        # Plot sample legends
        ax_legend.axis('on')
        ax_legend.patch.set_visible(False)
        score, p_val = compute_score_and_pvalue(samples.ix[:, 'state'], annotations)
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
        # Set up _nmf_and_score label, ticks, and lines
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
            ['State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)) for s in range(1, n_states_train + 1)],
            fontsize=legend_fontsize, weight='bold')
        ax_legend.yaxis.tick_right()
        # Plot sample markers
        l, r = ax_legend.axis()[:2]
        x = l - float((r - l) / 5)
        for i, s in enumerate(range(1, n_states_train + 1)):
            c = states_color[s]
            ax_legend.plot(x, i, marker='o', markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)
        # Plot colorbar
        if annotation_type == 'continuous':
            cax, kw = make_axes(ax_colorbar, location='top', fraction=0.39, shrink=1, aspect=16,
                                cmap=cmap, norm=Normalize(vmin=annotation_min, vmax=annotation_max),
                                ticks=[annotation_min, annotation_mean, annotation_max])
            ColorbarBase(cax, **kw)

    else:  # Plot samples and sample legends
        ax_legend.axis([0, 1, 0, 1])
        # Plot samples
        for idx, s in samples.iterrows():
            c = states_color[s.ix['state']]
            if 'pullratio' in samples.columns:
                a = samples.ix[idx, 'pullratio']
            else:
                a = 1
            ax_map.plot(s.ix['_nmf_and_score'], s.ix['y'], marker='o', markersize=sample_markersize, markerfacecolor=c,
                        alpha=a,
                        markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                        zorder=5)
            if a < 1:
                ax_map.plot(s.ix['_nmf_and_score'], s.ix['y'], marker='o', markersize=sample_markersize,
                            markerfacecolor='none',
                            markeredgewidth=sample_markeredgewidth, markeredgecolor=sample_markeredgecolor, aa=True,
                            zorder=5)
        # Plot sample legends
        for i, s in enumerate(range(1, n_states_train + 1)):
            y = 1 - float(1 / (n_states_train + 1)) * (i + 1)
            c = states_color[s]
            ax_legend.plot(0.16, y, marker='o', markersize=legend_markersize, markerfacecolor=c, aa=True, clip_on=False)
            ax_legend.text(0.26, y, 'State {} (n={})'.format(s, sum(samples.ix[:, 'state'] == s)),
                           fontsize=legend_fontsize, weight='bold', verticalalignment='center')

    if filepath:
        save_plot(filepath, dpi=dpi)
