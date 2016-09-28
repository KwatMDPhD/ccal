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
# TODO: refactor

from numpy import array, asarray, zeros, argmax
from pandas import DataFrame
from scipy.optimize import curve_fit
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

from .support import SEED, EPS, print_log, establish_path, write_gct, write_dictionary, nmf_and_score, \
    information_coefficient, normalize_pandas_object, consensus_cluster, exponential_function, mds
from .visualize import FIGURE_SIZE, DPI, plot_clustermap, plot_clusterings, plot_nmf_result, plot_clustering_scores, \
    plot_onco_gps

ro.conversion.py2ri = numpy2ri
mass = importr('MASS')
bcv = mass.bcv
kde2d = mass.kde2d


# ======================================================================================================================
# Define components
# ======================================================================================================================
def define_components(matrix, ks, n_clusterings=30, random_state=SEED, figure_size=FIGURE_SIZE,
                      dpi=DPI, filepath_prefix=None):
    """
    Define components.
    :param matrix:
    :param ks:
    :param n_clusterings:
    :param random_state:
    :param figure_size:
    :param dpi:
    :param filepath_prefix: str; `filepath_prefix`_nmf_k{k}_{w, h}.gct and  will be saved
    :return: dict and dict; {k: {W:w_matrix, H:h_matrix, ERROR:reconstruction_error}} and {k: cophenetic score}
    """

    # Rank normalize the input matrix by column
    matrix = normalize_pandas_object(matrix, method='rank', n_ranks=10000, axis=0)
    plot_clustermap(matrix, figure_size=figure_size, title='A Matrix', xticklabels=False, yticklabels=False)

    # NMF and score_dataframe_against_series, while saving a NMF result for each k
    nmf_results, nmf_scores = nmf_and_score(matrix=matrix, ks=ks, n_clusterings=n_clusterings,
                                            random_state=random_state)
    save_nmf_results(nmf_results, filepath_prefix)
    write_dictionary(nmf_scores, filepath_prefix + '_nmf_scores.txt', key_name='k', value_name='cophenetic_correlation')

    print_log('Plotting NMF scores ...')
    plot_clustering_scores(nmf_scores, figure_size=figure_size, filepath=filepath_prefix + '_nmf_scores.pdf', dpi=dpi)
    for k in ks:
        print_log('Plotting NMF result for k={} ...'.format(k))
        plot_nmf_result(nmf_results, k, figure_size=figure_size, filepath=filepath_prefix + '_nmf_k{}.pdf'.format(k),
                        dpi=dpi)

    return nmf_results, nmf_scores


def save_nmf_results(nmf_results, filepath_prefix):
    """
    Save `nmf_results` dictionary.
    :param nmf_results: dict; {k: {W:w, H:h, ERROR:error}}
    :param filepath_prefix: str; `filepath_prefix`_nmf_k{k}_{w, h}.gct and  will be saved
    :return: None
    """

    establish_path(filepath_prefix)
    for k, v in nmf_results.items():
        write_gct(v['W'], filepath_prefix + '_nmf_k{}_w.gct'.format(k))
        write_gct(v['H'], filepath_prefix + '_nmf_k{}_h.gct'.format(k))


# ======================================================================================================================
# Define states
# ======================================================================================================================
def define_states(h, ks, max_std=3, n_clusterings=50, figure_size=FIGURE_SIZE,
                  title='Clustering Labels', dpi=DPI, filepath_prefix=None):
    """
    Define states.
    :param h: pandas DataFrame; (n_features, m_samples)
    :param ks: iterable; list of ks used for clustering
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of clusterings for the consensus clustering
    :param figure_size: tuple,
    :param title: str; plot title
    :param dpi: int;
    :param filepath_prefix: str; filepath_prefix + '_labels.gct' and filepath_prefix + '_labels.pdf' will be saved
    :return: pandas DataFrame and Series; assignment matrix (n_ks, n_samples) and the cophenetic correlations (n_ks)
    """

    # Cluster
    labels, scores = consensus_cluster(h, ks, max_std=max_std, n_clusterings=n_clusterings)

    # Save
    if filepath_prefix:
        establish_path(filepath_prefix)
        write_gct(labels, filepath_prefix + '_labels.gct')
        write_dictionary(scores, filepath_prefix + '_clustering_scores.txt',
                         key_name='k', value_name='cophenetic_correlation')

    # Plot
    plot_clusterings(labels, title=title, filepath=filepath_prefix + '_labels.pdf')
    plot_clustering_scores(scores, figure_size=figure_size, filepath=filepath_prefix + '_clustering_scores.pdf',
                           dpi=dpi)

    return labels, scores


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

    # TODO: remove
    if isinstance(states_train[0], str):
        raise ValueError('states_train is an iterable (list) of int with values from [1, ..., <n_states_train>].')
    # TODO: remove
    if 0 in states_train:
        raise ValueError('Can\'t have \'0\' in states_train, whose values range from [1, ..., <n_states_train>].')

    cc, s, gp, gs = make_onco_gps_elements(h_train, states_train, std_max=std_max,
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
    plot_onco_gps(cc, s, gp, gs, len(set(states_train)),
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
                  contour_linewidth=contour_linewidth, contour_linecolor=contour_linecolor, contour_alpha=contour_alpha,
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


def make_onco_gps_elements(h_train, states_train, std_max=3, h_test=None, h_test_normalization='as_train',
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
             component_coordinates (n_components, [x, y]), samples (n_samples, [x, y, state, annotation]),
             grid_probabilities (n_grids, n_grids), and grid_states (n_grids, n_grids)
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
        fit_parameters = fit_columns(training_h, function_to_fit=function_to_fit, maxfev=fit_maxfev)
        print_log('Modeled columns by {}e^({}x) + {}.'.format(*fit_parameters))
        k = fit_parameters[1]
        # Linear transform
        k_normalized = (k - fit_min) / (fit_max - fit_min)
        component_pull_power = k_normalized * (pull_power_max - pull_power_min) + pull_power_min
        print_log('component_pulling_power = {:.3f}.'.format(component_pull_power))

    # Compute sample coordinates
    training_samples = get_sample_coordinates_via_pulling(component_coordinates, training_h,
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
    # Get KDE for each state using bandwidth created from all states' x & y coordinates; states starts from 1, not 0
    kdes = zeros((training_samples.ix[:, 'state'].unique().size + 1, n_grids, n_grids))
    bandwidths = asarray([bcv(asarray(training_samples.ix[:, 'x'].tolist()))[0],
                          bcv(asarray(training_samples.ix[:, 'y'].tolist()))[0]]) * kde_bandwidths_factor
    for s in sorted(training_samples.ix[:, 'state'].unique()):
        coordinates = training_samples.ix[training_samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = kde2d(asarray(coordinates.ix[:, 'x'], dtype=float), asarray(coordinates.ix[:, 'y'], dtype=float),
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
        testing_samples = get_sample_coordinates_via_pulling(component_coordinates, testing_h,
                                                             n_influencing_components=n_pulling_components,
                                                             component_pulling_power=component_pull_power)
        testing_samples.ix[:, 'state'] = states_test
        return component_coordinates, testing_samples, grid_probabilities, grid_states
    else:
        return component_coordinates, training_samples, grid_probabilities, grid_states


def fit_columns(dataframe, function_to_fit=exponential_function, maxfev=1000):
    """
    Fit columsn of `dataframe` to `function_to_fit`.
    :param dataframe: pandas DataFrame;
    :param function_to_fit: function;
    :param maxfev: int;
    :return: list; fit parameters
    """

    x = array(range(dataframe.shape[0]))
    y = asarray(dataframe.apply(sorted).apply(sum, axis=1)) / dataframe.shape[1]
    fit_parameters = curve_fit(function_to_fit, x, y, maxfev=maxfev)[0]
    return fit_parameters


def get_sample_coordinates_via_pulling(component_x_coordinates, component_x_samples,
                                       n_influencing_components='all', component_pulling_power=1):
    """
    Compute sample coordinates based on component coordinates, which pull samples.
    :param component_x_coordinates: pandas DataFrame; (n_points, [x, y])
    :param component_x_samples: pandas DataFrame; (n_points, n_samples)
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param component_pulling_power: str or number; power to raise components' influence on each sample
    :return: pandas DataFrame; (n_samples, [x, y])
    """

    sample_coordinates = DataFrame(index=component_x_samples.columns, columns=['x', 'y'])
    for sample in sample_coordinates.index:
        c = component_x_samples.ix[:, sample]
        if n_influencing_components == 'all':
            n_influencing_components = component_x_samples.shape[0]
        c = c.mask(c < c.sort_values().tolist()[-n_influencing_components], other=0)
        x = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'x']) / sum(c ** component_pulling_power)
        y = sum(c ** component_pulling_power * component_x_coordinates.ix[:, 'y']) / sum(c ** component_pulling_power)
        sample_coordinates.ix[sample, ['x', 'y']] = x, y
    return sample_coordinates
