"""
Computational Cancer Analysis Library

Authors:
Pablo Tamayo
ptamayo@ucsd.edu
Computational Cancer Analysis Laboratory, UCSD Cancer Center

Huwate (Kwat) Yeerna (Medetgul-Ernar)
kwat.medetgul.ernar@gmail.com
Computational Cancer Analysis Laboratory, UCSD Cancer Center

James Jensen
jdjensen@eng.ucsd.edu
Laboratory of Jill Mesirov
"""
from .support import SEED, print_log, write_dictionary, save_nmf_results
from .analyze import nmf_and_score, normalize_pandas_object, consensus_cluster, exponential_function, \
    make_onco_gps_elements
from .visualize import FIGURE_SIZE, DPI, plot_clustermap, plot_clusterings, plot_nmf_result, plot_clustering_scores, \
    plot_onco_gps


def preprocess_matrix():
    return None


def define_components(matrix, ks, filepath_prefix, n_clusterings=30, random_state=SEED, figure_size=FIGURE_SIZE,
                      dpi=DPI):
    """
    Define components.
    :param matrix:
    :param ks:
    :param n_clusterings:
    :param random_state:
    :param filepath_prefix:
    :param figure_size:
    :param dpi:
    :param filepath_prefix: str; `filepath_prefix`_nmf_k{k}_{w, h}.gct and  will be saved
    :return:
    """
    # Rank normalize the input matrix by column
    matrix = normalize_pandas_object(matrix, method='rank', n_ranks=10000, axis=0)
    plot_clustermap(matrix, figure_size=figure_size, title='A Matrix', xticklabels=False, yticklabels=False)

    # NMF and score, while saving a NMF result for each k
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


def define_states(h, ks, filepath_prefix, max_std=3, n_clusterings=50, figure_size=FIGURE_SIZE,
                  title='Clustering Labels', dpi=DPI):
    """
    Define states.
    :param h:
    :param ks:
    :param max_std:
    :param n_clusterings:
    :param filepath_prefix:
    :param figure_size:
    :param dpi:
    :return:
    """
    labels, scores = consensus_cluster(h, ks, max_std=max_std, n_clusterings=n_clusterings,
                                       filepath_prefix=filepath_prefix)
    plot_clusterings(labels, title=title, filepath=filepath_prefix + '_labels.pdf')
    plot_clustering_scores(scores, figure_size=figure_size, filepath=filepath_prefix + '_clustering_scores.pdf',
                           dpi=dpi)
    return labels, scores


# TODO: Simplify
def make_map(h_train, states_train, std_max=3, h_test=None, h_test_normalization='clip_and_0-1', states_test=None,
             informational_mds=True, mds_seed=SEED, mds_n_init=1000, mds_max_iter=1000,
             function_to_fit=exponential_function, fit_maxfev=1000,
             fit_min=0, fit_max=2, pull_power_min=1, pull_power_max=5,
             n_pulling_components='all', component_pull_power='auto', n_pullratio_components=0, pullratio_factor=5,
             n_grids=128, kde_bandwidths_factor=1,
             annotations=(), annotation_name='', annotation_type='continuous',
             title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
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
             filepath=None, figure_size=FIGURE_SIZE, dpi=DPI):
    """
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
    :param annotations: pandas Series; (n_samples); sample annotations; will color samples based on annotations
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
    :param figure_size: tuple;
    :param dpi: int;
    :return: None
    """
    if isinstance(states_train[0], str):
        raise ValueError('states_train is an iterable (list) of int with values from [1, ..., <n_states_train>].')
    if 0 in states_train:
        raise ValueError('Can\'t have \'0\' in states_train, whose values range from [1, ..., <n_states_train>].')

    cc, s, gp, gs = make_onco_gps_elements(h_train, states_train, std_max=std_max,
                                           h_test=h_test, h_test_normalization=h_test_normalization,
                                           states_test=states_test,
                                           informational_mds=informational_mds, mds_seed=mds_seed,
                                           mds_n_init=mds_n_init, mds_max_iter=mds_max_iter,
                                           function_to_fit=function_to_fit,
                                           fit_maxfev=fit_maxfev, fit_min=fit_min, fit_max=fit_max,
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
                  filepath=filepath, figure_size=figure_size, dpi=dpi)
