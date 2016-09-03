from .support import print_log, SEED
from .analyze import make_onco_gps
from .visualize import FIGURE_SIZE, DPI, plot_onco_gps


def make_map(h, states, std_max=3, n_grids=128,
             informational_mds=True, mds_seed=SEED,
             kde_bandwidths_factor=1, n_influencing_components='all', sample_stretch_factor='auto',
             annotations=(), annotation_name='', annotation_type='continuous',
             title='Onco-GPS Map', title_fontsize=24, title_fontcolor='#3326C0',
             subtitle_fontsize=16, subtitle_fontcolor='#FF0039',
             component_markersize=13, component_markerfacecolor='#000726', component_markeredgewidth=1.69,
             component_markeredgecolor='#FFFFFF', component_text_position='auto', component_fontsize=16,
             delaunay_linewidth=1, delaunay_linecolor='#000000',
             n_contours=26, contour_linewidth=0.81, contour_linecolor='#5A5A5A', contour_alpha=0.92,
             background_markersize=5.55, background_mask_markersize=7, background_max_alpha=0.7,
             sample_markersize=12, sample_without_annotation_markerfacecolor='#999999',
             sample_markeredgewidth=0.81, sample_markeredgecolor='#000000',
             legend_markersize=10, legend_fontsize=11, effectplot_type='violine',
             effectplot_mean_markerfacecolor='#FFFFFF', effectplot_mean_markeredgecolor='#FF0082',
             effectplot_median_markeredgecolor='#FF0082',
             output_filepath=None, figure_size=FIGURE_SIZE, dpi=DPI):
    """
    :param h: pandas DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param states: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values
    :param n_grids: int;
    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the multidimensional scaling
    :param kde_bandwidths_factor: number; factor to multiply KDE bandwidths
    :param n_influencing_components: int; [1, n_components]; number of components influencing a sample's coordinate
    :param sample_stretch_factor: str or number; power to raise components' influence on each sample; 'auto' to automate
    :param annotations: pandas Series; (n_samples); sample annotations; will color samples based on annotations
    :param annotation_name: str;
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
    :param std_max: number; threshold to clip standardized values
    :param n_grids: int;
    :param title: str;
    :param title_fontsize: number;
    :param title_fontcolor: matplotlib color;
    :param subtitle_fontsize: number;
    :param subtitle_fontcolor: matplotlib color;
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
    :param output_filepath: str;
    :param figure_size: tuple;
    :param dpi: int;
    :return: None
    """
    cc, s, gp, gs = make_onco_gps(h, states, std_max=std_max,
                                  n_grids=n_grids,
                                  informational_mds=informational_mds,
                                  mds_seed=mds_seed,
                                  kde_bandwidths_factor=kde_bandwidths_factor,
                                  n_influencing_components=n_influencing_components,
                                  sample_stretch_factor=sample_stretch_factor)
    plot_onco_gps(cc, s, gp, gs,
                  annotations=annotations, annotation_name=annotation_name, annotation_type=annotation_type,
                  std_max=std_max,
                  title=title, title_fontsize=title_fontsize, title_fontcolor=title_fontcolor,
                  subtitle_fontsize=subtitle_fontsize, subtitle_fontcolor=subtitle_fontcolor,
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
                  output_filepath=output_filepath, figure_size=figure_size, dpi=dpi)
