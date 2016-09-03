from .support import print_log, SEED
from .visualize import FIGURE_SIZE, DPI


def make_onco_gps(h, states, std_max=3, n_grids=128,
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
    :param std_max: number; threshold to clip standardized values
    :param annotation_type: str; {'continuous', 'categorical', 'binary'}
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
