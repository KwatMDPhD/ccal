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

from colorsys import hsv_to_rgb, rgb_to_hsv
from os.path import join

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.pyplot import figure, savefig, subplot
from numpy import (asarray, empty, linspace, ma, nansum, ndarray, ones, sqrt,
                   zeros, zeros_like)
from pandas import DataFrame, Series, isnull
from scipy.spatial import ConvexHull, Delaunay
from sklearn.svm import SVR

from .. import RANDOM_SEED
from ..machine_learning.classify import classify
from ..machine_learning.cluster import (hierarchical_consensus_cluster,
                                        nmf_consensus_cluster)
from ..machine_learning.fit import fit_matrix
from ..machine_learning.multidimentional_scale import mds
from ..machine_learning.score import compute_association_and_pvalue
from ..machine_learning.solve import solve_matrix_linear_equation
from ..mathematics.equation import define_exponential_function
from ..mathematics.information import EPS, bcv, information_coefficient, kde2d
from ..support.d2 import (drop_na_2d, drop_uniform_slice_from_dataframe,
                          normalize_2d_or_1d)
from ..support.file import establish_filepath, load_gct, read_gct, write_gct
from ..support.log import print_log
from ..support.plot import (CMAP_BINARY, CMAP_CATEGORICAL_PAIRED,
                            CMAP_CONTINUOUS, DPI, FIGURE_SIZE,
                            assign_colors_to_states, decorate, plot_heatmap,
                            plot_nmf, plot_points, save_plot)


# ==============================================================================
# Define components
# ==============================================================================
def define_components(a_matrix,
                      ks,
                      directory_path,
                      file_mark='',
                      how_to_drop_na_in_a_matrix='all',
                      a_matrix_normalization_method='standardize',
                      std_max=3,
                      n_jobs=1,
                      n_clusterings=40,
                      algorithm='Alternating Least Squares',
                      random_seed=RANDOM_SEED):
    """
    NMF-consensus cluster samples, compute cophenetic-correlation
    coefficients, and save 1 NMF decomposition for each k.

    :param a_matrix: DataFrame or str; (n_rows, n_columns), A matrix,
    or filepath to a GCT file
    :param ks: iterable or int; iterable of int k used for NMF
    :param directory_path: str; directory path where nmf_cc/nmf.pdf,
    nmf_cc/nmf_k{k}_{w, h}.gct will be saved
    :param file_mark: str;
    :param how_to_drop_na_in_a_matrix: str; {'all', 'any'}
    :param a_matrix_normalization_method: str; {'standardize', 'rank'}
    :param std_max: number;
    :param n_jobs: int;
    :param n_clusterings: int; number of NMF for consensus clustering
    :param algorithm: str; 'Alternating Least Squares' or 'Lee & Seung'
    :param random_seed: int;
    :return: dict; {k: {
                        w: W matrix (n_rows, k),
                        h: H matrix (k, n_columns),
                        e: Reconstruction Error,
                        ccc: Cophenetic Correlation Coefficient
                        }
                    }
    """

    # Load A matrix
    a_matrix = load_gct(a_matrix)

    # Drop na rows & columns
    a_matrix = drop_na_2d(a_matrix, how=how_to_drop_na_in_a_matrix)

    # Normaliza A matrix
    a_matrix = normalize_a_matrix(
        a_matrix,
        a_matrix_normalization_method=a_matrix_normalization_method,
        std_max=std_max)

    # NMF-consensus cluster (while saving 1 NMF result per k)
    nmfs = nmf_consensus_cluster(
        a_matrix,
        ks,
        n_jobs=n_jobs,
        n_clusterings=n_clusterings,
        algorithm=algorithm,
        random_seed=random_seed)
    # Name NMF components
    for k, nmf in nmfs.items():
        nmf['w'].columns = ['C{}'.format(c) for c in range(1, k + 1)]
        nmf['h'].index = ['C{}'.format(c) for c in range(1, k + 1)]

    print_log('Saving & plotting ...')
    directory_path = join(directory_path, 'nmf_cc{}/'.format(file_mark))
    establish_filepath(directory_path)
    with PdfPages(join(directory_path, 'nmf.pdf')) as pdf:
        plot_points(
            sorted(nmfs.keys()), [nmfs[k]['ccc'] for k in sorted(nmfs.keys())],
            title='NMF-CC Cophenetic-Correlation Coefficient vs. K',
            xlabel='K',
            ylabel='NMF-CC Cophenetic-Correlation Coefficient')
        savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

        for k, nmf_ in nmfs.items():
            print_log('\tK={} ...'.format(k))
            write_gct(nmf_['w'],
                      join(directory_path, 'nmf_k{}_w.gct'.format(k)))
            write_gct(nmf_['h'],
                      join(directory_path, 'nmf_k{}_h.gct'.format(k)))

            plot_nmf(nmfs, k, pdf=pdf)

    return nmfs


def get_w_or_h_matrix(nmf_results, k, w_or_h):
    """
    Get W or H matrix from nmf_results.
    :param nmf_results: dict;
    :param k: int;
    :param w_or_h: str; 'w', 'W', 'H', or 'h'
    :return: DataFrame; W or H matrix for this k
    """

    w_or_h = w_or_h.strip()
    if w_or_h.lower() not in ('w', 'h'):
        raise TypeError('w_or_h must be one of \'w\' or \'h\'.')

    return nmf_results[k][w_or_h.lower()]


def solve_for_components(w_matrix,
                         a_matrix,
                         w_matrix_normalization_method='sum',
                         how_to_drop_na_in_a_matrix='all',
                         a_matrix_normalization_method='standardize',
                         std_max=3,
                         method='nnls',
                         filepath_prefix=None):
    """
    Get H matrix of a_matrix in the space of w_matrix by solving W * H = A
    for H.
    :param w_matrix: str or DataFrame; (n_rows, k)
    :param a_matrix: str or DataFrame; (n_rows, n_columns)
    :param w_matrix_normalization_method: str; {'sum'}
    :param how_to_drop_na_in_a_matrix: str; {'all', 'any'}
    :param a_matrix_normalization_method: str; {'standardize', 'rank'}
    :param std_max: number;
    :param method: str; {'nnls', 'pinv'}
    :param filepath_prefix: str; filepath_prefix_solved_nmf_h_k{}.{gct,
    pdf} will be saved
    :return: DataFrame; (k, n_columns)
    """

    # Load A and W matrices
    w_matrix = load_gct(w_matrix)
    a_matrix = load_gct(a_matrix)

    # Drop na rows & columns
    a_matrix = drop_na_2d(a_matrix, how=how_to_drop_na_in_a_matrix)

    # Keep only indices shared by both
    common_indices = set(w_matrix.index) & set(a_matrix.index)
    w_matrix = w_matrix.ix[common_indices, :]
    a_matrix = a_matrix.ix[common_indices, :]
    print_log('{} W-matrix indices.'.format(w_matrix.shape[0]))
    print_log('{} A-matrix indices.'.format(a_matrix.shape[0]))
    print_log('{} common indices.'.format(len(common_indices)))

    # Normalize W matrix
    if w_matrix_normalization_method == 'sum':
        # Sum normalize W matrix by column
        w_matrix = w_matrix.apply(lambda c: c / c.sum())
    else:
        print_log('Not normalizing W matrix ...')

    # Normaliza A matrix
    a_matrix = normalize_a_matrix(
        a_matrix,
        a_matrix_normalization_method=a_matrix_normalization_method,
        std_max=std_max)

    # Solve W * H = A
    print_log('Solving for components: W({}x{}) * H = A({}x{}) ...'.format(
        *w_matrix.shape, *a_matrix.shape))
    h_matrix = solve_matrix_linear_equation(w_matrix, a_matrix, method=method)

    if filepath_prefix:  # Save H matrix
        write_gct(h_matrix, filepath_prefix +
                  '_solved_nmf_h_k{}.gct'.format(h_matrix.shape[0]))
        plot_filepath = filepath_prefix + '_solved_nmf_h_k{}.pdf'.format(
            h_matrix.shape[0])
    else:
        plot_filepath = None

    plot_nmf(w_matrix=w_matrix, h_matrix=h_matrix, filepath=plot_filepath)

    return h_matrix


def normalize_a_matrix(a_matrix,
                       a_matrix_normalization_method='standardize',
                       std_max=3):
    """

    :param a_matrix:
    :param a_matrix_normalization_method:
    :param std_max:
    :return:
    """

    # Normaliza A matrix columns
    if a_matrix_normalization_method == 'standardize':
        a_matrix = normalize_2d_or_1d(a_matrix, method='-0-', axis=0)
        a_matrix = a_matrix.clip(lower=-std_max, upper=std_max)
        a_matrix += std_max
    elif a_matrix_normalization_method == 'rank':
        a_matrix = normalize_2d_or_1d(a_matrix, 'rank', axis=0)
    else:
        print_log('Not normalizing A matrix columns ...')

    # Plot after normalization
    plot_heatmap(
        a_matrix,
        title='Matrix to be Decomposed ({}-normalized columns)'.format(
            a_matrix_normalization_method),
        xlabel='Sample',
        ylabel='Feature',
        xticklabels=False,
        yticklabels=False,
        cluster=True)

    return a_matrix


# ==============================================================================
# Define states
# ==============================================================================
def define_states(matrix,
                  ks,
                  directory_path,
                  file_mark='',
                  n_jobs=1,
                  distance_matrix=None,
                  max_std=3,
                  n_clusterings=40,
                  random_seed=RANDOM_SEED):
    """
    Hierarchical-consensus cluster samples (matrix columns) and compute
    cophenetic correlation coefficients.
    :param matrix: DataFrame or str; (n_rows, n_columns); filepath to a .gct
    :param ks: iterable; iterable of int k used for hierarchical clustering
    :param directory_path: str; directory path where
    clusterings/distance_matrix.txt, clusterings/clusterings.gct,
    clusterings/cophenetic_correlation_coefficients.txt,
    clusterings/clusterings.pdf will be saved
    :param file_mark: str;
    :param n_jobs: int;
    :param distance_matrix: str or DataFrame; (n_columns, n_columns);
    distance matrix to hierarchical cluster
    :param max_std: number; threshold to clip standardized values
    :param n_clusterings: int; number of hierarchical clusterings for
    consensus clustering
    :param random_seed: int;
    :return: DataFrame, DataFrame, and Series; distance_matrix (n_samples,
    n_samples), clusterings (n_ks, n_columns), and cophenetic correlation
    coefficients (n_ks); d, cs, cccs = define_states(...)
    """

    if isinstance(matrix, str):  # Read form a .gct file
        matrix = read_gct(matrix)

    # '-0-' normalize by rows and clip values max_std standard deviation
    # away; then '0-1' normalize by rows
    matrix = normalize_2d_or_1d(
        normalize_2d_or_1d(matrix, '-0-',
                           axis=1).clip(lower=-max_std, upper=max_std),
        method='0-1',
        axis=1)

    # Hierarchical-consensus cluster
    d, cs, cccs = hierarchical_consensus_cluster(
        matrix,
        ks,
        n_jobs=n_jobs,
        d=distance_matrix,
        n_clusterings=n_clusterings,
        random_seed=random_seed)

    # Save & plot distance matrix, clusterings, and
    # cophenetic correlation coefficients
    print_log('Saving & plotting ...')

    directory_path = join(directory_path, 'clusterings{}/'.format(file_mark))
    establish_filepath(directory_path)

    d.to_csv(join(directory_path, 'distance_matrix.txt'), sep='\t')

    write_gct(cs, join(directory_path, 'clusterings.gct'))

    with PdfPages(join(directory_path, 'clusterings.pdf')) as pdf:
        # Plot distance matrix
        plot_heatmap(
            d,
            cluster=True,
            title='Distance Matrix',
            xlabel='Sample',
            ylabel='Sample',
            xticklabels=False,
            yticklabels=False)
        savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

        # Plot clusterings
        plot_heatmap(
            cs,
            axis_to_sort=1,
            data_type='categorical',
            title='Clustering per K',
            xticklabels=False)
        savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

        # Plot cophenetic correlation coefficients
        plot_points(
            sorted(cccs.keys()), [cccs[k] for k in sorted(cccs.keys())],
            title='Clustering Cophenetic-Correlation Coefficients vs. K',
            xlabel='K',
            ylabel='Cophenetic-Correlation Coefficients')
        savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

        #
        for k in ks:
            plot_heatmap(
                matrix,
                column_annotation=cs.ix[k, :],
                normalization_method='-0-',
                normalization_axis=1,
                title='{} States'.format(k),
                xlabel='Sample',
                ylabel='Component',
                xticklabels=False)
            savefig(pdf, format='pdf', dpi=DPI, bbox_inches='tight')

    return d, cs, cccs


def get_state_labels(clusterings, k):
    """
    Get state labels from clusterings.
    :param clusterings: DataFrame;
    :param k: int;
    :return: Series;
    """

    return clusterings.ix[k, :].tolist()


def make_oncogps(training_h,
                 training_states,
                 std_max=3,
                 testing_h=None,
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
                 kde_bandwidth_factor=1,
                 samples_to_plot=None,
                 component_ratio=0,
                 training_annotation=(),
                 testing_annotation=(),
                 annotation_name='',
                 annotation_type=None,
                 normalize_annotation=True,
                 annotation_scale='std',
                 highlight_high_magnitude=True,
                 annotation_ascending=True,
                 plot_samples_with_missing_annotation=False,
                 annotate_background=False,
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
                 delaunay_linewidth=1,
                 delaunay_linecolor='#220530',
                 state_colors=(),
                 bad_color='#000000',
                 background_alpha_factor=1,
                 n_contours=26,
                 contour_linewidth=0.60,
                 contour_linecolor='#262626',
                 contour_alpha=0.8,
                 state_boundary_color=None,
                 sample_markersize=23,
                 sample_markeredgewidth=0.92,
                 sample_markeredgecolor='#000000',
                 sample_name_fontsize=16,
                 sample_name_color=None,
                 legend_markersize=16,
                 legend_fontsize=16,
                 filepath=None,
                 extension='pdf',
                 dpi=DPI):
    """

    :param training_h: DataFrame; (n_nmf_component, n_samples); NMF H matrix
    :param training_states: iterable of int; (n_samples); sample states
    :param std_max: number; threshold to clip standardized values

    :param testing_h: pandas DataFrame; (n_nmf_component, n_samples);
        NMF H matrix
    :param testing_h_normalization: str or None; {'using_training_h',
        'using_testing_h', None}

    :param components: DataFrame; (n_components, 2 [x, y]); component
    coordinates
    :param equilateral: bool;

    :param informational_mds: bool; use informational MDS or not
    :param mds_seed: int; random seed for setting the coordinates of the
    multidimensional scaling

    :param n_pulls: int; [1, n_components]; number of components influencing
    a sample's coordinate
    :param power: str or number; power to raise components' influence on each
    sample
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;

    :param n_grids: int; number of grids; larger the n_grids, higher the
    resolution
    :param kde_bandwidth_factor: number; factor to multiply KDE bandwidths

    :param samples_to_plot: indexer; (n_training_samples),
    (n_testing_samples), or (n_sample_indices)
    :param component_ratio: number; number if int; percentile if float & < 1
    :param training_annotation:
    :param testing_annotation:

    :param testing_annotation: pandas Series; (n_samples); sample annotation;
    will
    color samples based on annotation
    :param annotation_name: str;
    :param annotation_type: str;
    :param normalize_annotation: bool;
    :param annotation_scale: str; {'std', 'relative'}
    :param highlight_high_magnitude: bool;
    :param annotation_ascending: bool;
    :param plot_samples_with_missing_annotation: bool;

    :param annotate_background: bool;

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

    :param state_colors: matplotlib.colors.ListedColormap,
    matplotlib.colors.LinearSegmentedColormap, or iterable;
    :param bad_color: matplotlib color;
    :param background_alpha_factor: float; [0, 1]

    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]
    :param state_boundary_color: matplotlib color;

    :param sample_markersize: number;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param sample_name_fontsize: number;
    :param sample_name_color: matplotlib color; not plotting sample if None

    :param legend_markersize: number;
    :param legend_fontsize: number;

    :param filepath: str;
    :param extension: str;
    :param dpi: number;

    :return: None
    """

    # ==========================================================================
    # Process training H matrix
    #   Set H matrix's indices to be str (better for .ix)
    #   Drop samples with all-0 values before normalization
    #   Normalize H matrix (May save normalizing parameters for normalizing
    # testing H matrix later)
    #       -0- normalize
    #       Clip values over 3 standard deviation
    #       0-1 normalize
    #   Drop samples with all-0 values after normalization
    # ==========================================================================
    training_h_initial = training_h.copy()

    if isinstance(testing_h,
                  DataFrame) and testing_h_normalization == 'using_training_h':
        normalizing_size = training_h.shape[1]
        normalizing_mean = training_h.mean(axis=1)
        normalizing_std = training_h.std(axis=1)
    else:
        normalizing_size = None
        normalizing_mean = None
        normalizing_std = None

    training_h = drop_uniform_slice_from_dataframe(training_h, 0)

    training_h = normalize_2d_or_1d(training_h, '-0-', axis=1)

    training_h = training_h.clip(lower=-std_max, upper=std_max)

    if testing_h_normalization == 'using_training_h':
        normalizing_min = training_h.min(axis=1)
        normalizing_max = training_h.max(axis=1)
    else:
        normalizing_min = None
        normalizing_max = None

    training_h = normalize_2d_or_1d(training_h, '0-1', axis=1)

    training_h = drop_uniform_slice_from_dataframe(training_h, 0)

    # ==========================================================================
    # Get training component coordinates
    #   If there are 3 components and equilateral == True, then use
    # equilateral-triangle component coordinates;
    #   else if component coordinates are specified, use them;
    #   else, compute component coordinates using Newton's Laws
    # ==========================================================================
    if equilateral and training_h.shape[0] == 3:
        print_log('Using equilateral-triangle component coordinates ...'.
                  format(components))
        components = DataFrame(
            index=['Vertex 1', 'Vertex 2', 'Vertex 3'], columns=['x', 'y'])
        components.iloc[0, :] = [0.5, sqrt(3) / 2]
        components.iloc[1, :] = [1, 0]
        components.iloc[2, :] = [0, 0]

    elif isinstance(components, DataFrame):
        print_log('Using given component coordinates ...'.format(components))
        components.index = training_h.index

    else:
        if informational_mds:
            print_log(
                'Computing component coordinates using informational distance '
                '...')
            dissimilarity = information_coefficient
        else:
            print_log(
                'Computing component coordinates using Euclidean distance ...')
            dissimilarity = 'euclidean'
        components = mds(training_h,
                         dissimilarity=dissimilarity,
                         random_state=mds_seed)
        components = DataFrame(
            components, index=training_h.index, columns=['x', 'y'])
        components = normalize_2d_or_1d(components, '0-1', axis=0)

    # ==========================================================================
    # Get training component power
    #   If n_pulls is not specified, all components pull a sample
    #   If power is not specified, compute component power by fitting (power
    # will be 1 if fitting fails)
    # ==========================================================================
    if not n_pulls:
        n_pulls = training_h.shape[0]

    if not power:
        print_log('Computing component power ...')
        if training_h.shape[0] < 4:
            print_log(
                '\tCould\'t model with Ae^(kx) + C; too few data points.')
            power = 1
        else:
            try:
                power = _compute_component_power(training_h, fit_min, fit_max,
                                                 power_min, power_max)
            except RuntimeError as e:
                power = 1
                print_log(
                    '\tCould\'t model with Ae^(kx) + C; {}; set power to be '
                    '1.'.format(e))

    # ==========================================================================
    # Compute training sample coordinates
    # Process training states
    #   Series states
    #   Keep only samples in H matrix
    # ==========================================================================
    training_samples = DataFrame(
        index=training_h.columns,
        columns=['x', 'y', 'state', 'component_ratio', 'annotation'])

    print_log(
        'Computing training sample coordinates using {} components and {:.3f} '
        'power ...'.format(n_pulls, power))
    training_samples[['x', 'y']] = _compute_sample_coordinates(
        components, training_h, n_pulls, power)

    training_samples.ix[:, 'state'] = Series(
        training_states, index=training_h.columns)

    # ==========================================================================
    # Compute training component ratios
    # ==========================================================================
    if component_ratio and 0 < component_ratio:
        print_log('Computing training component ratios ...')
        training_samples['component_ratio'] = _compute_component_ratios(
            training_h, component_ratio)

    # ==========================================================================
    # Compute grid probabilities and states
    # ==========================================================================
    print_log('Computing state grids and probabilities ...')
    state_grids, state_grids_probabilities = \
        _compute_state_grids_and_probabilities(
            training_samples, n_grids, kde_bandwidth_factor)
    # ==========================================================================
    # Process training annotation
    # ==========================================================================
    annotation_grids = annotation_grids_probabilities = None
    if len(training_annotation):
        # ======================================================================
        # Series annotation
        # Keep only samples in H matrix
        # ======================================================================
        if isinstance(training_annotation, Series):
            training_samples['annotation'] = training_annotation.ix[
                training_samples.index]
        elif len(training_annotation):
            training_samples['annotation'] = training_annotation

        # ======================================================================
        # Compute grid probabilities and annotation states
        # ======================================================================
        if annotate_background:
            print_log('Computing annotation grids and probabilities ...')
            annotation_grids, annotation_grids_probabilities = \
                _compute_annotation_grids_and_probabilities(
                    training_samples, training_annotation, n_grids)

    # ==========================================================================
    # Process testing data
    # ==========================================================================
    if isinstance(testing_h, DataFrame):
        # ======================================================================
        # Process testing H matrix
        #   Set H matrix's indices to be str (better for .ix)
        #   Drop samples with all-0 values before normalization
        #   Normalize H matrix (may use the normalizing parameters used in
        # normalizing training H matrix)
        #       -0- normalize
        #       Clip values over 3 standard deviation
        #       0-1 normalize
        #   Drop samples with all-0 values after normalization
        # ======================================================================
        if testing_h_normalization:
            # TODO: fix passing of normalizing_
            testing_h = drop_uniform_slice_from_dataframe(testing_h, 0)

            testing_h = normalize_2d_or_1d(
                testing_h,
                '-0-',
                axis=1,
                normalizing_size=normalizing_size,
                normalizing_mean=normalizing_mean,
                normalizing_std=normalizing_std)

            testing_h = testing_h.clip(lower=-std_max, upper=std_max)

            testing_h = normalize_2d_or_1d(
                testing_h,
                '0-1',
                axis=1,
                normalizing_size=normalizing_size,
                normalizing_min=normalizing_min,
                normalizing_max=normalizing_max)

            testing_h = drop_uniform_slice_from_dataframe(testing_h, 0)

        # ======================================================================
        # Compute testing sample coordinates
        # Predict testing states
        # ======================================================================
        testing_samples = DataFrame(
            index=testing_h.columns,
            columns=['x', 'y', 'state', 'component_ratio', 'annotation'])

        print_log(
            'Computing testing sample coordinates with {} components & {:.3f} '
            'power ...'.format(n_pulls, power))
        testing_samples.ix[:, ['x', 'y']] = _compute_sample_coordinates(
            components, testing_h, n_pulls, power)

        testing_samples.ix[:, 'state'] = classify(
            training_samples.ix[:, ['x', 'y']], training_states,
            testing_samples.ix[:, ['x', 'y']])
        # TODO: classify in ND
        # if not testing_h_normalization:
        #     print('No normalization.')
        #     print(training_h_initial.T.head())
        #     print(testing_h.T.head())
        #     testing_samples.ix[:, 'state'] = classify(training_h_initial.T,
        #  training_states, testing_h.T)
        # else:
        #     print('Yes normalization.')
        #     print(training_h.T.head())
        #     print(testing_h.T.head())
        #     testing_samples.ix[:, 'state'] = classify(training_h.T,
        # training_states, testing_h.T)
        testing_samples.ix[:, 'state'].T.to_csv(
            '{}.testing_states.txt'.format(filepath), sep='\t')

        # ======================================================================
        # Compute training component ratios
        # ======================================================================
        if component_ratio and 0 < component_ratio:
            print_log('Computing testing component ratios ...')
            testing_samples.ix[:,
                               'component_ratio'] = _compute_component_ratios(
                                   testing_h, component_ratio)

        # ======================================================================
        # Process testing annotation
        # ======================================================================
        if len(testing_annotation):
            # ==================================================================
            # Series annotation
            # Keep only samples in testing H matrix
            # ==================================================================
            if isinstance(testing_annotation, Series):
                testing_samples.ix[:, 'annotation'] = testing_annotation.ix[
                    testing_samples.index]
            elif len(testing_annotation):
                testing_samples.ix[:, 'annotation'] = testing_annotation

        # ======================================================================
        # Use testing
        # ======================================================================
        samples = testing_samples
    else:
        # ======================================================================
        # Use training
        # ======================================================================
        samples = training_samples

    # ==========================================================================
    # Limit samples to plot
    # Plot Onco-GPS
    # ==========================================================================
    if samples_to_plot:
        samples = samples.ix[samples_to_plot, :]

    print_log('Plotting ...')
    _plot_onco_gps(
        components=components,
        samples=samples,
        state_grids=state_grids,
        state_grids_probabilities=state_grids_probabilities,
        n_training_states=training_states.unique().size,
        annotation_name=annotation_name,
        annotation_type=annotation_type,
        normalize_annotation=normalize_annotation,
        annotation_scale=annotation_scale,
        annotation_ascending=annotation_ascending,
        highlight_high_magnitude=highlight_high_magnitude,
        plot_samples_with_missing_annotation=plot_samples_with_missing_annotation,
        annotation_grids=annotation_grids,
        annotation_grids_probabilities=annotation_grids_probabilities,
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
        colors=state_colors,
        bad_color=bad_color,
        background_alpha_factor=background_alpha_factor,
        n_contours=n_contours,
        contour_linewidth=contour_linewidth,
        contour_linecolor=contour_linecolor,
        contour_alpha=contour_alpha,
        state_boundary_color=state_boundary_color,
        sample_markersize=sample_markersize,
        sample_markeredgewidth=sample_markeredgewidth,
        sample_markeredgecolor=sample_markeredgecolor,
        sample_name_size=sample_name_fontsize,
        sample_name_color=sample_name_color,
        legend_markersize=legend_markersize,
        legend_fontsize=legend_fontsize,
        filepath=filepath,
        format=extension,
        dpi=dpi)


def _compute_component_power(h, fit_min, fit_max, power_min, power_max):
    """
    Compute component power by fitting component magnitudes of samples to the
    exponential function.
    :param h: DataFrame;
    :param fit_min: number;
    :param fit_max: number;
    :param power_min: number;
    :param power_max: number;
    :return: float; power
    """

    fit_parameters = fit_matrix(
        h, define_exponential_function, sort_matrix=True)
    k = fit_parameters[1]

    # Linear transform
    k_zero_to_one = (k - fit_min) / (fit_max - fit_min)
    k_rescaled = k_zero_to_one * (power_max - power_min) + power_min

    return k_rescaled


def _compute_sample_coordinates(component_x_coordinates, component_x_samples,
                                n_influencing_components, power):
    """
    Compute sample coordinates based on component coordinates (components
    pull samples).
    :param component_x_coordinates: DataFrame; (n_points, n_dimensions)
    :param component_x_samples: DataFrame; (n_points, n_samples)
    :param n_influencing_components: int; [1, n_components]; number of
    components pulling a sample
    :param power: number; power to raise components' pull power
    :return: DataFrame; (n_samples, n_dimension); sample_coordinates
    """

    component_x_coordinates = asarray(component_x_coordinates)

    # (n_samples, n_dimensions)
    sample_coordinates = empty(
        (component_x_samples.shape[1], component_x_coordinates.shape[1]))

    for i, (_, c
            ) in enumerate(component_x_samples.iteritems()):  # For each sample

        # Sample column
        c = asarray(c)

        # Silence components that are not pulling
        threshold = sorted(c)[-n_influencing_components]
        c[c < threshold] = 0

        # Compute coordinate in each dimension
        for d in range(component_x_coordinates.shape[1]):
            sample_coordinates[i, d] = nansum(
                c**power * component_x_coordinates[:, d]) / nansum(c**power)

    return sample_coordinates


def _compute_component_ratios(h, n):
    """
    Compute the ratio between the sum of the top-n component values and the
    sum of the rest of the component values.
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


def _compute_state_grids_and_probabilities(samples, n_grids,
                                           kde_bandwidths_factor):
    """

    :param samples:
    :param n_grids:
    :param kde_bandwidths_factor:
    :return:
    """

    grids = zeros((n_grids, n_grids), dtype=int)
    grids_probabilities = zeros((n_grids, n_grids))

    # Compute bandwidths created from all states' x & y coordinates and
    # rescale them
    bandwidths = asarray([
        bcv(asarray(samples.ix[:, 'x'].tolist()))[0],
        bcv(asarray(samples.ix[:, 'y'].tolist()))[0]
    ]) * kde_bandwidths_factor

    # Estimate kernel density for each state using bandwidth created from all
    # states' x & y coordinates
    kdes = {}
    for s in samples.ix[:, 'state'].unique():
        coordinates = samples.ix[samples.ix[:, 'state'] == s, ['x', 'y']]
        kde = kde2d(
            asarray(coordinates.ix[:, 'x'], dtype=float),
            asarray(coordinates.ix[:, 'y'], dtype=float),
            bandwidths,
            n=asarray([n_grids]),
            lims=asarray([0, 1, 0, 1]))
        kdes[s] = asarray(kde[2])

    # Assign the best KDE probability and state for each grid
    for i in range(n_grids):
        for j in range(n_grids):

            # Find the maximum (best) probability and its state
            best_state = None
            best_probability = 0
            for s, kde in kdes.items():
                p = kde[i, j]
                if best_probability < p:
                    best_state = s
                    best_probability = p

            # Assign the maximum (best) probability and its state
            grids[i, j] = best_state
            grids_probabilities[i, j] = best_probability

    return grids, grids_probabilities


# TODO: use 1 regressor instead of 2
def _compute_annotation_grids_and_probabilities(samples,
                                                annotation,
                                                n_grids,
                                                svr_kernel='rbf'):
    """

    :param samples:
    :param annotation:
    :param n_grids:
    :return:
    """

    i = ~annotation.isnull()

    annotation = normalize_2d_or_1d(annotation, '-0-')

    svr_state = SVR(kernel=svr_kernel)
    svr_probability = SVR(kernel=svr_kernel)

    svr_state.fit(
        asarray(samples.ix[i, ['x', 'y']]), asarray(annotation.ix[i]))
    svr_probability.fit(
        asarray(samples.ix[i, ['x', 'y']]), asarray(annotation.ix[i].abs()))

    grids = empty((n_grids, n_grids), dtype=int)
    grids_probability = empty((n_grids, n_grids))

    for i, fraction_i in enumerate(linspace(0, 1, n_grids)):
        for j, fraction_j in enumerate(linspace(0, 1, n_grids)):

            # Predicted annotation
            p = svr_state.predict(asarray([[fraction_i, fraction_j]]))
            if annotation.mean() <= p:
                grids[i, j] = 1
            else:
                grids[i, j] = -1

            # Predicted probability
            p = svr_probability.predict(asarray([[fraction_i, fraction_j]]))
            grids_probability[i, j] = p

    return grids, grids_probability


# ==============================================================================
# Plot Onco-GPS map
# ==============================================================================
def _plot_onco_gps(
        components, samples, state_grids, state_grids_probabilities,
        n_training_states, annotation_name, annotation_type,
        normalize_annotation, annotation_scale, annotation_ascending,
        highlight_high_magnitude, plot_samples_with_missing_annotation,
        annotation_grids, annotation_grids_probabilities, std_max, title,
        title_fontsize, title_fontcolor, subtitle_fontsize, subtitle_fontcolor,
        component_marker, component_markersize, component_markerfacecolor,
        component_markeredgewidth, component_markeredgecolor, component_names,
        component_fontsize, delaunay_linewidth, delaunay_linecolor, colors,
        bad_color, background_alpha_factor, n_contours, contour_linewidth,
        contour_linecolor, contour_alpha, state_boundary_color,
        sample_markersize, sample_markeredgewidth, sample_markeredgecolor,
        sample_name_size, sample_name_color, legend_markersize,
        legend_fontsize, filepath, format, dpi):
    """
    Plot Onco-GPS map.

    :param components: DataFrame; (n_components, 2 [x, y]);
    :param samples: DataFrame; (n_samples, 3 [x, y, state, component_ratio]);
    :param state_grids: numpy 2D array; (n_grids, n_grids)
    :param state_grids_probabilities: numpy 2D array; (n_grids, n_grids)
    :param n_training_states: int; number of training-sample states

    :param annotation_name: str;
    :param annotation_type: str;
    :param normalize_annotation: bool;
    :param annotation_scale: str; {'std', 'relative'}
    :param annotation_ascending: logical True or False
    :param highlight_high_magnitude: bool;

    :param annotation_grids: numpy 2D array; (n_grids, n_grids)
    :param annotation_grids_probabilities: numpy 2D array; (n_grids, n_grids)

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

    :param colors: matplotlib.colors.ListedColormap,
    matplotlib.colors.LinearSegmentedColormap, or iterable;
    :param bad_color: matplotlib color;
    :param background_alpha_factor: float; [0, 1]

    :param n_contours: int; set to 0 to disable drawing contours
    :param contour_linewidth: number;
    :param contour_linecolor: matplotlib color;
    :param contour_alpha: float; [0, 1]

    :param state_boundary_color: tuple; (r, g, b) where each color is between
    0 and 1

    :param sample_markersize: number;
    :param sample_markeredgewidth: number;
    :param sample_markeredgecolor: matplotlib color;
    :param sample_name_size: number;
    :param sample_name_color: None or matplotlib color; not plotting sample
    if None

    :param legend_markersize: number;
    :param legend_fontsize: number;

    :param filepath: str;
    :param format: str;
    :param dpi: number;

    :return: None
    """

    # Set up figure
    figure(figsize=FIGURE_SIZE)
    gridspec = GridSpec(10, 16)

    # Set up title ax
    ax_title = subplot(gridspec[0, :])
    ax_title.axis([0, 1, 0, 1])
    ax_title.axis('off')

    # Set up map ax
    ax_map = subplot(gridspec[0:, :12])
    ax_map.axis([0, 1, 0, 1])
    ax_map.axis('off')

    # Set up legend ax
    ax_legend = subplot(gridspec[1:, 14:])
    ax_legend.axis([0, 1, 0, 1])
    ax_legend.axis('off')

    # Plot title
    ax_map.text(
        0,
        1.16,
        title,
        fontsize=title_fontsize,
        weight='bold',
        color=title_fontcolor,
        horizontalalignment='left')
    ax_map.text(
        0,
        1.12,
        '{} samples, {} components, & {} states'.format(
            samples.shape[0], components.shape[0], n_training_states),
        fontsize=subtitle_fontsize,
        weight='bold',
        color=subtitle_fontcolor,
        horizontalalignment='left')

    # Plot component markers
    ax_map.plot(
        components.ix[:, 'x'],
        components.ix[:, 'y'],
        linestyle='',
        marker=component_marker,
        markersize=component_markersize,
        markerfacecolor=component_markerfacecolor,
        markeredgewidth=component_markeredgewidth,
        markeredgecolor=component_markeredgecolor,
        aa=True,
        clip_on=False,
        zorder=6)

    # Compute convexhull
    convexhull = ConvexHull(components)
    convexhull_region = Path(convexhull.points[convexhull.vertices])

    # Plot component labels
    if len(component_names):
        components.index = component_names
    for i in components.index:
        # Get components' x & y coordinates
        x = components.ix[i, 'x']
        y = components.ix[i, 'y']
        # Shift component label
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
        # Flip
        if convexhull_region.contains_point((components.ix[i, 'x'] + h_shift,
                                             components.ix[i, 'y'] + v_shift)):
            h_shift *= -1
            v_shift *= -1
        x += h_shift
        y += v_shift
        # Plot component label
        ax_map.text(
            x,
            y,
            i,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=component_fontsize,
            weight='bold',
            color=component_markerfacecolor,
            zorder=6)

    # Compute and plot Delaunay triangulation
    delaunay = Delaunay(components)
    ax_map.triplot(
        delaunay.points[:, 0],
        delaunay.points[:, 1],
        delaunay.simplices.copy(),
        linewidth=delaunay_linewidth,
        color=delaunay_linecolor,
        aa=True,
        clip_on=False,
        zorder=4)

    # Assign colors to states
    state_colors = assign_colors_to_states(n_training_states, colors=colors)

    # Plot background
    fraction_grids = linspace(0, 1, state_grids.shape[0])
    image = ones((*state_grids.shape, 3))

    if isinstance(annotation_grids, ndarray):
        grids = annotation_grids
        grids_probabilities = annotation_grids_probabilities

        grid_probabilities_min = grids_probabilities.min()
        grid_probabilities_max = grids_probabilities.max()
        grid_probabilities_range = grid_probabilities_max - \
                                   grid_probabilities_min

        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):

                if convexhull_region.contains_point(
                    (fraction_grids[i], fraction_grids[j])):
                    if 0 < grids[i, j]:
                        c = (1, 0, 0)
                    else:
                        c = (0, 0, 1)
                    hsv = rgb_to_hsv(*c)
                    o = (grids_probabilities[i, j] - grid_probabilities_min
                         ) / grid_probabilities_range
                    image[j, i] = hsv_to_rgb(hsv[0],
                                             min(o * background_alpha_factor,
                                                 1), hsv[2] * o + (1 - o))

        grids = state_grids

        # Plot soft contours for each state (masking points outside of Onco-GPS)
        for s in range(1, n_training_states + 1):
            mask = zeros_like(grids, dtype=bool)
            for i in range(grids.shape[0]):
                for j in range(grids.shape[1]):

                    if not convexhull_region.contains_point(
                        (fraction_grids[i],
                         fraction_grids[j])) or grids[i, j] != s:
                        mask[i, j] = True

            z = ma.array(state_grids_probabilities, mask=mask)
            ax_map.contour(
                z.transpose(),
                n_contours // 2,
                origin='lower',
                aspect='auto',
                extent=ax_map.axis(),
                corner_mask=True,
                linewidths=contour_linewidth * 2,
                colors=[state_colors[s]],
                alpha=contour_alpha / 2,
                linestyle='solid',
                aa=True,
                clip_on=False,
                zorder=2)

        # Plot boundary
        if state_boundary_color:
            for i in range(0, grids.shape[0] - 1):
                for j in range(0, grids.shape[1] - 1):

                    if convexhull_region.contains_point(
                        (fraction_grids[i], fraction_grids[j])) and (
                            grids[i, j] != grids[i + 1, j] or
                            grids[i, j] != grids[i, j + 1]):
                        image[j, i] = state_boundary_color

        ax_map.imshow(
            image,
            interpolation=None,
            origin='lower',
            aspect='auto',
            extent=ax_map.axis(),
            clip_on=False,
            zorder=1)
    else:
        grids = state_grids
        grids_probabilities = state_grids_probabilities

        grid_probabilities_min = grids_probabilities.min()
        grid_probabilities_max = grids_probabilities.max()
        grid_probabilities_range = grid_probabilities_max - \
                                   grid_probabilities_min

        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):

                if convexhull_region.contains_point(
                    (fraction_grids[i], fraction_grids[j])):
                    hsv = rgb_to_hsv(*state_colors[grids[i, j]][:3])
                    o = (grids_probabilities[i, j] - grid_probabilities_min
                         ) / grid_probabilities_range
                    image[j, i] = hsv_to_rgb(hsv[0],
                                             min(o * background_alpha_factor,
                                                 1), hsv[2] * o + (1 - o))
        ax_map.imshow(
            image,
            interpolation=None,
            origin='lower',
            aspect='auto',
            extent=ax_map.axis(),
            clip_on=False,
            zorder=1)

        # Plot contours (masking points outside of Onco-GPS)
        mask = zeros_like(grids, dtype=bool)
        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):

                if not convexhull_region.contains_point(
                    (fraction_grids[i], fraction_grids[j])):
                    mask[i, j] = True

        z = ma.array(state_grids_probabilities, mask=mask)
        ax_map.contour(
            z.transpose(),
            n_contours,
            origin='lower',
            aspect='auto',
            extent=ax_map.axis(),
            corner_mask=True,
            linewidths=contour_linewidth,
            colors=contour_linecolor,
            alpha=contour_alpha,
            aa=True,
            clip_on=False,
            zorder=2)

    # Plot state legends
    for i, s in enumerate(range(1, n_training_states + 1)):
        y = 1 - float(1 / (n_training_states + 1)) * (i + 1)
        c = state_colors[s]
        ax_legend.plot(
            -0.05,
            y,
            marker='s',
            markersize=legend_markersize,
            markerfacecolor=c,
            aa=True,
            clip_on=False)
        ax_legend.text(
            0.16,
            y,
            'State {} (n={})'.format(s, (samples.ix[:, 'state'] == s).sum()),
            fontsize=legend_fontsize,
            weight='bold',
            verticalalignment='center')

    if not samples.ix[:, 'annotation'].isnull().all():
        try:
            o_to_i = None

            # Make vector
            if normalize_annotation:
                samples.ix[:, 'a'] = normalize_2d_or_1d(
                    samples.ix[:, 'annotation'].astype(float), '-0-').clip(
                        lower=-std_max, upper=std_max)
            else:
                annotation_scale == 'relative'

            # Get annotation statistics
            a_mean = samples.ix[:, 'a'].mean()
            if annotation_scale == 'relative':
                a_min = samples.ix[:, 'a'].min()
                a_max = samples.ix[:, 'a'].max()
            elif annotation_scale == 'std':
                a_min = -std_max
                a_max = std_max
            a_range = a_max - a_min

        except ValueError:
            # Make vector by mapping object to int
            o_to_i = {}
            i_to_o = {}
            for i, o in enumerate(samples.ix[:, 'annotation'].dropna()
                                  .sort_values().unique()):
                o_to_i[o] = i
                i_to_o[i] = o
            samples.ix[:, 'a'] = samples.ix[:, 'annotation'].apply(o_to_i.get)

            # Get annotation statistics
            a_mean = samples.ix[:, 'a'].mean()
            a_min = 0
            a_max = samples.ix[:, 'a'].max()
            a_range = a_max - a_min

        # Compute and plot IC
        ic, p = compute_association_and_pvalue(samples.ix[:, 'a'],
                                               samples.ix[:, 'state'])
        ax_legend.text(
            0.5,
            1,
            '{}\nIC={:.3f} (p-val={:.3f})'.format(annotation_name, ic, p),
            fontsize=legend_fontsize * 1.26,
            weight='bold',
            horizontalalignment='center')

        if not annotation_type:  # Set annotation type
            if samples.ix[:, 'annotation'].dropna().unique().size <= 2:
                annotation_type = 'binary'
            elif samples.ix[:, 'annotation'].dropna().unique().size <= int(
                    0.5 * samples.ix[:, 'annotation'].dropna().size):
                annotation_type = 'categorical'
            else:
                annotation_type = 'continuous'

        # Set colormap
        if annotation_type == 'binary':
            cmap = CMAP_BINARY
        elif annotation_type == 'categorical':
            cmap = CMAP_CATEGORICAL_PAIRED
        else:
            cmap = CMAP_CONTINUOUS

        # Set plotting order and plot
        if highlight_high_magnitude:
            samples = samples.ix[samples.ix[:, 'a'].abs().sort_values(
                na_position='first').index, :]
        else:
            samples.sort_values(
                'a',
                ascending=annotation_ascending,
                na_position='first',
                inplace=True)
        for i, (x, y, a) in samples[['x', 'y', 'a']].iterrows():
            if isnull(a):
                if not plot_samples_with_missing_annotation:
                    continue
                else:
                    markersize = 1
                    c = bad_color
            else:
                markersize = sample_markersize
                if a_range:
                    c = cmap((a - a_min) / a_range)
                else:
                    c = cmap(0)
            ax_map.plot(
                x,
                y,
                marker='o',
                markersize=markersize,
                markerfacecolor=c,
                markeredgewidth=sample_markeredgewidth,
                markeredgecolor=sample_markeredgecolor,
                aa=True,
                clip_on=False,
                zorder=5)

        if annotation_type == 'continuous':  # Plot color bar
            cax, kw = make_axes(
                ax_legend,
                location='bottom',
                fraction=0.1,
                shrink=1,
                aspect=8,
                cmap=cmap,
                norm=Normalize(vmin=a_min, vmax=a_max),
                ticks=[a_min, a_mean, a_max])
            ColorbarBase(cax, **kw)
            decorate(ax=cax, xtick_rotation=90)

        if o_to_i:  # Plot categorical legends below the map
            for i, o in enumerate(sorted(o_to_i, reverse=True)):
                int_ = o_to_i.get(o)
                x = 1 - float(1 / (len(o_to_i) + 1)) * (i + 1)
                y = -0.1
                if a_range:
                    c = cmap((int_ - a_min) / a_range)
                else:
                    c = cmap(0)
                if 5 < len(o):
                    rotation = 90
                else:
                    rotation = 0
                ax_map.plot(
                    x,
                    y,
                    marker='o',
                    markersize=legend_markersize,
                    markerfacecolor=c,
                    aa=True,
                    clip_on=False)
                ax_map.text(
                    x,
                    y - 0.03,
                    o,
                    fontsize=legend_fontsize,
                    weight='bold',
                    color=title_fontcolor,
                    rotation=rotation,
                    horizontalalignment='center',
                    verticalalignment='top')

    else:  # Plot samples using state colors
        normalized_component_ratio = normalize_2d_or_1d(
            samples.ix[:, 'component_ratio'], '0-1')
        if not normalized_component_ratio.isnull().all():
            samples.ix[:,
                       'component_ratio_for_plot'] = normalized_component_ratio
        else:
            samples.ix[:, 'component_ratio_for_plot'] = 1

        for i, s in samples.iterrows():
            ax_map.plot(
                s.ix['x'],
                s.ix['y'],
                marker='o',
                markersize=sample_markersize,
                markerfacecolor=state_colors[s.ix['state']],
                alpha=s.ix['component_ratio_for_plot'],
                markeredgewidth=sample_markeredgewidth,
                markeredgecolor=sample_markeredgecolor,
                aa=True,
                clip_on=False,
                zorder=5)

    if sample_name_color:  # Plot sample names
        for i, s in samples.iterrows():
            ax_map.text(
                s.ix['x'],
                s.ix['y'] + 0.03,
                i,
                fontsize=sample_name_size,
                weight='bold',
                color=sample_name_color,
                horizontalalignment='center',
                zorder=7)

    if filepath:
        save_plot(filepath, file_extension=format, dpi=dpi)


def make_oncogps_in_3d(
        training_h,
        training_states,
        filepath,
        std_max=3,
        mds_seed=RANDOM_SEED,
        power=None,
        fit_min=0,
        fit_max=2,
        power_min=1,
        power_max=5,
        samples_to_plot=(),
        training_annotation=(),
        title='3D Onco-GPS',
        titlefont_size=39,
        titlefont_color='4E41D9',
        paper_bgcolor='FFFFFF',
        plot_bgcolor='000000',
        component_marker_size=26,
        component_marker_opacity=0.92,
        component_marker_line_width=2.2,
        component_marker_line_color='9017E6',
        component_marker_color='000726',
        component_textfont_size=22,
        component_textfont_color='FFFFFF',
        state_colors=(),
        sample_marker_size=13,
        sample_marker_opacity=0.92,
        sample_marker_line_width=0.19,
        sample_marker_line_color='9017E6', ):
    """

    :param training_h:
    :param training_states:
    :param filepath:
    :param std_max:
    :param mds_seed:
    :param power:
    :param fit_min:
    :param fit_max:
    :param power_min:
    :param power_max:
    :param samples_to_plot:
    :param training_annotation:
    :param title:
    :param titlefont_size:
    :param titlefont_color:
    :param paper_bgcolor:
    :param plot_bgcolor:
    :param component_marker_size:
    :param component_marker_opacity:
    :param component_marker_line_width:
    :param component_marker_line_color:
    :param component_marker_color:
    :param component_textfont_size:
    :param component_textfont_color:
    :param state_colors:
    :param sample_marker_size:
    :param sample_marker_opacity:
    :param sample_marker_line_width:
    :param sample_marker_line_color:
    :return:
    """

    # ==========================================================================
    # Process training H matrix
    #   Set H matrix's indices to be str (better for .ix)
    #   Drop samples with all-0 values before normalization
    #   Normalize H matrix (May save normalizing parameters for normalizing
    # testing H matrix later)
    #       -0- normalize
    #       Clip values over 3 standard deviation
    #       0-1 normalize
    #   Drop samples with all-0 values after normalization
    # ==========================================================================
    training_h = drop_uniform_slice_from_dataframe(training_h, 0)
    training_h = normalize_2d_or_1d(training_h, '-0-', axis=1)
    training_h = training_h.clip(lower=-std_max, upper=std_max)
    training_h = normalize_2d_or_1d(training_h, '0-1', axis=1)
    training_h = drop_uniform_slice_from_dataframe(training_h, 0)

    # ==========================================================================
    # Get training component coordinates
    # ==========================================================================
    print_log(
        'Computing component coordinates using informational distance ...')
    dissimilarity = information_coefficient
    components = mds(training_h,
                     n_components=3,
                     dissimilarity=dissimilarity,
                     random_state=mds_seed)
    components = DataFrame(
        components, index=training_h.index, columns=['x', 'y', 'z'])
    components = normalize_2d_or_1d(components, '-0-', axis=0)

    # ==========================================================================
    # Get training component power
    #   If n_pulls is not specified, all components pull a sample
    #   If power is not specified, compute component power by fitting (power
    # will be 1 if fitting fails)
    # ==========================================================================
    n_pulls = training_h.shape[0]
    if not power:
        print_log('Computing component power ...')
        if training_h.shape[0] < 4:
            print_log(
                '\tCould\'t model with Ae^(kx) + C; too few data points.')
            power = 1
        else:
            try:
                power = _compute_component_power(training_h, fit_min, fit_max,
                                                 power_min, power_max)
            except RuntimeError as e:
                power = 1
                print_log(
                    '\tCould\'t model with Ae^(kx) + C; {}; set power to be 1.'.
                    format(e))

    # ==========================================================================
    # Compute training sample coordinates
    # Process training states
    #   Series states
    #   Keep only samples in H matrix
    # ==========================================================================
    training_samples = DataFrame(
        index=training_h.columns,
        columns=['x', 'y', 'z', 'state', 'annotation'])

    print_log(
        'Computing training sample coordinates using {} components and {:.3f} '
        'power ...'.format(n_pulls, power))
    training_samples.ix[:, ['x', 'y', 'z']] = _compute_sample_coordinates(
        components, training_h, n_pulls, power)

    training_samples.ix[:, 'state'] = Series(
        training_states, index=training_h.columns)

    # ==========================================================================
    # Process training annotation
    # ==========================================================================
    if len(training_annotation):
        # ======================================================================
        # Series annotation
        # Keep only samples in H matrix
        # ======================================================================
        if isinstance(training_annotation, Series):
            training_samples.ix[:, 'annotation'] = training_annotation.ix[
                training_samples.index]
        else:
            training_samples.ix[:, 'annotation'] = training_annotation

    # ==========================================================================
    # Limit samples to plot
    # Plot 3D Onco-GPS
    # ==========================================================================
    if len(samples_to_plot):
        training_samples = training_samples.ix[samples_to_plot, :]

    print_log('Plotting ...')
    import plotly

    layout = plotly.graph_objs.Layout(
        title=title,
        titlefont=dict(
            size=titlefont_size,
            color=titlefont_color, ),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor, )

    data = []
    trace_components = plotly.graph_objs.Scatter3d(
        name='Component',
        x=components.ix[:, 'x'],
        y=components.ix[:, 'y'],
        z=components.ix[:, 'z'],
        text=components.index,
        mode='markers+text',
        marker=dict(
            size=component_marker_size,
            opacity=component_marker_opacity,
            line=dict(
                width=component_marker_line_width,
                color=component_marker_line_color, ),
            color=component_marker_color, ),
        textposition='middle center',
        textfont=dict(
            size=component_textfont_size,
            color=component_textfont_color, ))
    data.append(trace_components)

    # Assign colors to states
    state_colors = assign_colors_to_states(
        training_samples.ix[:, 'state'].unique().size, colors=state_colors)
    for s in sorted(training_samples.ix[:, 'state'].unique()):
        trace = plotly.graph_objs.Scatter3d(
            name='State {}'.format(s),
            x=training_samples.ix[training_samples.ix[:, 'state'] == s, 'x'],
            y=training_samples.ix[training_samples.ix[:, 'state'] == s, 'y'],
            z=training_samples.ix[training_samples.ix[:, 'state'] == s, 'z'],
            text=training_samples.index[training_samples.ix[:, 'state'] == s],
            mode='markers',
            marker=dict(
                size=sample_marker_size,
                opacity=sample_marker_opacity,
                line=dict(
                    width=sample_marker_line_width,
                    color=sample_marker_line_color, ),
                color='rgba{}'.format(state_colors[s]), ), )
        data.append(trace)

    fig = plotly.graph_objs.Figure(layout=layout, data=data)

    plotly.offline.plot(fig, filename=filepath)
