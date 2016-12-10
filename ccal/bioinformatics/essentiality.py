"""
Computational Cancer Analysis Library

Author:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

from os.path import join

from numpy import linspace, histogram, argmax, empty, zeros, cumsum, log
from pandas import read_csv, DataFrame, Series, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from seaborn import set_style, despine, distplot, rugplot

from ..support.plot import FIGURE_SIZE, save_plot
from ..support.log import timestamp, print_log
from ..support.file import establish_filepath


def fit_essentiality(feature_x_sample, bar_df, features=(),
                     directory_path=None, plot=True, overwrite=False, show_plot=True):
    """

    :param feature_x_sample:
    :param bar_df:
    :param features:
    :param directory_path:
    :param plot:
    :param overwrite:
    :param show_plot:
    :return: dataframe;
    """

    if isinstance(feature_x_sample, str):  # Read from a file
        feature_x_sample = read_csv(feature_x_sample, sep='\t', index_col=0)

    if not any(features):  # Use all features
        features = feature_x_sample.index

    # Result data structure
    feature_x_fit = DataFrame(index=features, columns=['n', 'df', 'shape', 'location', 'scale'])

    for i, (f_i, f_v) in enumerate(feature_x_sample.ix[features, :].iterrows()):
        print_log('Fitting {} (@{}) ...'.format(f_i, i))

        # Fit skew-t PDF on this gene
        f_v.dropna(inplace=True)
        skew_t = ACSkewT_gen()
        n = f_v.size
        df, shape, location, scale = skew_t.fit(f_v)
        feature_x_fit.ix[f_i, :] = n, df, shape, location, scale

        # Plot
        if plot:

            # Make an output filepath
            if directory_path:
                filepath = join(directory_path, 'essentiality_plots', '{}.pdf'.format(f_i))
            else:
                filepath = None

            _plot_essentiality(feature_x_sample.ix[f_i, :], get_amp_mut_del(bar_df, f_i),
                               n=n, df=df, shape=shape, location=location, scale=scale,
                               filepath=filepath, overwrite=overwrite, show_plot=show_plot)

    # Sort by shape
    feature_x_fit.sort_values('shape', inplace=True)

    if directory_path:  # Save
        filepath = join(directory_path, '{}_skew_t_fit.txt'.format(timestamp()))
        establish_filepath(filepath)
        feature_x_fit.to_csv(filepath, sep='\t')

    return feature_x_fit


def plot_essentiality(feature_x_sample, feature_x_fit, bar_df, features=None,
                      directory_path=None, overwrite=False, show_plot=True):
    """
    Make essentiality plot for each gene.
    :param feature_x_sample: DataFrame or str; (n_features, n_samples) or a filepath to a file
    :param feature_x_fit: DataFrame or str; (n_features, 5 (n, df, shape, location, scale)) or a filepath to a file
    :param bar_df: dataframe;
    :param features: iterable; (n_selected_features)
    :param overwrite: bool; overwrite the existing figure or not
    :param show_plot: bool; show plot or not
    :param directory_path: str; directory_path/essentiality_plots/feature<id>.pdf will be saved
    :return: None
    """

    if isinstance(feature_x_sample, str):  # Read from a file
        feature_x_sample = read_csv(feature_x_sample, sep='\t', index_col=0)

    if isinstance(feature_x_fit, str):  # Read from a file
        feature_x_fit = read_csv(feature_x_fit, sep='\t', index_col=0)

    if not features:  # Use all features
        features = feature_x_sample.index

    # Plot each feature
    for i, (f_i, fit) in enumerate(feature_x_fit.ix[features, :].iterrows()):
        print_log('{}: Plotting {} (@{}) ...'.format(timestamp(time_only=True), f_i, i))

        # Make an output filepath
        if directory_path:
            filepath = join(directory_path, 'essentiality_plots', '{}.pdf'.format(f_i))
        else:
            filepath = None

        # Parse fitted parameters
        n, df, shape, location, scale = fit

        _plot_essentiality(feature_x_sample.ix[f_i, :], get_amp_mut_del(bar_df, f_i),
                           n=n, df=df, shape=shape, location=location, scale=scale,
                           filepath=filepath, overwrite=overwrite, show_plot=show_plot)


def _plot_essentiality(vector, bars, n=None, df=None, shape=None, location=None, scale=None,
                       n_bins=50, n_xgrids=1000,
                       figure_size=FIGURE_SIZE, plot_vertical_extention_factor=1.26,
                       pdf_color='#20D9BA', pdf_reversed_color='#4E41D9', essentiality_index_color='#FC154F',
                       gene_fontsize=30, labels_fontsize=22,
                       bars_linewidth=2.4,
                       bar0_color='#9017E6', bar1_color='#6410A0', bar2_color='#470B72',
                       filepath=None, overwrite=True, show_plot=True):
    # ==================================================================================================================
    # Set up
    # ==================================================================================================================
    # Initialize a figure
    figure = plt.figure(figsize=figure_size)

    # Set figure styles
    set_style('ticks')
    despine(offset=9)

    # Set figure grids
    n_rows = 10
    n_rows_graph = 5
    gridspec = GridSpec(n_rows, 1)

    # Make graph ax
    ax_graph = plt.subplot(gridspec[:n_rows_graph, :])

    # Set bar axes
    ax_bar0 = plt.subplot(gridspec[n_rows_graph + 1:n_rows_graph + 2, :])
    ax_bar1 = plt.subplot(gridspec[n_rows_graph + 2:n_rows_graph + 3, :])
    ax_bar2 = plt.subplot(gridspec[n_rows_graph + 3:n_rows_graph + 4, :])
    for ax in [ax_bar1, ax_bar0, ax_bar2]:
        # TODO: remove?
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for t in ax.get_xticklines():
            t.set_visible(False)
        for t in ax.get_xticklabels():
            t.set_visible(False)
        for t in ax.get_yticklines():
            t.set_visible(False)
        for t in ax.get_yticklabels():
            t.set_visible(False)

    # ==================================================================================================================
    # Plot histogram
    # ==================================================================================================================
    distplot(vector, hist=True, bins=n_bins, kde=False, hist_kws={'linewidth': 0.92, 'alpha': 0.24, 'color': pdf_color},
             ax=ax_graph)

    # ==================================================================================================================
    # Plot skew-t fit PDF
    # ==================================================================================================================
    # Initialize a skew-t generator
    skew_t = ACSkewT_gen()

    # Set up x-grids
    xgrids = linspace(vector.min(), vector.max(), n_xgrids)

    # Generate skew-t PDF
    skew_t_pdf = skew_t.pdf(xgrids, df, shape, loc=location, scale=scale)

    # Scale skew-t PDF
    histogram_max = histogram(vector, bins=n_bins)[0].max()
    scale_factor = histogram_max / skew_t_pdf.max()
    skew_t_pdf *= scale_factor

    # Plot skew-t PDF
    line_kwargs = {'linestyle': '-', 'linewidth': 2.6}
    ax_graph.plot(xgrids, skew_t_pdf, color=pdf_color, **line_kwargs)

    # Extend plot vertically
    ax_graph.axis([vector.min(), vector.max(), 0, histogram_max * plot_vertical_extention_factor])

    # ==================================================================================================================
    # Plot reflected skew-t PDF
    # ==================================================================================================================
    # Get the reflection point
    x_reflection = xgrids[argmax(skew_t_pdf)]

    # Reflect the x-grids
    xgrids_reflected = empty(n_xgrids)
    for i in range(n_xgrids):
        a_x = xgrids[i]

        if xgrids[i] < x_reflection:  # Left of the reflection point
            xgrids_reflected[i] = a_x + 2 * abs(a_x - x_reflection)

        else:  # Right of the reflection point
            xgrids_reflected[i] = a_x - 2 * abs(a_x - x_reflection)

    # Generate skew-t PDF over reflected x-grids, and scale
    pdf_reflected = skew_t.pdf(xgrids_reflected, df, shape, loc=location, scale=scale) * scale_factor

    # Plot over the original x-grids
    ax_graph.plot(xgrids, pdf_reflected, color=pdf_reversed_color, **line_kwargs)

    # ==================================================================================================================
    # Plot essentiality index
    # ==================================================================================================================
    # Compute dx
    dx = xgrids[1] - xgrids[0]

    # Compute darea
    dareas = skew_t_pdf / sum(skew_t_pdf) * dx
    dareas_reflected = pdf_reflected / sum(pdf_reflected) * dx

    # Compute cumulative area
    if shape < 0:
        cumulative_areas = cumsum(dareas)
        cumulative_areas_reflected = cumsum(dareas_reflected)
    else:
        cumulative_areas = cumsum(dareas[::-1])[::-1]
        cumulative_areas_reflected = cumsum(dareas_reflected[::-1])[::-1]

    # TODO: Try KL divergence
    essentiality_indices = log(cumulative_areas / cumulative_areas_reflected)
    ax_graph.plot(xgrids, essentiality_indices, color=essentiality_index_color, **line_kwargs)

    # ==================================================================================================================
    # Decorate
    # ==================================================================================================================
    # Set title
    figure.text(0.5, 0.96,
                vector.name,
                fontsize=gene_fontsize, weight='bold', horizontalalignment='center')
    figure.text(0.5, 0.92,
                'n={:.2f}    df={:.2f}    shape={:.2f}    location={:.2f}    scale={:.2f}'.format(n, df, shape,
                                                                                                  location, scale),
                fontsize=gene_fontsize * 0.6, weight='bold', horizontalalignment='center')

    # Set labels
    label_kwargs = {'weight': 'bold', 'fontsize': labels_fontsize}
    ax_graph.set_xlabel('RNAi Score', **label_kwargs)
    ax_graph.set_ylabel('Frequency', **label_kwargs)

    # Set ticks
    tick_kwargs = {'size': labels_fontsize * 0.81, 'weight': 'normal'}
    for t in ax_graph.get_xticklabels():
        t.set(**tick_kwargs)
    for t in ax_graph.get_yticklabels():
        t.set(**tick_kwargs)

    # ==================================================================================================================
    # Plot bars
    # ==================================================================================================================
    bar_kwargs = {'rotation': 90, 'weight': 'bold', 'fontsize': labels_fontsize * 0.81}
    bar_specifications = {0: {'vector': bars.iloc[0, :], 'ax': ax_bar0, 'color': bar0_color},
                          1: {'vector': bars.iloc[1, :], 'ax': ax_bar1, 'color': bar1_color},
                          2: {'vector': bars.iloc[2, :], 'ax': ax_bar2, 'color': bar2_color}}

    for i, spec in bar_specifications.items():
        v = spec['vector']
        ax = spec['ax']
        c = spec['color']

        rugplot(v * vector, height=1, color=c, ax=ax, linewidth=bars_linewidth)
        ax.set_ylabel(v.name[-3:], **bar_kwargs)

    # ==================================================================================================================
    # Save
    # ==================================================================================================================
    if filepath:
        save_plot(filepath, overwrite=overwrite)

    if show_plot:
        plt.show()

    # TODO: properly close
    plt.clf()
    plt.close()


def get_amp_mut_del(gene_x_samples, gene):
    """
    Get AMP, MUT, and DEL information for a gene in the CCLE mutation file.
    :param gene_x_samples: dataframe; (n_genes, n_samplesa)
    :param gene: str; gene index used in gene_x_sample
    :return: dataframe; (3 (AMP, MUT, DEL), n_samples)
    """

    null = Series(zeros(gene_x_samples.shape[1]), index=gene_x_samples.columns)

    # Amplification
    try:
        amplifications = gene_x_samples.ix['{}_AMP'.format(gene), :]
    except KeyError:
        print('No amplification data for {}.'.format(gene))
        amplifications = null

    # Mutation
    try:
        mutations = gene_x_samples.ix['{}_MUT'.format(gene), :]
    except KeyError:
        print('No mutation data for {}.'.format(gene))
        mutations = null

    # Deletion
    try:
        deletions = gene_x_samples.ix['{}_DEL'.format(gene), :]
    except KeyError:
        print('No deletion data for {}.'.format(gene))
        deletions = null

    return concat([amplifications, mutations, deletions], axis=1).fillna(0).astype(int).T
